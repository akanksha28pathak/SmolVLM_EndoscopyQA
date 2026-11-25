"""Shared inference logic for SmolVLM2 + Kvasir-VQA LoRA adapter.

Used by both the FastAPI service (app/main.py) and the Hugging Face
Gradio Space (space/app.py). Heavy imports (torch, transformers, peft)
happen inside functions so the API layer can be tested without a GPU.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from PIL import Image

# Must match the prompt used in fine_tune_smolvlm_kvasir.py EXACTLY
# (including the missing space before "Examples"), otherwise outputs drift.
SYSTEM_INSTRUCTION = (
    "You are a medical AI assistant specialized in gastrointestinal endoscopy. "
    "Analyze the provided endoscopic image carefully and answer the clinical question. "
    "Answer in keywords or short phrases only. "
    "Questions based on: anatomical landmarks, pathological findings, instrument presence, image quality"
    "Examples for answers: '0', 'colonoscopy', 'none', 'yes', 'center'."
)


@dataclass
class InferenceConfig:
    base_model_id: str = os.getenv("BASE_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    # Local folder (e.g. /models/adapter) OR a Hugging Face Hub repo id.
    adapter_path: str = os.getenv("ADAPTER_PATH", "/models/adapter")
    model_version: str = os.getenv("MODEL_VERSION", "checkpoint-900")
    # 4-bit matches how the reported metrics were produced. Set LOAD_IN_4BIT=0 for bf16.
    load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "1") == "1"
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "16"))
    device: str = os.getenv("DEVICE", "cuda")


@dataclass
class VQAPredictor:
    config: InferenceConfig = field(default_factory=InferenceConfig)
    model: object = None
    processor: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cfg = self.config
        self.processor = AutoProcessor.from_pretrained(cfg.base_model_id)

        kwargs = {"dtype": torch.bfloat16}
        if cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}

        base = AutoModelForImageTextToText.from_pretrained(cfg.base_model_id, **kwargs)
        if not cfg.load_in_4bit:
            base = base.to(cfg.device)

        model = PeftModel.from_pretrained(base, cfg.adapter_path)
        model.eval()
        self.model = model

    def warmup(self) -> None:
        """One dummy forward pass so the first real request is not slow."""
        self.predict(Image.new("RGB", (384, 384), (128, 64, 64)), "Is there a polyp?")

    def predict(self, image: Image.Image, question: str) -> tuple[str, float]:
        """Return (answer, latency_ms). Thread safe: one generation at a time on the GPU."""
        import torch

        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        with self._lock:
            start = time.perf_counter()
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=prompt, images=[[image.convert("RGB")]], return_tensors="pt"
            ).to(self.model.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False
                )
            generated = output_ids[0][inputs["input_ids"].shape[-1]:]
            answer = self.processor.decode(generated, skip_special_tokens=True).strip()
            latency_ms = (time.perf_counter() - start) * 1000
        return answer, latency_ms
