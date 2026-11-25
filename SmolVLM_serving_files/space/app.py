"""Gradio demo for Hugging Face Spaces (ZeroGPU).

inference.py is copied next to this file by scripts/deploy_space.py,
so the Space and the FastAPI service share the same prompt and decoding.

On ZeroGPU the base model runs in bf16 (bitsandbytes 4-bit is not used there).
"""
import os
from pathlib import Path

import gradio as gr
import spaces

from inference import InferenceConfig, VQAPredictor

ADAPTER_REPO = os.getenv("ADAPTER_REPO", "YOUR_HF_USERNAME/smolvlm2-kvasir-vqa-lora")

predictor = VQAPredictor(
    config=InferenceConfig(
        adapter_path=ADAPTER_REPO,
        model_version=os.getenv("MODEL_VERSION", "checkpoint-900"),
        load_in_4bit=False,
        device="cuda",
    )
)
predictor.load()  # model moves to "cuda" at import; ZeroGPU attaches a real GPU per call


@spaces.GPU(duration=30)
def answer(image, question):
    if image is None:
        raise gr.Error("Please upload an endoscopy image.")
    question = (question or "").strip()
    if not question:
        raise gr.Error("Please type a question.")
    if len(question) > 300:
        raise gr.Error("Question is too long (max 300 characters).")
    text, latency_ms = predictor.predict(image, question)
    return text, f"{latency_ms:.0f} ms"


SAMPLE_QUESTIONS = [
    "Is there a polyp?",
    "What type of procedure is the image taken from?",
    "Is there an instrument?",
    "How many polyps are in the image?",
    "Where in the image is the abnormality?",
    "Is there text?",
]

with gr.Blocks(title="Endoscopy VQA (SmolVLM2)") as demo:
    gr.Markdown(
        "# 🩺 Endoscopy Visual Question Answering\n"
        "SmolVLM2 2.2B fine tuned with QLoRA on the Kvasir VQA dataset. "
        "Upload a GI endoscopy image and ask a short question.\n\n"
        "**⚠️ Research and educational use only. Not for clinical diagnosis.** "
        "The model was trained on Kvasir images only and may be wrong on other images."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Endoscopy image")
            question = gr.Dropdown(
                choices=SAMPLE_QUESTIONS,
                value=SAMPLE_QUESTIONS[0],
                allow_custom_value=True,
                label="Question (pick one or type your own)",
            )
            btn = gr.Button("Ask", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Answer")
            latency = gr.Textbox(label="Inference time")

    examples_dir = Path(__file__).parent / "examples"
    if examples_dir.exists():
        files = sorted(p for p in examples_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if files:
            gr.Examples(examples=[[str(f), SAMPLE_QUESTIONS[0]] for f in files], inputs=[image, question])

    gr.Markdown(
        "Model: [SmolVLM2 2.2B Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) + LoRA adapter. "
        "Data: [Kvasir VQA](https://datasets.simula.no/kvasir-vqa/) (CC BY NC 4.0). "
        "Code: [GitHub](https://github.com/akanksha28pathak/SmolVLM_EndoscopyQA)"
    )
    btn.click(answer, inputs=[image, question], outputs=[out, latency])

demo.launch()
