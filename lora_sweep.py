# ═══════════════════════════════════════════════════════════
#  SmolVLM2-2.2B × Kvasir-VQA — LoRA (r, alpha) Sweep
# ═══════════════════════════════════════════════════════════
import torch
import os
import time
import json
import gc
import csv

# ── PATHS ─────────────────────────────────────────────────
cache_path    = "/mnt/d/huggingface_cache"
datasets_path = os.path.join(cache_path, "datasets")
tmp_path      = "/mnt/d/tmp"
BASE_OUTPUT   = "/mnt/d/workspace/Lama_EndoscopyQA/lora_sweep"
SUMMARY_FILE  = os.path.join(BASE_OUTPUT, "sweep_results.csv")

for path in [cache_path, datasets_path, tmp_path, BASE_OUTPUT]:
    os.makedirs(path, exist_ok=True)

os.environ["HF_HOME"]            = cache_path
os.environ["HF_DATASETS_CACHE"]  = datasets_path
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
os.environ["HF_HUB_CACHE"]       = os.path.join(cache_path, "hub")
os.environ["TMPDIR"]             = tmp_path
os.environ["TEMP"]               = tmp_path
os.environ["TMP"]                = tmp_path

import datasets as hf_datasets
hf_datasets.disable_caching()

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── SWEEP CONFIG ──────────────────────────────────────────
SWEEP_CONFIGS = [
    (16,  32),
    (64,  128),
    (32,  16),
    (128, 64),
   ]

MODEL_ID      = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
TRAIN_SAMPLES = 5000
EVAL_SAMPLES  = 100
EPOCHS        = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 16
LEARNING_RATE = 2e-4

SYSTEM_INSTRUCTION = (
    "You are a medical AI assistant specialized in gastrointestinal endoscopy. "
    "Analyze the provided endoscopic image carefully and answer the clinical question. "
    "Answer in keywords or short phrases only. "
    "Questions based on: anatomical landmarks, pathological findings, instrument presence, image quality"
    "Examples for answers: '0', 'colonoscopy', 'none', 'yes', 'center'."
)


# ── CALLBACKS ─────────────────────────────────────────────
class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"\n🧹 GPU cache cleared (Free: {free_gb:.1f} GB)")


# ── DATASET ───────────────────────────────────────────────
def load_and_preprocess_dataset(processor):
    print("📂 Loading Kvasir-VQA dataset...")
    ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")["raw"].shuffle(seed=42)
    raw_train = ds.select(range(TRAIN_SAMPLES))
    raw_eval  = ds.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))

    def preprocess_vqa(examples):
        images = [[img.convert("RGB")] for img in examples["image"]]
        texts  = []
        for q, a in zip(examples["question"], examples["answer"]):
            messages = [
                {"role": "system",    "content": [{"type": "text",  "text": SYSTEM_INSTRUCTION}]},
                {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text",  "text": a}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False))

        batch = processor(text=texts, images=images, return_tensors=None, padding=False)

        assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        labels = []
        for input_ids in batch["input_ids"]:
            label = list(input_ids)
            last_assistant_pos = 0
            for i, tid in enumerate(label):
                if tid == assistant_token_id:
                    last_assistant_pos = i
            for i in range(last_assistant_pos):
                label[i] = -100
            labels.append(label)

        batch["labels"] = labels
        return batch

    print("⚙️  Preprocessing dataset (runs once for all sweep configs)...")
    train_ds = raw_train.map(preprocess_vqa, batched=True, batch_size=8,
                             remove_columns=raw_train.column_names)
    eval_ds  = raw_eval.map(preprocess_vqa,  batched=True, batch_size=8,
                            remove_columns=raw_eval.column_names)
    print(f"✅ Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    return train_ds, eval_ds


# ── MODEL ─────────────────────────────────────────────────
def load_fresh_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return prepare_model_for_kbit_training(model)


# ── RESULTS ───────────────────────────────────────────────
def extract_best_eval_loss(output_dir: str) -> float:
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return float("nan")
    with open(state_path) as f:
        state = json.load(f)
    evals = [
        entry["eval_loss"]
        for entry in state.get("log_history", [])
        if "eval_loss" in entry
    ]
    return min(evals) if evals else float("nan")


def save_summary(results: list[dict]):
    with open(SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["r", "alpha", "best_eval_loss", "training_time_seconds"]
        )
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]):
    print("\n" + "=" * 55)
    print("  Sweep complete — results (sorted by eval loss)")
    print("=" * 55)
    print(f"{'r':>6}  {'alpha':>6}  {'best_eval_loss':>14}  {'time (s)':>10}")
    print("-" * 45)
    for row in sorted(results, key=lambda x: x["best_eval_loss"]):
        print(
            f"{row['r']:>6}  {row['alpha']:>6}  "
            f"{row['best_eval_loss']:>14.4f}  {row['training_time_seconds']:>10.0f}"
        )
    print(f"\nFull results saved to: {SUMMARY_FILE}")


# ── SINGLE RUN ────────────────────────────────────────────
def run_one(r: int, alpha: int, processor, train_ds, eval_ds) -> dict:
    tag        = f"r{r}_a{alpha}"
    output_dir = os.path.join(BASE_OUTPUT, tag)
    log_dir    = os.path.join(BASE_OUTPUT, "logs", tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Run: r={r}, alpha={alpha}  →  {tag}")
    print(f"{'='*55}")

    model = load_fresh_model()

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        use_dora=False,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
        save_total_limit=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        report_to="tensorboard",
        logging_dir=log_dir,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[ClearCacheCallback()],
    )

    t0 = time.time()
    trainer.train()
    elapsed = round(time.time() - t0, 2)

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    best_eval_loss = extract_best_eval_loss(output_dir)
    meta = {
        "r": r,
        "alpha": alpha,
        "best_eval_loss": best_eval_loss,
        "training_time_seconds": elapsed,
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ {tag} done | best eval_loss={best_eval_loss:.4f} | time={elapsed}s")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return meta


# ── MAIN ──────────────────────────────────────────────────
def main():
    print("📦 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    train_ds, eval_ds = load_and_preprocess_dataset(processor)

    results = []
    for r, alpha in SWEEP_CONFIGS:
        result = run_one(r, alpha, processor, train_ds, eval_ds)
        results.append(result)
        save_summary(results)  # checkpoint after every run

    print_summary(results)


if __name__ == "__main__":
    main()
