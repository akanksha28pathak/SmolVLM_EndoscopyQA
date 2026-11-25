"""
Evaluate all 4 LoRA sweep checkpoints on the same 5000 test samples
used in test_finetuned_notebook.ipynb.

Metrics: Strict Exact Match, ROUGE-1/2/L, BLEU, METEOR, per-source accuracy.

Results are saved per-checkpoint as eval_results.json and a summary
comparison is written to data/lora_sweep/sweep_eval_summary.json.
"""

import os
import re
import json
import gc
import torch
import numpy as np
import nltk
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)

# ── PATHS ──────────────────────────────────────────────────────────────────
cache_path = "/mnt/d/huggingface_cache"
os.environ["HF_HOME"]            = cache_path
os.environ["HF_DATASETS_CACHE"]  = os.path.join(cache_path, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
os.environ["HF_HUB_CACHE"]       = os.path.join(cache_path, "hub")
os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = "/mnt/d/tmp"

MODEL_ID       = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
SWEEP_BASE     = "/mnt/d/workspace/Lama_EndoscopyQA/data/lora_sweep"
CHECKPOINTS    = {
    "r16_a32":   os.path.join(SWEEP_BASE, "r16_a32"),
    "r32_a16":   os.path.join(SWEEP_BASE, "r32_a16"),
    "r64_a128":  os.path.join(SWEEP_BASE, "r64_a128"),
    "r128_a64":  os.path.join(SWEEP_BASE, "r128_a64"),
}
MAX_NEW_TOKENS = 128
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── SAME TEST SPLIT AS NOTEBOOK ────────────────────────────────────────────
TRAIN_SAMPLES  = 5000
EVAL_SAMPLES   = 100
TEST_SAMPLES   = 5000

# ── ANSWER NORMALISATION (identical to notebook) ───────────────────────────
ANSWER_SYNONYMS = {
    "none":        ["none", "not present", "absent", "nothing", "zero", "there are none", "no instruments"],
    "yes":         ["yes", "correct", "true", "present", "there is", "i can see"],
    "no":          ["false", "not", "none", "absent", "cannot see"],
    "normal":      ["normal", "no abnormality", "no findings", "unremarkable"],
    "colonoscopy": ["colonoscopy", "colon", "colonoscopic"],
}

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    for canonical, synonyms in ANSWER_SYNONYMS.items():
        for syn in synonyms:
            if re.search(rf"\b{re.escape(syn)}\b", text):
                return canonical
    return text

SYSTEM_INSTRUCTION = (
    "You are a medical AI assistant specialized in gastrointestinal endoscopy. "
    "Analyze the provided endoscopic image carefully and answer the clinical question. "
    "Answer in keywords or short phrases only. "
    "Questions based on: anatomical landmarks, pathological findings, instrument presence, image quality"
    "Examples for answers: '0', 'colonoscopy', 'none', 'yes', 'center'."
)

# ── LOAD TEST DATA ONCE ────────────────────────────────────────────────────
print("\n📂 Loading Kvasir-VQA test samples...")
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")["raw"]
ds = ds.shuffle(seed=42)
test_start = TRAIN_SAMPLES + EVAL_SAMPLES
test_end   = test_start + TEST_SAMPLES
test_ds    = ds.select(range(test_start, test_end))
print(f"✅ Testing on {len(test_ds)} samples (indices {test_start}–{test_end - 1})")

# ── METRICS HELPER ─────────────────────────────────────────────────────────
def compute_metrics(ground_truths, predictions, results):
    rouge_sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for gt, pred in zip(ground_truths, predictions):
        s = rouge_sc.score(gt, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    smoothie = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)
        for gt, pred in zip(ground_truths, predictions)
    ]
    meteor_scores = [
        meteor_score([gt.split()], pred.split())
        for gt, pred in zip(ground_truths, predictions)
    ]

    strict_matches = [
        normalize_answer(r["gt_answer"]) == normalize_answer(r["predicted"])
        for r in results
    ]
    strict_accuracy = float(np.mean(strict_matches))

    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        source_stats[r["source"]]["total"]   += 1
        source_stats[r["source"]]["correct"] += int(r["correct"])

    return {
        "total_samples":      len(results),
        "strict_exact_match": round(strict_accuracy, 4),
        "rouge1":             round(float(np.mean(r1)), 4),
        "rouge2":             round(float(np.mean(r2)), 4),
        "rougeL":             round(float(np.mean(rL)), 4),
        "bleu":               round(float(np.mean(bleu_scores)), 4),
        "meteor":             round(float(np.mean(meteor_scores)), 4),
        "per_source": {
            src: {"accuracy": round(s["correct"] / s["total"], 4), **s}
            for src, s in source_stats.items()
        },
        "per_sample_results": results,
    }

# ── MAIN EVAL LOOP ─────────────────────────────────────────────────────────
summary_all = {}

for run_name, adapter_dir in CHECKPOINTS.items():
    print(f"\n{'='*60}")
    print(f"  Evaluating: {run_name}  ({adapter_dir})")
    print(f"{'='*60}")

    # Load base model fresh for each checkpoint
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    print(f"✅ Model loaded from {adapter_dir}")

    def predict(image: Image.Image, question: str) -> str:
        messages = [
            {"role": "system",  "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
            {"role": "user",    "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[[image.convert("RGB")]],
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return processor.decode(generated, skip_special_tokens=True).strip()

    predictions, ground_truths, results = [], [], []
    for sample in tqdm(test_ds, desc=f"  {run_name}"):
        image     = sample["image"]
        question  = sample["question"]
        gt_answer = str(sample["answer"]).strip().lower()
        source    = sample["source"]
        img_id    = sample["img_id"]

        pred_answer = predict(image, question).lower()
        gt_norm     = normalize_answer(gt_answer)
        pred_norm   = normalize_answer(pred_answer)
        correct     = (gt_norm == pred_norm) or (gt_norm in pred_answer.lower())

        predictions.append(pred_answer)
        ground_truths.append(gt_answer)
        results.append({
            "img_id":    img_id,
            "source":    source,
            "question":  question,
            "gt_answer": gt_answer,
            "gt_norm":   gt_norm,
            "predicted": pred_answer,
            "pred_norm": pred_norm,
            "correct":   correct,
        })

    metrics = compute_metrics(ground_truths, predictions, results)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  RESULTS: {run_name}")
    print(f"{'='*55}")
    print(f"  Strict Exact Match : {metrics['strict_exact_match']:.2%}  ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"  ROUGE-1            : {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2            : {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L            : {metrics['rougeL']:.4f}")
    print(f"  BLEU               : {metrics['bleu']:.4f}")
    print(f"  METEOR             : {metrics['meteor']:.4f}")
    print(f"  Per-Source:")
    for src, stat in sorted(metrics["per_source"].items()):
        print(f"    {src:<30} {stat['accuracy']:.2%}  ({stat['correct']}/{stat['total']})")
    print(f"{'='*55}")

    # Save per-checkpoint results
    out_path = os.path.join(adapter_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved: {out_path}")

    # Store for summary (exclude per_sample_results to keep summary small)
    summary_all[run_name] = {k: v for k, v in metrics.items() if k != "per_sample_results"}

    # Free GPU memory before next checkpoint
    del model, base_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── SAVE COMPARISON SUMMARY ────────────────────────────────────────────────
summary_path = os.path.join(SWEEP_BASE, "sweep_eval_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary_all, f, indent=2)
print(f"\n✅ Sweep comparison saved to: {summary_path}")

# Print comparison table
print(f"\n{'='*80}")
print(f"  SWEEP COMPARISON")
print(f"{'='*80}")
header = f"{'Config':<15} {'ExactMatch':>12} {'ROUGE-1':>9} {'ROUGE-2':>9} {'ROUGE-L':>9} {'BLEU':>9} {'METEOR':>9}"
print(header)
print("-" * 80)
for run_name, m in summary_all.items():
    print(
        f"{run_name:<15} "
        f"{m['strict_exact_match']:>11.2%} "
        f"{m['rouge1']:>9.4f} "
        f"{m['rouge2']:>9.4f} "
        f"{m['rougeL']:>9.4f} "
        f"{m['bleu']:>9.4f} "
        f"{m['meteor']:>9.4f}"
    )
print("=" * 80)
