# ═══════════════════════════════════════════════════════════════════════════
#  SmolVLM2-2.2B  ×  Kvasir-VQA  —  BASELINE EVALUATION (No Fine-Tuning)
#
#  Bugs fixed vs. test_baselinemodel.ipynb:
#   1. WRONG TEST SPLIT  — old: TRAIN=2000+EVAL=50 → start=2050
#                          fix: TRAIN=5000+EVAL=100 → start=5100
#                          (matches exactly the split used in test_notebook.ipynb)
#   2. SYSTEM PROMPT     — old: injected inline into user message
#                          fix: proper role="system" message (same structure as
#                          fine-tuned eval) so prompt handling is identical
#   3. COMMENTED-OUT SAVE — per-source breakdown & JSON save were never run;
#                          now always executed, output matches eval_results.json
#                          structure for direct comparison
# ═══════════════════════════════════════════════════════════════════════════

import os
os.environ["HF_HOME"]            = "/mnt/d/huggingface_cache"
os.environ["HF_DATASETS_CACHE"]  = "/mnt/d/huggingface_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/d/huggingface_cache/transformers"
os.environ["HF_HUB_CACHE"]       = "/mnt/d/huggingface_cache/hub"

import re
import json
import torch
import numpy as np
import nltk
from PIL import Image
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from rouge_score import rouge_scorer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
OUTPUT_DIR     = "/mnt/d/workspace/Lama_EndoscopyQA/smolvlm2-baseline-results"
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "baseline_eval_results.json")

# BUG FIX 1: must match the fine-tuned eval split exactly
# Fine-tuned (test_notebook.ipynb): TRAIN=5000, EVAL=100 → test starts at 5100
TRAIN_SAMPLES  = 5000
EVAL_SAMPLES   = 100
TEST_SAMPLES   = 5000          # same count as fine-tuned eval
MAX_NEW_TOKENS = 128
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. PROCESSOR ──────────────────────────────────────────────────────────
print("Loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

# ── 2. BASE MODEL IN 4-BIT (no LoRA) ─────────────────────────────────────
print("Loading BASE model in 4-bit (no adapter)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()
print("Base model ready — no LoRA adapter loaded.")

# ── 3. SYSTEM PROMPT & INFERENCE ──────────────────────────────────────────
# BUG FIX 2: use role="system" message (same structure as fine-tuned eval),
# not inline injection into the user message.
SYSTEM_INSTRUCTION = (
    "You are a medical AI assistant specialized in gastrointestinal endoscopy. "
    "Analyze the provided endoscopic image carefully and answer the clinical question. "
    "Answer in keywords or short phrases only. "
    "Questions based on: anatomical landmarks, pathological findings, instrument presence, image quality. "
    "Examples for answers: '0', 'colonoscopy', 'none', 'yes', 'center'."
)

def predict(image: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
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
            do_sample=False,   # greedy — deterministic for evaluation
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# ── 4. LOAD SAME TEST SPLIT AS FINE-TUNED EVAL ───────────────────────────
print("\nLoading Kvasir-VQA test samples...")
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")["raw"]
ds = ds.shuffle(seed=42)          # identical seed to training script

# BUG FIX 1: correct index arithmetic
test_start = TRAIN_SAMPLES + EVAL_SAMPLES    # 5100
test_end   = test_start + TEST_SAMPLES       # 10100
test_ds    = ds.select(range(test_start, test_end))
print(f"Testing on {len(test_ds)} samples (indices {test_start}–{test_end - 1})")

# ── 5. ANSWER NORMALISATION (identical to test_notebook.ipynb) ────────────
ANSWER_SYNONYMS = {
    "none":        ["none", "not present", "absent", "nothing", "zero",
                    "there are none", "no instruments"],
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

# ── 6. BATCH EVALUATION ───────────────────────────────────────────────────
print("\nRunning baseline inference...")
predictions   = []
ground_truths = []
results       = []

for sample in tqdm(test_ds, desc="Baseline eval"):
    image     = sample["image"]
    question  = sample["question"]
    gt_answer = str(sample["answer"]).strip().lower()
    source    = sample["source"]
    img_id    = sample["img_id"]

    pred_answer = predict(image, question).lower()

    gt_norm   = normalize_answer(gt_answer)
    pred_norm = normalize_answer(pred_answer)
    correct   = (gt_norm == pred_norm) or (gt_norm in pred_answer)

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

# ── 7. METRICS ────────────────────────────────────────────────────────────
# BUG FIX 3: all metrics now always computed + saved (were commented out)
print("\nComputing metrics...")

# Strict exact match (normalised) — primary metric for comparison
strict_matches  = [
    normalize_answer(r["gt_answer"]) == normalize_answer(r["predicted"])
    for r in results
]
strict_accuracy = float(np.mean(strict_matches))

# ROUGE
_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
for gt, pred in zip(ground_truths, predictions):
    s = _scorer.score(gt, pred)
    rouge1_scores.append(s["rouge1"].fmeasure)
    rouge2_scores.append(s["rouge2"].fmeasure)
    rougeL_scores.append(s["rougeL"].fmeasure)

# BLEU
smoothie = SmoothingFunction().method1
bleu_scores = [
    sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)
    for gt, pred in zip(ground_truths, predictions)
]

# METEOR
meteor_scores = [
    meteor_score([gt.split()], pred.split())
    for gt, pred in zip(ground_truths, predictions)
]

# Per-source breakdown
source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    source_stats[r["source"]]["total"]   += 1
    source_stats[r["source"]]["correct"] += int(
        normalize_answer(r["gt_answer"]) == normalize_answer(r["predicted"])
    )

# ── 8. PRINT SUMMARY ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BASELINE RESULTS  (no fine-tuning)")
print("=" * 60)
print(f"  Strict Exact Match : {strict_accuracy:.2%}  ({sum(strict_matches)}/{len(strict_matches)})")
print(f"  ROUGE-1            : {np.mean(rouge1_scores):.4f}")
print(f"  ROUGE-2            : {np.mean(rouge2_scores):.4f}")
print(f"  ROUGE-L            : {np.mean(rougeL_scores):.4f}")
print(f"  BLEU               : {np.mean(bleu_scores):.4f}")
print(f"  METEOR             : {np.mean(meteor_scores):.4f}")
print("-" * 60)
print("  Per-Source Breakdown:")
for src, stat in sorted(source_stats.items()):
    acc = stat["correct"] / stat["total"]
    print(f"    {src:<30} {acc:.2%}  ({stat['correct']}/{stat['total']})")
print("=" * 60)

# ── 9. SAVE — matches eval_results.json structure exactly ─────────────────
summary = {
    "model":              BASE_MODEL_ID,
    "adapter":            "none (baseline)",
    "test_split":         f"seed=42, indices {test_start}–{test_end-1}",
    "total_samples":      len(results),
    "strict_exact_match": round(strict_accuracy, 4),
    "rouge1":             round(float(np.mean(rouge1_scores)), 4),
    "rouge2":             round(float(np.mean(rouge2_scores)), 4),
    "rougeL":             round(float(np.mean(rougeL_scores)), 4),
    "bleu":               round(float(np.mean(bleu_scores)), 4),
    "meteor":             round(float(np.mean(meteor_scores)), 4),
    "per_source": {
        src: {
            "accuracy": round(s["correct"] / s["total"], 4),
            "correct":  s["correct"],
            "total":    s["total"],
        }
        for src, s in source_stats.items()
    },
    "per_sample_results": results,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {OUTPUT_FILE}")

# ── 10. SIDE-BY-SIDE COMPARISON WITH FINE-TUNED ───────────────────────────
ft_path = "/mnt/d/workspace/Lama_EndoscopyQA/smolvlm2-kvasir-finetuned/eval_results.json"
if os.path.exists(ft_path):
    with open(ft_path) as f:
        ft = json.load(f)

    print("\n" + "=" * 65)
    print("  BASELINE  vs.  FINE-TUNED  (same 5000-sample test split)")
    print("=" * 65)
    metrics = [
        ("Strict Exact Match", "strict_exact_match"),
        ("ROUGE-1",            "rouge1"),
        ("ROUGE-2",            "rouge2"),
        ("ROUGE-L",            "rougeL"),
        ("BLEU",               "bleu"),
        ("METEOR",             "meteor"),
    ]
    print(f"  {'Metric':<22} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 65)
    for label, key in metrics:
        b = summary.get(key, 0)
        f_val = ft.get(key, 0)
        delta = f_val - b
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"  {label:<22} {b:>10.4f} {f_val:>12.4f}   {arrow}{abs(delta):.4f}")
    print("-" * 65)
    print("  Per-Source Accuracy:")
    for src in sorted(summary["per_source"]):
        b_acc  = summary["per_source"][src]["accuracy"]
        ft_acc = ft.get("per_source", {}).get(src, {}).get("accuracy", 0)
        delta  = ft_acc - b_acc
        arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"    {src:<28} {b_acc:.4f} {ft_acc:>12.4f}   {arrow}{abs(delta):.4f}")
    print("=" * 65)
