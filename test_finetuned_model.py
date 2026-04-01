# %%
# ═══════════════════════════════════════════════════════════
#  SmolVLM2-2.2B  ×  Kvasir-VQA  —  Fine-Tuned Model Testing
# ═══════════════════════════════════════════════════════════

import os
os.environ["HF_HOME"] = "/mnt/d/huggingface_cache"

import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
import json
from tqdm import tqdm


# ── CONFIG ────────────────────────────────────────────────
BASE_MODEL_ID   = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
FINETUNED_DIR   = "./smolvlm2-kvasir-finetuned"   # your saved adapter path
TEST_SAMPLES    = 100       # number of test samples; set None for all
MAX_NEW_TOKENS  = 64        # answer is usually short for VQA
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ── 1. LOAD PROCESSOR ─────────────────────────────────────
print("📦 Loading processor...")
processor = AutoProcessor.from_pretrained(FINETUNED_DIR)


# ── 2. LOAD BASE MODEL IN 4-BIT ───────────────────────────
print("🧠 Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# ── 3. LOAD FINE-TUNED LORA ADAPTER ───────────────────────
print(f"🔧 Loading LoRA adapter from {FINETUNED_DIR}...")
model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
model.eval()
print("✅ Fine-tuned model ready!")


# ── 4. INFERENCE HELPER ───────────────────────────────────
def predict(image: Image.Image, question: str) -> str:
    """Run a single VQA inference and return the model's answer."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
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
            do_sample=False,       # greedy — deterministic for evaluation
        )

    # Decode only the newly generated tokens (strip the prompt)
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── 5. LOAD TEST DATA ─────────────────────────────────────
print("\n📂 Loading Kvasir-VQA test samples...")
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")["raw"]
ds = ds.shuffle(seed=99)   # different seed from training

if TEST_SAMPLES:
    test_ds = ds.select(range(TEST_SAMPLES))
else:
    test_ds = ds

print(f"✅ Testing on {len(test_ds)} samples")

# Map semantically equivalent answers to a canonical form
ANSWER_SYNONYMS = {
    "none":     ["no", "none", "not present", "absent", "nothing", "0", "zero", "there are none", "no instruments"],
    "yes":      ["yes", "correct", "true", "present", "there is", "i can see"],
    "no":       ["no", "false", "not", "none", "absent", "cannot see"],
    "normal":   ["normal", "no abnormality", "no findings", "unremarkable"],
    "colonoscopy": ["colonoscopy", "colon", "colonoscopic"],
}

def normalize_answer(text: str) -> str:
    """Map synonyms to a single canonical answer."""
    text = text.lower().strip()
    for canonical, synonyms in ANSWER_SYNONYMS.items():
        if any(syn in text for syn in synonyms):
            return canonical
    return text  # return as-is if no match found


# ── 6. BATCH EVALUATION ───────────────────────────────────
print("\n🔍 Running inference...")

predictions = []
ground_truths = []
results = []

for sample in tqdm(test_ds, desc="Evaluating"):
    image    = sample["image"]
    question = sample["question"]
    gt_answer = str(sample["answer"]).strip().lower()

    pred_answer = predict(image, question).lower()

    gt_norm   = normalize_answer(gt_answer)
    pred_norm = normalize_answer(pred_answer)
    correct   = (gt_norm == pred_norm) or (gt_norm in pred_answer.lower())

    predictions.append(pred_answer)
    ground_truths.append(gt_answer)
    results.append({
        "img_id":    img_id,
        "source":    source,
        "question":  question,
        "gt_answer": gt_answer,
        "predicted": predicted,
        "correct":   correct,
        
    })


# ── 7. METRICS ────────────────────────────────────────────
print("\n📊 Computing metrics...")

# Exact Match Accuracy
exact_matches = [r["exact_match"] for r in results]
accuracy = np.mean(exact_matches)

# ROUGE-L (measures overlap / partial credit)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = [
    scorer.score(gt, pred)["rougeL"].fmeasure
    for gt, pred in zip(ground_truths, predictions)
]
avg_rouge_l = np.mean(rouge_scores)

print("\n" + "=" * 45)
print("  EVALUATION RESULTS")
print("=" * 45)
print(f"  Exact Match Accuracy : {accuracy:.2%}  ({sum(exact_matches)}/{len(exact_matches)})")
print(f"  Average ROUGE-L      : {avg_rouge_l:.4f}")
print("=" * 45)


# ── 8. QUALITATIVE SAMPLES ────────────────────────────────
print("\n🔎 Sample Predictions (first 10):\n")
print(f"{'Question':<45} {'GT Answer':<25} {'Predicted':<25} {'Match'}")
print("-" * 110)
for r in results[:10]:
    match_icon = "✅" if r["exact_match"] else "❌"
    print(f"{r['question'][:44]:<45} {r['ground_truth'][:24]:<25} {r['prediction'][:24]:<25} {match_icon}")


# ── 9. SAVE RESULTS ───────────────────────────────────────
output_path = os.path.join(FINETUNED_DIR, "eval_results.json")
summary = {
    "total_samples":       len(results),
    "exact_match_accuracy": round(float(accuracy), 4),
    "avg_rouge_l":          round(float(avg_rouge_l), 4),
    "per_sample_results":   results,
}
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Full results saved to: {output_path}")


# ── 10. SINGLE IMAGE INTERACTIVE TEST ─────────────────────
# Uncomment below to test on your own image interactively

# custom_image_path = "path/to/your/endoscopy_image.jpg"
# custom_question   = "Is there any polyp in the image?"
#
# img = Image.open(custom_image_path).convert("RGB")
# answer = predict(img, custom_question)
# print(f"\nQ: {custom_question}")
# print(f"A: {answer}")
