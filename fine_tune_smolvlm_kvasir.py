# %%
# ═══════════════════════════════════════════════════════════
#  SmolVLM2-2.2B  ×  Kvasir-VQA  —  QLoRA Fine-Tuning
# ═══════════════════════════════════════════════════════════
import torch
import os
import shutil
import time
import json
import gc

# ── SET ALL PATHS FIRST ───────────────────────────────────
cache_path = "/mnt/d/huggingface_cache"
datasets_path = os.path.join(cache_path, "datasets")
tmp_path      = "/mnt/d/tmp"
LOG_DIR = "/mnt/d/workspace/Lama_EndoscopyQA/logs/r32"

for path in [cache_path, datasets_path, tmp_path]:
    os.makedirs(path, exist_ok=True)

os.environ["HF_HOME"]            = cache_path
os.environ["HF_DATASETS_CACHE"]  = datasets_path
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
os.environ["HF_HUB_CACHE"]       = os.path.join(cache_path, "hub")
os.environ["TMPDIR"]             = tmp_path
os.environ["TEMP"]               = tmp_path
os.environ["TMP"]                = tmp_path   

#-----------------------------------
import datasets
datasets.disable_caching()

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,          # ← add this
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import Trainer, DataCollatorForSeq2Seq


# Instead of token = "hf_...", use:
HF_TOKEN = #put your token
token = os.getenv("HF_TOKEN")
#

#%% ── 1. CONFIG ─────────────────────────────────────────────
MODEL_ID   = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
OUTPUT_DIR = "/mnt/d/workspace/Lama_EndoscopyQA/smolvlm2-kvasir-finetuned"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training settings — tuned for RTX 5080 16GB
TRAIN_SAMPLES   = 5000    # use subset to start; set None for full 58k
EVAL_SAMPLES    = 100
EPOCHS          = 3
BATCH_SIZE      = 1       # per device
GRAD_ACCUM      = 16       # effective batch = 4 × 4 = 16
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 2048

# %%
print("=" * 55)
print("  SmolVLM2 × Kvasir-VQA Fine-Tuning")
print("=" * 55)


# ── 2. LOAD PROCESSOR ─────────────────────────────────────
print("\n📦 Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)


#%% ── 3. LOAD MODEL IN 4-BIT (QLoRA) ────────────────────────
# QLoRA = load model in 4-bit quantization, then add LoRA adapters on top
# This uses ~5GB VRAM instead of ~14GB — perfect for your RTX 5080

print("🧠 Loading model in 4-bit (QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # compress model to 4-bit
    bnb_4bit_quant_type="nf4",                # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,    # compute in bfloat16 for speed
    bnb_4bit_use_double_quant=True,           # quantize the quantization params too!
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)
print("✅ Model loaded in 4-bit!")



print("\n🔧 Applying LoRA adapters...")
lora_config = LoraConfig(
    r=32,                    # rank — higher = more capacity, more memory
    lora_alpha=64,           # scaling factor (usually 2× rank)
    lora_dropout=0.1,        # dropout for regularization
    target_modules=[         # which layers to apply LoRA to
        "q_proj",            # query projection (attention)
        "k_proj",            # key projection (attention)
        "v_proj",            # value projection (attention)
        "o_proj",            # output projection (attention)
        "gate_proj",         # gating (MLP)
        "up_proj",           # up projection (MLP)
        "down_proj",         # down projection (MLP)
        "embed_tokens", 
        "lm_head"
    ],
    use_dora=False,          # set True for slightly better quality (slower)
    init_lora_weights="gaussian",
    bias="none",
    task_type="CAUSAL_LM",
)


# Expected output: ~1-2% of total params are trainable


#%% ── 5. LOAD & PREPARE KVASIR-VQA DATASET ──────────────────
print("\n📂 Loading Kvasir-VQA dataset...")
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")["raw"]

# Optional: use a subset for faster experimentation
if TRAIN_SAMPLES:
    ds = ds.shuffle(seed=42)
    train_ds = ds.select(range(TRAIN_SAMPLES))
    eval_ds  = ds.select(range(TRAIN_SAMPLES, TRAIN_SAMPLES + EVAL_SAMPLES))
else:
    split    = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

print(f"✅ Train: {len(train_ds)} | Eval: {len(eval_ds)}")

#%%
# ── PROMPT TEMPLATE ───────────────────────────────────────
SYSTEM_INSTRUCTION = (
 "You are a medical AI assistant specialized in gastrointestinal endoscopy. "
    "Analyze the provided endoscopic image carefully and answer the clinical question. "
    "Answer in keywords or short phrases only. "
    "Questions based on: anatomical landmarks, pathological findings, instrument presence, image quality"
    "Examples for answers: '0', 'colonoscopy', 'none', 'yes', 'center'."
)
def preprocess_vqa_no_padding(examples):
    """
     Moves your collate_fn logic here to run ONCE and cache to disk.
     """
    images = [[img.convert("RGB")] for img in examples["image"]]
    texts = []

    for q, a in zip(examples["question"], examples["answer"]):
        messages = [
            {
                "role": "system",                          # ← add system role
                "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]
            },
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": q}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": a}]
            }
        ]
        # Apply chat template
        texts.append(processor.apply_chat_template(messages, tokenize=False))
    
    # CHANGE: Set padding=False here
    batch_inputs = processor(text=texts, images=images, return_tensors=None, padding=False)

    
        # ── MASK QUESTION TOKENS ──────────────────────────────
    labels = []
    for input_ids in batch_inputs["input_ids"]:
        label = list(input_ids)

        # Find where assistant response starts
        # Look for assistant token in the sequence
        assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        # Mask everything before the last assistant turn
        last_assistant_pos = 0
        for i, token_id in enumerate(label):
            if token_id == assistant_token_id:
                last_assistant_pos = i

        # Set question tokens to -100 (ignored in loss)
        for i in range(last_assistant_pos):
            label[i] = -100

        labels.append(label)

    batch_inputs["labels"] = labels
    return batch_inputs
#%%
# Re-map the datasets
train_ds = train_ds.map(preprocess_vqa_no_padding, batched=True, batch_size=8, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess_vqa_no_padding, batched=True, batch_size=8, remove_columns=eval_ds.column_names)



# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,          # ← add this, eval uses more memory
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # Using ratio is often safer than steps
    
    # 5080 Optimized
    bf16=True, 
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # avoids warnings
    optim="adamw_torch_fused", 
    
    # Efficiency
    dataloader_num_workers=2, 
    save_total_limit=1, # Save space in WSL
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,

    report_to="tensorboard",
    logging_dir=LOG_DIR,
    
    # save the best checkpoint based on eval loss (or your chosen metric)
    save_strategy="steps",
    save_steps=50,           # align with eval_steps
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # or your custom metric
    greater_is_better=False,
    
    # Crucial for VLMs
    remove_unused_columns=False, 
)

# %% some checks to make sure everything is in order before training
print(f"Dataset columns: {eval_ds.column_names}")

# Standard VLM expected keys:
expected_keys = ["input_ids", "attention_mask", "pixel_values", "labels"]
for key in expected_keys:
    if key in train_ds.column_names:
        print(f"✅ Found {key}")
    else:
        print(f"❌ Missing {key}!")




# %%

# MISSING STEP: You must wrap the model with the LoRA config!
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 
print("✅ LoRA adapters attached!")

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        """Clear GPU cache before evaluation to free VRAM"""
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n🧹 GPU cache cleared before eval "
              f"(Free: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB)")
# This is safer than DefaultDataCollator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer, # Use the tokenizer from your processor
    model=model,
    padding="longest",             # Pad to the longest sequence in the batch (saves VRAM)
    label_pad_token_id=-100        # Ignore padding in loss calculation
)

# Re-initialize the trainer with the new collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[ClearCacheCallback()],      # ← add callback
)

# %%
# ── 9. TRAIN ──────────────────────────────────────────────
print("\n🔥 Starting fine-tuning...")
print(f"   Model : {MODEL_ID}")
print(f"   Epochs: {EPOCHS}")
print(f"   Train : {len(train_ds)} samples")
print(f"   Batch : {BATCH_SIZE} × {GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print()

start = time.time()
trainer.train()
# # ── 10. SAVE FINE-TUNED MODEL ─────────────────────────────
print("\n💾 Saving fine-tuned LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Elapsed: {time.time() - start}s")




# Assuming 'total_time' is the variable you calculated earlier
total_time = round(time.time() - start, 2)

# Save timing to trainer_state.json
state_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
if os.path.exists(state_path):
    with open(state_path, "r") as f:
        state_data = json.load(f)
    state_data["total_training_time_seconds"] = total_time
    with open(state_path, "w") as f:
        json.dump(state_data, f, indent=2)
    print(f"💾 Training time saved to trainer_state.json")