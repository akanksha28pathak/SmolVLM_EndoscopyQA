# %%
# ═══════════════════════════════════════════════════════════
#  SmolVLM2-2.2B  ×  Kvasir-VQA  —  QLoRA Fine-Tuning
# ═══════════════════════════════════════════════════════════
import torch
import os

# 1. Define and physically create the directory first
cache_path = "/mnt/d/huggingface_cache"
if not os.path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)
    print(f"📁 Created missing directory: {cache_path}")

# 2. Set the environment variable
os.environ["HF_HOME"] = cache_path
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_path, "datasets")


#os.environ["HF_HOME"] = "/mnt/d/huggingface_cache"
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import gc
from transformers import Trainer, DataCollatorForSeq2Seq



#%% ── 1. CONFIG ─────────────────────────────────────────────
MODEL_ID   = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
OUTPUT_DIR = "./smolvlm2-kvasir-finetuned"

# Training settings — tuned for RTX 5080 16GB
TRAIN_SAMPLES   = 2000    # use subset to start; set None for full 58k
EVAL_SAMPLES    = 50
EPOCHS          = 3
BATCH_SIZE      = 2       # per device
GRAD_ACCUM      = 8       # effective batch = 4 × 4 = 16
LEARNING_RATE   = 2e-4
MAX_SEQ_LEN     = 512

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


# %%
# ── 4. LORA CONFIG ────────────────────────────────────────
# LoRA adds small trainable matrices to attention layers
# Instead of training 2.2B params, we only train ~30M — much faster!

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
def preprocess_vqa_no_padding(examples):
    """
     Moves your collate_fn logic here to run ONCE and cache to disk.
     """
    images = [[img.convert("RGB")] for img in examples["image"]]
    texts = []

    for q, a in zip(examples["question"], examples["answer"]):
        messages = [
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
    
    # Labels = input_ids (the collator will handle the -100 padding later)
    batch_inputs["labels"] = batch_inputs["input_ids"]
    return batch_inputs
#%%
# Re-map the datasets
train_ds = train_ds.map(preprocess_vqa_no_padding, batched=True, batch_size=16, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess_vqa_no_padding, batched=True, batch_size=16, remove_columns=eval_ds.column_names)
# %% previous tokenizer
# def preprocess_vqa(examples):
#     """
#     Moves your collate_fn logic here to run ONCE and cache to disk.
#     """
#     images = [[img.convert("RGB")] for img in examples["image"]]
#     texts = []

#     for q, a in zip(examples["question"], examples["answer"]):
#         messages = [
#             {
#                 "role": "user",
#                 "content": [{"type": "image"}, {"type": "text", "text": q}]
#             },
#             {
#                 "role": "assistant",
#                 "content": [{"type": "text", "text": a}]
#             }
#         ]
#         # Apply chat template
#         texts.append(processor.apply_chat_template(messages, tokenize=False))

#     # Process images and text into tensors
#     batch_inputs = processor(text=texts, images=images, return_tensors="pt", padding="longest")
    
#     # Create Labels (standard causal LM training)
#     labels = batch_inputs["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     batch_inputs["labels"] = labels
    
#     return batch_inputs

# print("Preprocessing and caching dataset to hard drive...")
# # We remove columns to keep RAM lean; only pre-computed tensors will remain
# train_ds = train_ds.map(preprocess_vqa, batched=True, batch_size=16, remove_columns=train_ds.column_names)
# eval_ds = eval_ds.map(preprocess_vqa, batched=True, batch_size=16, remove_columns=eval_ds.column_names)



# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
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
    eval_steps=25,
    
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
# Grab the first sample
sample = train_ds[0]

# Decode the input_ids back to text
decoded_text = processor.decode(sample["input_ids"], skip_special_tokens=False)

print("--- DECODED SAMPLE ---")
print(decoded_text)
print("----------------------")

# Check if labels are correctly aligned
# (Where labels are -100, the model is NOT being graded)
labels = [l for l in sample["labels"] if l != -100]
decoded_labels = processor.decode(labels, skip_special_tokens=False)
print(f"Target Answer (Labels): {decoded_labels}")

# %%
# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# Allow the model to learn new visual features for endoscopy
for name, param in model.vision_model.named_parameters():
    param.requires_grad = True

# MISSING STEP: You must wrap the model with the LoRA config!
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 
print("✅ LoRA adapters attached!")

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
)

# %%
# ── 9. TRAIN ──────────────────────────────────────────────
print("\n🔥 Starting fine-tuning...")
print(f"   Model : {MODEL_ID}")
print(f"   Epochs: {EPOCHS}")
print(f"   Train : {len(train_ds)} samples")
print(f"   Batch : {BATCH_SIZE} × {GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print()

trainer.train()
# # ── 10. SAVE FINE-TUNED MODEL ─────────────────────────────
print("\n💾 Saving fine-tuned LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)


# # ── 10. SAVE FINE-TUNED MODEL ─────────────────────────────
# print("\n💾 Saving fine-tuned LoRA adapter...")
# trainer.save_model(OUTPUT_DIR)
# processor.save_pretrained(OUTPUT_DIR)

# # ── 6. DATA COLLATOR ──────────────────────────────────────
# # This function converts each dataset row into model-ready tensors
# # It's called automatically during training for each batch

# def collate_fn(batch):
#     """
#     Converts a batch of Kvasir-VQA samples into model inputs.
#     Each sample: image + question → answer
#     We format it as a chat message so the model learns
#     to answer questions about colonoscopy images.
#     """
#     images    = []
#     texts     = []

#     for sample in batch:
#         image    = sample["image"].convert("RGB")
#         question = sample["question"]
#         answer   = sample["answer"]

#         # Format as a conversation: user asks, assistant answers
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": question}
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": answer}
#                 ]
#             }
#         ]

#         # Apply the chat template to format text correctly
#         text = processor.apply_chat_template(
#             messages,
#             add_generation_prompt=False,  # False during training (answer is included)
#             tokenize=False,
#         )
#         images.append([image])   # processor expects list of images per sample
#         texts.append(text)

#     # Tokenize everything together (image + text)
#     batch_inputs = processor(
#         text=texts,
#         images=images,
#         return_tensors="pt",
#         padding=True,
#         truncation=False,
       
#     )

#     # Labels = input_ids shifted by 1 (standard causal LM training)
#     # -100 tells the model to ignore padding tokens in loss calculation
#     labels = batch_inputs["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100

#     batch_inputs["labels"] = labels
#     return batch_inputs


# # ── 7. TRAINING ARGUMENTS ─────────────────────────────────
# print("\n⚙️  Setting up training arguments...")
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,

#     # Training schedule
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRAD_ACCUM,

#     # Optimizer
#     learning_rate=LEARNING_RATE,
#     lr_scheduler_type="cosine",       # gradually reduce LR — better convergence
#     warmup_steps=0.1,                 # warm up for first 10% of steps

#     # Memory optimizations
#     gradient_checkpointing=True,      # trade compute for memory — saves ~30% VRAM
#     bf16=True,                        # bfloat16 training — faster on RTX 5080
#     optim="paged_adamw_8bit",         # 8-bit AdamW — saves optimizer memory

#     # Evaluation & saving
#     eval_strategy="steps",
#     eval_steps=200,
#     save_strategy="steps",
#     save_steps=200,
#     save_total_limit=2,               # only keep last 2 checkpoints
#     load_best_model_at_end=True,

#     # Logging
#     logging_steps=50,
#     report_to="none",                 # set "wandb" if you want experiment tracking
#     dataloader_num_workers=0,
#     remove_unused_columns=False,      # important! keeps image column
# )


# # ── 8. TRAINER ────────────────────────────────────────────
# print("🚀 Initializing trainer...")
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=eval_ds,
#     data_collator=collate_fn,
#     processing_class=processor,
#     peft_config=lora_config,
# )


# # ── 9. TRAIN ──────────────────────────────────────────────
# print("\n🔥 Starting fine-tuning...")
# print(f"   Model : {MODEL_ID}")
# print(f"   Epochs: {EPOCHS}")
# print(f"   Train : {len(train_ds)} samples")
# print(f"   Batch : {BATCH_SIZE} × {GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective")
# print()

# trainer.train()


# # ── 10. SAVE FINE-TUNED MODEL ─────────────────────────────
# print("\n💾 Saving fine-tuned LoRA adapter...")
# trainer.save_model(OUTPUT_DIR)
# processor.save_pretrained(OUTPUT_DIR)

# %%


# print(f"✅ Fine-tuned adapter saved to: {OUTPUT_DIR}")
# print("   (Only ~50–100MB saved — just the LoRA adapter weights, not the full model)")


# # ── 11. MERGE & SAVE FULL MODEL (optional) ────────────────
# # If you want a standalone model (no need for base model at inference):
# print("\n🔀 Merging LoRA adapter into base model...")
# merged_model = model.merge_and_unload()   # merge adapter → base model
# merged_model.save_pretrained(f"{OUTPUT_DIR}-merged")
# processor.save_pretrained(f"{OUTPUT_DIR}-merged")
# print(f"✅ Merged model saved to: {OUTPUT_DIR}-merged")


