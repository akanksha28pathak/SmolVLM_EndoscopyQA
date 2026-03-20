# 🩺 SmolVLM2 Endoscopy VQA Fine-Tuning

This repository contains the pipeline for fine-tuning **SmolVLM2-2.2B-Instruct** on the **Kvasir-VQA dataset**.  
The goal is to evaluate and improve Vision-Language Model (VLM) performance on **medical endoscopic question-answering tasks**.

---

## 💻 System Configuration

Training is performed on a high-performance local consumer setup:

- **GPU:** NVIDIA GeForce RTX 5080 (16 GB VRAM)  
- **RAM:** 32 GB DDR5  
- **OS:** Windows Subsystem for Linux (WSL2)

---

## 🛠️ Environment Setup & Optimization

### 1. WSL2 Resource Allocation

By default, WSL2 may limit resource access. To ensure sufficient resources for training:

1. Press `Win + R`, type `.` and press Enter  
2. Create (or edit) a file named `.wslconfig`  
3. Add the following configuration:

```ini
[wsl2]
memory=24GB
processors=8
```

4. Restart WSL:

```bash
wsl --shutdown
```

---

### 2. Storage Management

VLM datasets and checkpoints are large. To avoid filling up the `C:` drive, redirect the Hugging Face cache to another drive:

```python
import os
os.environ["HF_HOME"] = "/mnt/d/huggingface_cache"
```

---

## 🚀 Optimization Strategy

### 4-Bit Quantization & BF16

To fit the model and training pipeline within **16GB VRAM**, we use:

- **BitsAndBytes 4-bit quantization**
- **BFloat16 (BF16)** precision

This combination:
- Reduces memory footprint significantly  
- Maintains strong numerical stability and performance  

---

## ⚙️ Data Preprocessing & Caching

To maximize GPU utilization and avoid CPU bottlenecks:

- Dataset is **preprocessed and cached before training**
- **Padding is disabled during `.map()`**
- Dynamic padding is handled later by the **DataCollator**

### 🔑 Key Optimization

- `padding=False`
- `return_tensors=None`

This reduces memory usage and keeps cache size manageable.

---

### 📌 Preprocessing Function

```python
def preprocess_vqa_no_padding(examples):
    """
    Applies chat templates and processes images for storage.
    """
    images = [[img.convert("RGB")] for img in examples["image"]]
    texts = []

    for q, a in zip(examples["question"], examples["answer"]):
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False))

    # Process without padding to keep cache size manageable
    batch_inputs = processor(
        text=texts,
        images=images,
        return_tensors=None,
        padding=False
    )

    # Labels match input_ids; collator will handle padding masking
    batch_inputs["labels"] = batch_inputs["input_ids"]

    return batch_inputs
```

---

### 📌 Dataset Mapping & Caching

```python
train_ds = train_ds.map(
    preprocess_vqa_no_padding,
    batched=True,
    batch_size=16,
    remove_columns=train_ds.column_names
)

eval_ds = eval_ds.map(
    preprocess_vqa_no_padding,
    batched=True,
    batch_size=16,
    remove_columns=eval_ds.column_names
)
```

---

## 📊 Dataset & Model References

- **Model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct`  
- **Dataset:** `SimulaMet-HOST/Kvasir-VQA`

---

## ✅ Summary of Key Design Choices

- Efficient training on **consumer GPU (16GB VRAM)**
- Memory optimization via:
  - 4-bit quantization
  - BF16 precision
- Smart preprocessing:
  - No padding during mapping
  - Dynamic padding during training
- Disk-based caching for scalability
