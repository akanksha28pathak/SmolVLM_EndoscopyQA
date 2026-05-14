# 🩺 SmolVLM2 Endoscopy VQA Fine-Tuning

This repository contains the pipeline for fine-tuning **SmolVLM2-2.2B-Instruct** on the **Kvasir-VQA dataset** ([(PDF)](SmolVLM_Finetuning_Notebook.pdf).  
The goal is to evaluate and improve Vision-Language Model (VLM) performance on **medical endoscopic question-answering tasks**.

---
![Finetuning Smolvlm-2.2B on Kvasir-VQA](Block_diagram.png)

## 💻 System Configuration

Training is performed on a high-performance local consumer setup:

- **GPU:** NVIDIA GeForce RTX 5080 (16 GB VRAM)  
- **RAM:** 32 GB DDR5  
- **OS:** Windows Subsystem for Linux (WSL2)

---
## Project Structure
* [Fine-tuning Script](fine_tune_smolvlm_kvasir.py): Main training logic.
* [Baseline evalauation](eval_baseline.py): baseline evaluation.
* [Test finetuned model](test_finetuned_notebook.ipynb): finetuned model evaluation
* [Training Stats](trainer_state.json): JSON containing loss and eval metrics.
* [Results](eval_results.json) : JSON containing results

## Results
### Performance Comparison

| Metric | Baseline | Fine-tuned | Improvement |
| :--- | :---: | :---: | :---: |
| **Strict Exact Match** | 0.1756 | **0.8216** | +0.6460 |
| **ROUGE-1** | 0.2179 | **0.8742** | +0.6563 |
| **ROUGE-2** | 0.0004 | **0.1792** | +0.1788 |
| **ROUGE-L** | 0.2179 | **0.8717** | +0.6538 |
| **BLEU** | 0.0075 | **0.2022** | +0.1947 |
| **METEOR** | 0.0326 | **0.4993** | +0.4667 |

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
    ....
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
