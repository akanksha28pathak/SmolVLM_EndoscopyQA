🩺 **SmolVLM2 Endoscopy VQA Fine-Tuning**
This repository contains the pipeline for fine-tuning SmolVLM2-2.2B-Instruct on the Kvasir-VQA dataset. The goal is to evaluate and improve Vision-Language Model (VLM) performance on medical endoscopic question-answering tasks.

💻 **System Configuration**
Training is performed on a high-performance local consumer setup:

GPU: NVIDIA GeForce RTX 5080 (16 GB VRAM)

RAM: 32 GB DDR5

OS: Windows Subsystem for Linux (WSL2)

🛠️ **Environment Setup & Optimization**
1. WSL2 Resource Allocation
By default, WSL2 may limit resource access. To ensure the trainer has enough overhead, create or edit your .wslconfig file:

Press Win + R, type ., and press Enter.

Create a file named .wslconfig (if it doesn't exist).

Add the following configuration to allocate 24GB of System RAM:

'''
[wsl2]
memory=24GB
processors=8
'''

Restart WSL by running wsl --shutdown in PowerShell.

2. Storage Management
VLM datasets and checkpoints are large. To prevent the primary C: drive from filling up, the Hugging Face cache is redirected to a secondary drive:

Python
'''
import os
os.environ["HF_HOME"] = "/mnt/d/huggingface_cache"
'''

🚀 **Optimization Strategy**
4-Bit Quantization & BF16
To fit the model and its gradients into the 16GB VRAM of the RTX 5080, we load the model using BitsAndBytes 4-bit quantization and the BFloat16 (Brain Floating Point) data type. This maintains high precision while significantly reducing the memory footprint.

**Data Preprocessing & Caching**
To maximize GPU throughput and avoid CPU bottlenecks, the dataset is preprocessed and cached to the disk before training starts.

Key Change: We disable padding during the .map() phase (padding=False) and set return_tensors=None. This allows the DataCollator to handle dynamic padding during the training loop, saving significant System RAM.

Python
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
    batch_inputs = processor(text=texts, images=images, return_tensors=None, padding=False)
    
    # Labels match input_ids; the Collator handles -100 masking for padding later
    batch_inputs["labels"] = batch_inputs["input_ids"]
    return batch_inputs

# Map and Cache
train_ds = train_ds.map(preprocess_vqa_no_padding, batched=True, batch_size=16, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess_vqa_no_padding, batched=True, batch_size=16, remove_columns=eval_ds.column_names)

📊 **Dataset Reference**
Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct

Dataset: SimulaMet-HOST/Kvasir-VQA
