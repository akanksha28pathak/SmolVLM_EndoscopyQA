---
base_model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
library_name: peft
license: cc-by-nc-4.0
datasets:
  - SimulaMet-HOST/Kvasir-VQA
language:
  - en
pipeline_tag: image-text-to-text
tags:
  - lora
  - qlora
  - medical
  - endoscopy
  - visual-question-answering
  - smolvlm2
---

# SmolVLM2 2.2B Kvasir VQA LoRA adapter

A LoRA adapter for [SmolVLM2 2.2B Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct),
fine tuned with QLoRA on [Kvasir VQA](https://datasets.simula.no/kvasir-vqa/) for visual question answering
on gastrointestinal endoscopy images.

> **Research and educational use only. Not for clinical diagnosis or any medical decision.**

* Code, training and evaluation: https://github.com/akanksha28pathak/SmolVLM_EndoscopyQA
* Live demo: https://huggingface.co/spaces/YOUR_HF_USERNAME/endoscopy-vqa

## Training

| Setting | Value |
|---|---|
| Base model | SmolVLM2 2.2B Instruct, 4 bit NF4 (bitsandbytes), BF16 compute |
| Method | QLoRA, rank 32, alpha 64 |
| Training data | 5,000 QA pairs from Kvasir VQA (shuffled, seed 42) |
| Checkpoint | step 900 |
| Hardware | Single NVIDIA RTX 5080 (16 GB), WSL2 |

## Evaluation

Evaluated on 5,000 held out QA pairs with greedy decoding.

| Metric | Base model | This adapter |
|---|---:|---:|
| Strict exact match | 0.176 | **0.822** |
| ROUGE 1 | 0.218 | **0.874** |
| ROUGE L | 0.218 | **0.872** |
| BLEU | 0.008 | **0.202** |
| METEOR | 0.033 | **0.499** |

Exact match by image category: Normal 1.00, Ulcerative Colitis 0.92, Esophagitis 0.91, Polyps 0.73, Instrument 0.71.

## Limitations

* **Image overlap between train and test.** The split is over QA pairs, not images. Kvasir VQA has several
  questions per image, so many test images were also seen in training with different questions. The scores
  above are therefore likely optimistic for unseen images. An image level split is planned.
* Answers in Kvasir VQA are short, templated keywords. A high exact match partly reflects learning the answer format.
* Trained only on Kvasir (HyperKvasir and Kvasir Instrument) images. Performance on other endoscopy systems,
  hospitals or image qualities is unknown.
* Metrics were measured with the 4 bit base model. The public demo runs the base model in BF16, which can
  give slightly different answers.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(base_id)
model = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, "YOUR_HF_USERNAME/smolvlm2-kvasir-vqa-lora").eval()
```

Use the same system prompt as in training (see `app/inference.py` in the GitHub repo).

## License and citation

The adapter is released under **CC BY NC 4.0**, following the Kvasir VQA dataset license.
The base model is released by Hugging Face under its own license.

```bibtex
@article{gautam2024kvasirvqa,
  title   = {Kvasir-VQA: A Text-Image Pair GI Tract Dataset},
  author  = {Gautam, Sushant and Stor{\aa}s, Andrea and Midoglu, Cise and Hicks, Steven A. and
             Thambawita, Vajira and Halvorsen, P{\aa}l and Riegler, Michael A.},
  journal = {arXiv preprint arXiv:2409.01437},
  year    = {2024}
}
```
