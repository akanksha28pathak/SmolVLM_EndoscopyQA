"""Upload ONLY the LoRA adapter files (not optimizer/scheduler state) to the Hugging Face Hub.

Usage (inside WSL, after `hf auth login`):
    python scripts/push_adapter_to_hub.py \
        --checkpoint /mnt/d/workspace/Lama_EndoscopyQA/smolvlm2-kvasir-finetuned/checkpoint-900 \
        --repo YOUR_HF_USERNAME/smolvlm2-kvasir-vqa-lora
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

MODEL_CARD = Path(__file__).resolve().parent.parent / "MODEL_CARD.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ckpt / f).exists():
            raise SystemExit(f"Missing {f} in {ckpt}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        commit_message="Upload LoRA adapter (checkpoint-900)",
    )
    api.upload_file(path_or_fileobj=str(MODEL_CARD), path_in_repo="README.md", repo_id=args.repo)
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
