"""Create/update the Gradio Space and copy the shared inference code into it.

Usage (inside WSL, after `hf auth login`):
    python scripts/deploy_space.py \
        --space YOUR_HF_USERNAME/endoscopy-vqa \
        --adapter-repo YOUR_HF_USERNAME/smolvlm2-kvasir-vqa-lora

Afterwards, in the Space's Settings, set Hardware to "ZeroGPU".
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--adapter-repo", required=True)
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(args.space, "ADAPTER_REPO", args.adapter_repo)
    api.upload_folder(folder_path=str(ROOT / "space"), repo_id=args.space, repo_type="space",
                      commit_message="Update Space app")
    api.upload_file(path_or_fileobj=str(ROOT / "app" / "inference.py"), path_in_repo="inference.py",
                    repo_id=args.space, repo_type="space", commit_message="Sync shared inference code")
    print(f"Done: https://huggingface.co/spaces/{args.space}")
    print("Next: Space Settings -> Hardware -> ZeroGPU")


if __name__ == "__main__":
    main()
