# Serving and deployment guide

Run every command inside **WSL (Ubuntu)** unless it says PowerShell.

## Part 1. Local API with Docker (GPU)

### 1.1 One time: let Docker use the GPU (NVIDIA Container Toolkit)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test: should print your RTX 5080
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### 1.2 Build and run

```bash
cd /mnt/d/workspace/SmolVLM_EndoscopyQA      # your clone of this repo
docker compose up --build
```

The first build downloads PyTorch (several GB) and takes a while. Wait until the log shows `Model ready`.
The base model is read from your existing cache at `/mnt/d/huggingface_cache`.
The adapter is mounted from `.../smolvlm2-kvasir-finetuned/checkpoint-900`.
To use different folders, create a `.env` file next to `docker-compose.yml`:

```
ADAPTER_DIR=/mnt/d/workspace/Lama_EndoscopyQA/smolvlm2-kvasir-finetuned/checkpoint-900
HF_CACHE_DIR=/mnt/d/huggingface_cache
```

### 1.3 Try it

* Browser: http://localhost:8000/docs → `POST /predict` → **Try it out**
* Command line:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/endoscopy.jpg" \
  -F "question=Is there a polyp?"
```

### 1.4 Run the tests (no GPU needed)

```bash
pip install fastapi python-multipart httpx pytest pillow
pytest -q
```

## Part 2. Public demo on Hugging Face Spaces (ZeroGPU)

Free ZeroGPU hosting needs a verified Hugging Face account that is at least 30 days old.

### 2.1 Log in

```bash
pip install -U huggingface_hub
hf auth login          # paste a token with "write" access from huggingface.co/settings/tokens
```

### 2.2 Upload the adapter (only adapter files, not optimizer state)

```bash
python scripts/push_adapter_to_hub.py \
  --checkpoint /mnt/d/workspace/Lama_EndoscopyQA/smolvlm2-kvasir-finetuned/checkpoint-900 \
  --repo YOUR_HF_USERNAME/smolvlm2-kvasir-vqa-lora
```

Before running, replace `YOUR_HF_USERNAME` in `MODEL_CARD.md` too.

### 2.3 (Optional) Add example images

Put 3 or 4 Kvasir VQA test images in `space/examples/` (jpg or png). They appear as clickable examples.
They are CC BY NC 4.0, and the Space README already credits the dataset.

### 2.4 Deploy the Space

```bash
python scripts/deploy_space.py \
  --space YOUR_HF_USERNAME/endoscopy-vqa \
  --adapter-repo YOUR_HF_USERNAME/smolvlm2-kvasir-vqa-lora
```

Then open the Space → **Settings** → **Space hardware** → choose **ZeroGPU**.
The first build takes about 5 to 10 minutes. Logs are under the **Logs** tab.

### 2.5 Sanity check

Compare a few Space answers with your local API on the same image and question.
The Space runs the base model in BF16 instead of 4 bit, so a small number of answers may differ.
