# SmolVLM2 Endoscopy VQA API (GPU)
# Build:  docker build -t endoscopy-vqa:latest .
# Run:    docker compose up   (see docker-compose.yml for GPU + volume mounts)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache

WORKDIR /srv

# PyTorch wheels bundle the CUDA 12.8 runtime (needed for RTX 50 series / Blackwell).
# Only the NVIDIA driver on the host is required.
RUN pip install --upgrade pip && \
    pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY app/ app/

# Run as a non-root user
RUN useradd -m -u 1000 appuser && mkdir -p /hf_cache && chown appuser /hf_cache
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# One worker on purpose: each worker would load its own copy of the model onto the GPU.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
