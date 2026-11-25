"""FastAPI service for SmolVLM2 Kvasir-VQA.

Run locally:   uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs:          http://localhost:8000/docs

RESEARCH AND EDUCATIONAL USE ONLY. NOT FOR CLINICAL DIAGNOSIS.
"""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.inference import VQAPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("endoscopy-vqa")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024
MAX_QUESTION_CHARS = 300
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
DISCLAIMER = "Research and educational use only. Not for clinical diagnosis."


class PredictResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float
    model_version: str
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


def create_app(predictor: VQAPredictor | None = None, load_model: bool = True) -> FastAPI:
    predictor = predictor or VQAPredictor()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model and not predictor.is_loaded:
            log.info("Loading %s + adapter %s ...", predictor.config.base_model_id, predictor.config.adapter_path)
            predictor.load()
            predictor.warmup()
            log.info("Model ready (%s)", predictor.config.model_version)
        yield

    app = FastAPI(
        title="SmolVLM2 Endoscopy VQA API",
        description=(
            "Visual question answering on GI endoscopy images using SmolVLM2-2.2B "
            "fine-tuned with QLoRA on Kvasir-VQA. " + DISCLAIMER
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.predictor = predictor

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        p: VQAPredictor = request.app.state.predictor
        if not p.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        return HealthResponse(status="ok", model_loaded=True, model_version=p.config.model_version)

    # Sync endpoint on purpose: FastAPI runs it in a threadpool, so GPU work
    # does not block the event loop; the predictor lock serialises GPU access.
    @app.post("/predict", response_model=PredictResponse)
    def predict(
        request: Request,
        image: UploadFile = File(..., description="Endoscopy image (jpg, png, webp, bmp)"),
        question: str = Form(..., description="Question about the image, e.g. 'Is there a polyp?'"),
    ):
        p: VQAPredictor = request.app.state.predictor
        if not p.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        question = question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(status_code=422, detail=f"Question longer than {MAX_QUESTION_CHARS} characters")
        if image.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported image type: {image.content_type}")

        data = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image too large")
        try:
            pil = Image.open(io.BytesIO(data))
            pil.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Could not read image")

        answer, latency_ms = p.predict(pil, question)
        log.info("predict q=%r answer=%r latency=%.0fms", question, answer, latency_ms)
        return PredictResponse(
            question=question,
            answer=answer,
            latency_ms=round(latency_ms, 1),
            model_version=p.config.model_version,
        )

    return app


app = create_app()
