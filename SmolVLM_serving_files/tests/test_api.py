"""API tests with a fake predictor: no GPU or model download needed.

Run: pytest -q
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import InferenceConfig, VQAPredictor
from app.main import create_app


class FakePredictor(VQAPredictor):
    def load(self):
        self.model = object()

    def predict(self, image, question):
        return "polyp", 12.3


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    fake = FakePredictor(config=InferenceConfig(model_version="test"))
    fake.load()
    with TestClient(create_app(predictor=fake, load_model=False)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_predict_ok(client):
    r = client.post(
        "/predict",
        files={"image": ("x.png", _png_bytes(), "image/png")},
        data={"question": "Is there a polyp?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "polyp"
    assert body["model_version"] == "test"
    assert "Not for clinical diagnosis" in body["disclaimer"]


def test_rejects_non_image(client):
    r = client.post(
        "/predict",
        files={"image": ("x.txt", b"hello", "text/plain")},
        data={"question": "Is there a polyp?"},
    )
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post(
        "/predict",
        files={"image": ("x.png", b"not really a png", "image/png")},
        data={"question": "Is there a polyp?"},
    )
    assert r.status_code == 400


def test_rejects_empty_question(client):
    r = client.post(
        "/predict",
        files={"image": ("x.png", _png_bytes(), "image/png")},
        data={"question": "   "},
    )
    assert r.status_code == 422


def test_health_503_when_not_loaded():
    with TestClient(create_app(predictor=FakePredictor(), load_model=False)) as c:
        assert c.get("/health").status_code == 503
