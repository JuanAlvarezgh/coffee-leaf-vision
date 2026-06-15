import io
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


def _tiny_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


MOCK_METRICS = {
    "architecture": "resnet50",
    "model_version": "1",
    "cv_summary": {
        "accuracy": {"mean": 0.91, "std": 0.02},
        "f1_macro": {"mean": 0.90, "std": 0.02},
        "auc_macro_ovr": {"mean": 0.97, "std": 0.01},
    },
    "n_parameters": 25_000_000,
    "size_mb": 96.0,
    "latency_median_ms": 80.0,
    "trained_at": "2026-06-04T03:52:34",
}


@pytest.fixture
def client():
    fake_model = torch.nn.Linear(3 * 224 * 224, 5)
    with patch("api.main.model_loader.load"):
        with (
            patch("api.main.model_loader.model", fake_model),
            patch("api.main.model_loader.architecture", "resnet50"),
            patch("api.main.model_loader.model_version", "1"),
            patch("api.main.model_loader.metrics", MOCK_METRICS),
        ):
            from api.main import app

            with TestClient(app) as c:
                yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["architecture"] == "resnet50"
    assert body["cv_f1_macro_mean"] == 0.90


def test_predict_with_mocked_inference(client):
    fake_result = {
        "predicted_class": "leaf_rust",
        "predicted_class_es": "Roya del cafeto",
        "confidence": 0.87,
        "probabilities": {
            "healthy": 0.05,
            "leaf_rust": 0.87,
            "leaf_miner": 0.03,
            "phoma": 0.03,
            "cercospora": 0.02,
        },
        "gradcam_overlay_base64": "iVBORw0KG",
        "model_version": "1",
        "inference_ms": 150.0,
    }
    with patch("api.main.predict_image", return_value=fake_result):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.jpg", _tiny_image_bytes(), "image/jpeg")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "leaf_rust"
    assert body["confidence"] == pytest.approx(0.87)


def test_predict_rejects_non_image(client):
    r = client.post(
        "/api/v1/predict",
        files={"file": ("text.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "Expected an image" in r.json()["detail"]


def test_predict_rejects_corrupted_image(client):
    r = client.post(
        "/api/v1/predict",
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_rejects_invalid_gradcam_method(client):
    r = client.post(
        "/api/v1/predict",
        params={"gradcam_method": "evil_method"},
        files={"file": ("leaf.jpg", _tiny_image_bytes(), "image/jpeg")},
    )
    assert r.status_code == 422
