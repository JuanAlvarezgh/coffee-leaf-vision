"""Per-request inference: forward pass + Grad-CAM overlay."""

import base64
import io
import time

import torch
from PIL import Image

from config import CLASS_LABELS, CLASS_LABELS_ES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import generate_gradcam


def _preprocess(pil_image: Image.Image) -> torch.Tensor:
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = (
        torch.tensor(list(img.getdata()), dtype=torch.float32).view(IMAGE_SIZE, IMAGE_SIZE, 3)
        / 255.0
    )
    mean = torch.tensor(IMAGENET_MEAN)
    std = torch.tensor(IMAGENET_STD)
    arr = (arr - mean) / std
    return arr.permute(2, 0, 1).unsqueeze(0)


def _encode_pil_to_base64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_image(
    pil_image: Image.Image,
    model: torch.nn.Module,
    architecture: str,
    model_version: str,
    gradcam_method: str = "gradcam",
) -> dict:
    """Run forward pass + Grad-CAM, return structured response payload."""
    t0 = time.perf_counter()
    model.eval()
    tensor = _preprocess(pil_image)

    with torch.no_grad():
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1)[0].tolist()

    pred_idx = int(max(range(len(proba)), key=lambda i: proba[i]))
    pred_label = CLASS_LABELS[pred_idx]

    overlay_pil, _ = generate_gradcam(
        model, pil_image, arch=architecture, method=gradcam_method, target_class=pred_idx
    )
    overlay_b64 = _encode_pil_to_base64(overlay_pil)

    inference_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "predicted_class": pred_label,
        "predicted_class_es": CLASS_LABELS_ES[pred_label],
        "confidence": float(proba[pred_idx]),
        "probabilities": {label: float(p) for label, p in zip(CLASS_LABELS, proba, strict=False)},
        "gradcam_overlay_base64": overlay_b64,
        "model_version": model_version,
        "inference_ms": float(inference_ms),
    }
