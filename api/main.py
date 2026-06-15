"""FastAPI application."""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from loguru import logger
from PIL import Image, UnidentifiedImageError

from api.exceptions import (
    ImageValidationError,
    generic_exception_handler,
    image_validation_handler,
    validation_exception_handler,
)
from api.inference import predict_image
from api.model_loader import model_loader
from api.schemas import ModelInfo, PredictResponse, ProbabilityMap
from config import settings

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", serialize=True, level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from HF Hub...")
    model_loader.load()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="CoffeeLeafVision Scoring API",
    description="Coffee leaf disease classification with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_exception_handler(ImageValidationError, image_validation_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/api/v1/model/info", response_model=ModelInfo, tags=["model"])
def get_model_info():
    metrics = model_loader.metrics
    cv = metrics.get("cv_summary", {})
    return ModelInfo(
        architecture=model_loader.architecture or "unknown",
        model_version=model_loader.model_version,
        cv_accuracy_mean=cv.get("accuracy", {}).get("mean"),
        cv_accuracy_std=cv.get("accuracy", {}).get("std"),
        cv_f1_macro_mean=cv.get("f1_macro", {}).get("mean"),
        cv_f1_macro_std=cv.get("f1_macro", {}).get("std"),
        cv_auc_macro_mean=cv.get("auc_macro_ovr", {}).get("mean"),
        cv_auc_macro_std=cv.get("auc_macro_ovr", {}).get("std"),
        n_parameters=metrics.get("n_parameters"),
        size_mb=metrics.get("size_mb"),
        latency_median_ms=metrics.get("latency_median_ms"),
        trained_at=metrics.get("trained_at"),
    )


def _validate_upload(file: UploadFile, content: bytes) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ImageValidationError(
            f"Expected an image upload, got content_type={file.content_type}"
        )
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ImageValidationError(
            f"Image too large: {len(content) / 1024 / 1024:.1f} MB > {settings.max_upload_mb} MB"
        )
    try:
        import io

        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
    except UnidentifiedImageError as e:
        raise ImageValidationError(f"Could not parse image: {e}") from e
    if img.width < 32 or img.height < 32:
        raise ImageValidationError(
            f"Image too small: {img.width}x{img.height}, need at least 32x32"
        )
    return img


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["scoring"])
async def predict(
    file: UploadFile = File(..., description="Image of a coffee leaf (JPEG/PNG, <= 10 MB)"),
    gradcam_method: str = Query("gradcam", pattern="^(gradcam|gradcam\\+\\+|eigencam)$"),
):
    content = await file.read()
    pil_image = _validate_upload(file, content)
    logger.info(f"Predicting image size={pil_image.size}, method={gradcam_method}")

    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = predict_image(
        pil_image,
        model=model_loader.model,
        architecture=model_loader.architecture,
        model_version=model_loader.model_version,
        gradcam_method=gradcam_method,
    )
    result["probabilities"] = ProbabilityMap(**result["probabilities"])
    return PredictResponse(**result)
