"""Pydantic schemas for the API."""

from pydantic import BaseModel, Field

from config import CLASS_LABELS


class ProbabilityMap(BaseModel):
    healthy: float = Field(..., ge=0.0, le=1.0)
    leaf_rust: float = Field(..., ge=0.0, le=1.0)
    leaf_miner: float = Field(..., ge=0.0, le=1.0)
    phoma: float = Field(..., ge=0.0, le=1.0)
    cercospora: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_class: str = Field(..., description="One of: " + ", ".join(CLASS_LABELS))
    predicted_class_es: str = Field(..., description="Spanish label for display")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: ProbabilityMap
    gradcam_overlay_base64: str = Field(..., description="PNG bytes encoded as base64")
    model_version: str
    inference_ms: float


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    architecture: str
    model_version: str
    cv_accuracy_mean: float | None = None
    cv_accuracy_std: float | None = None
    cv_f1_macro_mean: float | None = None
    cv_f1_macro_std: float | None = None
    cv_auc_macro_mean: float | None = None
    cv_auc_macro_std: float | None = None
    n_parameters: int | None = None
    size_mb: float | None = None
    latency_median_ms: float | None = None
    trained_at: str | None = None
