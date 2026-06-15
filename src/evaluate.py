"""Compute classification metrics for model evaluation."""

import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from config import CLASS_LABELS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Returns a dict with accuracy, F1 macro/weighted, per-class F1, and macro AUC."""
    accuracy = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_LABELS))))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS))))

    try:
        auc_macro = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc_macro = float("nan")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": {
            label: float(s) for label, s in zip(CLASS_LABELS, f1_per_class, strict=False)
        },
        "confusion_matrix": cm.tolist(),
        "auc_macro_ovr": auc_macro,
    }


def measure_inference_latency(
    model: torch.nn.Module,
    image_size: int = 224,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """Measure forward-pass latency in milliseconds (CPU by default)."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        timings = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)

    timings_arr = np.array(timings)
    return {
        "mean_ms": float(timings_arr.mean()),
        "median_ms": float(np.median(timings_arr)),
        "p95_ms": float(np.percentile(timings_arr, 95)),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in megabytes."""
    n_params = count_parameters(model)
    return (n_params * 4) / (1024 * 1024)


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std for each numeric metric across folds."""
    keys = ["accuracy", "f1_macro", "f1_weighted", "auc_macro_ovr"]
    summary = {}
    for k in keys:
        values = np.array([m[k] for m in fold_metrics if not np.isnan(m[k])])
        summary[k] = {
            "mean": float(values.mean()) if len(values) > 0 else float("nan"),
            "std": float(values.std()) if len(values) > 0 else float("nan"),
        }
    return summary


def pretty_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_LABELS))),
        target_names=CLASS_LABELS,
        zero_division=0,
    )
