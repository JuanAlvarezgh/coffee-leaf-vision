"""One-off script to upload the winning checkpoint + metrics.json to HF Hub.

Usage:
    huggingface-cli login
    python scripts/upload_to_hf.py path/to/best_model.pt path/to/results.json
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from loguru import logger

from config import settings


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/upload_to_hf.py <checkpoint.pt> <results.json>")
        return 1

    ckpt = Path(sys.argv[1])
    results_path = Path(sys.argv[2])

    if not ckpt.exists() or not results_path.exists():
        logger.error("Both files must exist.")
        return 1

    with open(results_path) as f:
        results = json.load(f)

    best_arch = max(results.keys(), key=lambda a: results[a]["cv_summary"]["f1_macro"]["mean"])
    best_data = results[best_arch]

    metrics_payload = {
        "architecture": best_arch,
        "model_version": "v1",
        "cv_summary": best_data["cv_summary"],
        "n_parameters": best_data["n_params"],
        "size_mb": best_data["size_mb"],
        "latency_median_ms": best_data["latency_cpu"]["median_ms"],
        "trained_at": "2026-06-04T00:00:00",
    }
    metrics_path = Path("metrics.json")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    repo_id = settings.hf_model_repo
    api = HfApi()

    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        logger.info(f"Repo {repo_id} ready.")
    except Exception as e:
        logger.warning(f"create_repo: {e}")

    logger.info(f"Uploading {ckpt} as model.pt")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=settings.hf_model_filename,
        repo_id=repo_id,
        repo_type="model",
    )

    logger.info("Uploading metrics.json")
    api.upload_file(
        path_or_fileobj=str(metrics_path),
        path_in_repo=settings.hf_metrics_filename,
        repo_id=repo_id,
        repo_type="model",
    )

    logger.info(f"Done. Repo URL: https://huggingface.co/{repo_id}")
    metrics_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
