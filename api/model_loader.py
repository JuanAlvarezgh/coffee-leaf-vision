"""Loads the production model + metrics from Hugging Face Hub at startup."""

import json

import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from config import settings
from src.models import build_model


class ModelLoader:
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.architecture: str | None = None
        self.metrics: dict = {}
        self.model_version: str = "unknown"

    def load(self) -> None:
        logger.info(
            f"Downloading checkpoint from {settings.hf_model_repo}/{settings.hf_model_filename}"
        )

        ckpt_path = hf_hub_download(
            repo_id=settings.hf_model_repo,
            filename=settings.hf_model_filename,
        )
        metrics_path = hf_hub_download(
            repo_id=settings.hf_model_repo,
            filename=settings.hf_metrics_filename,
        )

        with open(metrics_path) as f:
            self.metrics = json.load(f)

        self.architecture = self.metrics["architecture"]
        self.model_version = self.metrics.get("model_version", "v1")

        logger.info(f"Building {self.architecture} (num_classes=5)")
        self.model = build_model(self.architecture, num_classes=5, pretrained=False)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info(f"Loaded {self.architecture} v{self.model_version}")


model_loader = ModelLoader()
