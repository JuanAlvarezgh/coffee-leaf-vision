"""Two-phase training (warm-up then fine-tune) with early stopping for one fold."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.evaluate import compute_metrics


def _freeze_backbone(model: nn.Module, freeze: bool) -> None:
    """Freeze or unfreeze all params except the classification head.

    timm models expose the classifier via `get_classifier()`.
    """
    classifier = model.get_classifier()
    classifier_params = set(id(p) for p in classifier.parameters())
    for p in model.parameters():
        if id(p) in classifier_params:
            p.requires_grad = True
        else:
            p.requires_grad = not freeze


def _one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.train() if train else model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_proba = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = proba.argmax(axis=1)
        all_true.append(labels.cpu().numpy())
        all_pred.append(pred)
        all_proba.append(proba)

    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba, axis=0),
    )


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    warmup_epochs: int = 10,
    finetune_epochs: int = 20,
    warmup_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    patience: int = 5,
    checkpoint_path: Path | None = None,
) -> dict:
    """Train one fold and return its best validation metrics + per-epoch history."""
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_metrics = None
    patience_left = patience

    # Phase 1: warm-up (frozen backbone)
    _freeze_backbone(model, freeze=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=warmup_lr)
    for epoch in range(warmup_epochs):
        train_loss, *_ = _one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, y_true, y_pred, y_proba = _one_epoch(
            model, val_loader, None, criterion, device, train=False
        )
        metrics = compute_metrics(y_true, y_pred, y_proba)
        history.append(
            {
                "phase": "warmup",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
        )
        logger.info(
            f"[warmup {epoch+1}/{warmup_epochs}] train={train_loss:.4f} val={val_loss:.4f} f1={metrics['f1_macro']:.4f}"
        )
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_metrics = metrics
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("Early stop during warm-up")
                break

    # Phase 2: fine-tune (full unfreeze)
    _freeze_backbone(model, freeze=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    patience_left = patience
    for epoch in range(finetune_epochs):
        train_loss, *_ = _one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, y_true, y_pred, y_proba = _one_epoch(
            model, val_loader, None, criterion, device, train=False
        )
        metrics = compute_metrics(y_true, y_pred, y_proba)
        history.append(
            {
                "phase": "finetune",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
        )
        logger.info(
            f"[finetune {epoch+1}/{finetune_epochs}] train={train_loss:.4f} val={val_loss:.4f} f1={metrics['f1_macro']:.4f}"
        )
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_metrics = metrics
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("Early stop during fine-tune")
                break

    return {"best_metrics": best_metrics, "history": history}
