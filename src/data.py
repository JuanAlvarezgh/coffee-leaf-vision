"""PyTorch Dataset class and Albumentations transforms for coffee leaf images."""

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from config import CLASS_LABELS, IMAGENET_MEAN, IMAGENET_STD

LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}


def build_train_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size * 1.15)),
            A.PadIfNeeded(
                min_height=int(image_size * 1.15), min_width=int(image_size * 1.15), border_mode=0
            ),
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class CoffeeLeafDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: A.Compose | None = None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = np.array(Image.open(row["filepath"]).convert("RGB"))
        label_idx = LABEL_TO_INDEX[row["label"]]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label_idx
