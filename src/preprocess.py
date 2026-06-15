"""Harmonize BRACOL and RoCoLe label schemes into a unified 5-class manifest.

BRACOL (arabica) provides all 5 classes via a folder-per-class layout.
RoCoLe (robusta) provides only `healthy` and `leaf_rust` via a CSV annotation
file; its `red_spider_mite` class has no equivalent in the 5-class scheme and is
intentionally dropped (it is a different pest from the leaf miner).
"""

import csv
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split

# BRACOL "symptom" dataset folder names (e.g. 1_health, 2_miner) plus a few
# aliases for robustness across BRACOL distributions.
BRACOL_LABEL_MAP = {
    "1_health": "healthy",
    "2_miner": "leaf_miner",
    "3_rust": "leaf_rust",
    "4_phoma": "phoma",
    "5_cercospora": "cercospora",
    "healthy": "healthy",
    "health": "healthy",
    "miner": "leaf_miner",
    "rust": "leaf_rust",
    "phoma": "phoma",
    "cercospora": "cercospora",
    "cerc": "cercospora",
}

# RoCoLe only contributes healthy + leaf_rust. `red_spider_mite` is a distinct
# pest with no counterpart in BRACOL, so it is mapped to None and skipped.
ROCOLE_LABEL_MAP = {
    "healthy": "healthy",
    "rust_level_1": "leaf_rust",
    "rust_level_2": "leaf_rust",
    "rust_level_3": "leaf_rust",
    "rust_level_4": "leaf_rust",
    "red_spider_mite": None,
}


def harmonize_bracol_label(raw_label: str) -> str:
    key = raw_label.lower().strip()
    if key not in BRACOL_LABEL_MAP:
        raise ValueError(f"Unknown BRACOL label: {raw_label}")
    return BRACOL_LABEL_MAP[key]


def harmonize_rocole_label(raw_label: str) -> str | None:
    """Map a RoCoLe class to the unified scheme, or None if it should be skipped."""
    key = raw_label.lower().strip()
    if key not in ROCOLE_LABEL_MAP:
        raise ValueError(f"Unknown RoCoLe label: {raw_label}")
    return ROCOLE_LABEL_MAP[key]


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _collect_bracol_class_dirs(bracol_root: Path) -> list[Path]:
    """Return the class folders.

    The BRACOL 'symptom' dataset is laid out as {train,val,test}/{class}/. We
    merge the predefined splits (we run our own k-fold CV) and also support a
    flat {class}/ layout for robustness.
    """
    split_names = {"train", "val", "test"}
    children = [c for c in bracol_root.iterdir() if c.is_dir()]
    if children and all(c.name.lower() in split_names for c in children):
        class_dirs = []
        for split_dir in children:
            class_dirs.extend(c for c in split_dir.iterdir() if c.is_dir())
        return class_dirs
    return children


BRACOL_LEAF_STRESS_MAP = {
    0: "healthy",
    1: "leaf_miner",
    2: "leaf_rust",
    3: "phoma",
    4: "cercospora",
    # 5 = "unknown / not labelled" — drop
}


def _build_bracol_leaf_manifest(leaf_root: Path) -> list[dict]:
    """Read BRACOL 'leaf' dataset: full-leaf images + dataset.csv.

    Layout:
        leaf/dataset.csv         (id, predominant_stress, miner, rust, phoma, cercospora, severity)
        leaf/images/<id>.jpg     (full-leaf photos)
    """
    rows = []
    skipped = 0
    images_dir = leaf_root / "images"
    csv_path = leaf_root / "dataset.csv"
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            stress = int(record["predominant_stress"])
            if stress not in BRACOL_LEAF_STRESS_MAP:
                skipped += 1
                continue
            img_path = images_dir / f"{record['id']}.jpg"
            if not img_path.exists():
                continue
            rows.append(
                {
                    "filepath": str(img_path),
                    "label": BRACOL_LEAF_STRESS_MAP[stress],
                    "variety": "arabica",
                    "source": "BRACOL",
                }
            )
    logger.info(f"BRACOL leaf: {len(rows)} images kept, {skipped} unknown stress skipped")
    return rows


def build_bracol_manifest(bracol_root: Path) -> list[dict]:
    """Detect BRACOL layout and return its manifest.

    - 'leaf' layout (full-leaf images + dataset.csv) → richer real-world photos.
    - 'symptom' layout (folders per class under train/val/test) → cropped symptoms.
    """
    if (bracol_root / "dataset.csv").exists() and (bracol_root / "images").is_dir():
        return _build_bracol_leaf_manifest(bracol_root)

    rows = []
    for class_dir in _collect_bracol_class_dirs(bracol_root):
        try:
            label = harmonize_bracol_label(class_dir.name)
        except ValueError:
            logger.warning(f"Skipping unknown BRACOL class folder: {class_dir.name}")
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "filepath": str(img_path),
                        "label": label,
                        "variety": "arabica",
                        "source": "BRACOL",
                    }
                )
    return rows


def build_rocole_manifest(rocole_root: Path) -> list[dict]:
    """Read the RoCoLe CSV annotations and map to the unified scheme.

    RoCoLe stores all images flat in a Photos/ folder; labels live in
    Annotations/RoCoLE-csv.csv. Each row's `Label` column is a JSON blob whose
    `classification` field holds the class. `red_spider_mite` rows are skipped.
    """
    csv_path = rocole_root / "Annotations" / "RoCoLE-csv.csv"
    photos_dir = rocole_root / "Photos"
    if not csv_path.exists():
        logger.warning(f"RoCoLe CSV not found at {csv_path} — skipping RoCoLe")
        return []

    rows = []
    skipped = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            filename = record.get("External ID", "").strip()
            try:
                raw_class = json.loads(record["Label"]).get("classification", "")
            except (json.JSONDecodeError, KeyError):
                continue
            try:
                label = harmonize_rocole_label(raw_class)
            except ValueError:
                logger.warning(f"Skipping unknown RoCoLe class: {raw_class}")
                continue
            if label is None:  # red_spider_mite — not in the 5-class scheme
                skipped += 1
                continue
            img_path = photos_dir / filename
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "filepath": str(img_path),
                        "label": label,
                        "variety": "robusta",
                        "source": "RoCoLe",
                    }
                )
    logger.info(f"RoCoLe: {len(rows)} images kept, {skipped} red_spider_mite skipped")
    return rows


def build_manifest(bracol_root: Path, rocole_root: Path) -> pd.DataFrame:
    """Combine BRACOL (folders) + RoCoLe (CSV) into one unified manifest."""
    rows = build_bracol_manifest(bracol_root) + build_rocole_manifest(rocole_root)

    df = pd.DataFrame(rows)
    logger.info(
        f"Built manifest: {len(df)} images, "
        f"classes={df['label'].value_counts().to_dict()}, "
        f"varieties={df['variety'].value_counts().to_dict()}"
    )
    return df


def save_manifest(manifest: pd.DataFrame, out_path: Path) -> None:
    manifest.to_csv(out_path, index=False)
    logger.info(f"Saved manifest to {out_path}")


def stratified_split(
    manifest: pd.DataFrame,
    test_size: float = 0.15,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Stratified train/test split with k-fold CV on train.

    Stratification key = label + variety to keep both class and variety balance.

    Returns dict with:
        test_idx: list of indices held out for final test
        folds: list of (train_idx, val_idx) for k-fold CV on the remaining data
    """
    strat_key = manifest["label"].astype(str) + "_" + manifest["variety"].astype(str)
    indices = manifest.index.to_numpy()

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )

    train_val_strat = strat_key.loc[train_val_idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_train_positions, fold_val_positions in skf.split(train_val_idx, train_val_strat):
        folds.append(
            (
                train_val_idx[fold_train_positions],
                train_val_idx[fold_val_positions],
            )
        )

    logger.info(
        f"Stratified split: test={len(test_idx)}, "
        f"train_val={len(train_val_idx)}, {n_folds} folds"
    )

    return {"test_idx": test_idx, "folds": folds}


if __name__ == "__main__":
    bracol = Path("data") / "BRACOL"
    rocole = Path("data") / "RoCoLe"
    manifest = build_manifest(bracol, rocole)
    save_manifest(manifest, Path("data") / "manifest.csv")
