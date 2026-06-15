import pandas as pd
import pytest


def test_harmonize_bracol_label_known_classes():
    from src.preprocess import harmonize_bracol_label

    # BRACOL symptom folder names
    assert harmonize_bracol_label("1_health") == "healthy"
    assert harmonize_bracol_label("2_miner") == "leaf_miner"
    assert harmonize_bracol_label("3_rust") == "leaf_rust"
    assert harmonize_bracol_label("4_phoma") == "phoma"
    assert harmonize_bracol_label("5_cercospora") == "cercospora"
    # aliases
    assert harmonize_bracol_label("healthy") == "healthy"
    assert harmonize_bracol_label("rust") == "leaf_rust"


def test_harmonize_bracol_label_unknown_raises():
    from src.preprocess import harmonize_bracol_label

    with pytest.raises(ValueError, match="Unknown BRACOL label"):
        harmonize_bracol_label("totally_unknown")


def test_harmonize_rocole_label_known_classes():
    from src.preprocess import harmonize_rocole_label

    assert harmonize_rocole_label("healthy") == "healthy"
    assert harmonize_rocole_label("rust_level_1") == "leaf_rust"
    assert harmonize_rocole_label("rust_level_4") == "leaf_rust"
    # red_spider_mite has no equivalent in the 5-class scheme -> None (skipped).
    assert harmonize_rocole_label("red_spider_mite") is None


def test_harmonize_rocole_label_unknown_raises():
    from src.preprocess import harmonize_rocole_label

    with pytest.raises(ValueError, match="Unknown RoCoLe label"):
        harmonize_rocole_label("ufo")


def _write_rocole_fixture(rocole_root, rows):
    """Create a minimal RoCoLe dataset: Annotations/RoCoLE-csv.csv + Photos/."""
    import csv
    import json

    ann_dir = rocole_root / "Annotations"
    photos_dir = rocole_root / "Photos"
    ann_dir.mkdir(parents=True)
    photos_dir.mkdir(parents=True)

    csv_path = ann_dir / "RoCoLE-csv.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["External ID", "Label"])
        for filename, classification in rows:
            (photos_dir / filename).touch()
            label_json = json.dumps({"classification": classification})
            writer.writerow([filename, label_json])


def test_build_manifest_combines_both_datasets(tmp_path):
    from src.preprocess import build_manifest

    # BRACOL symptom: {split}/{class}/ layout
    bracol_root = tmp_path / "BRACOL"
    (bracol_root / "train" / "1_health").mkdir(parents=True)
    (bracol_root / "train" / "3_rust").mkdir(parents=True)
    (bracol_root / "val" / "2_miner").mkdir(parents=True)
    (bracol_root / "train" / "1_health" / "img1.jpg").touch()
    (bracol_root / "train" / "3_rust" / "img2.jpg").touch()
    (bracol_root / "val" / "2_miner" / "img3.jpg").touch()

    # RoCoLe: CSV annotations + flat Photos. red_spider_mite must be dropped.
    rocole_root = tmp_path / "RoCoLe"
    _write_rocole_fixture(
        rocole_root,
        [
            ("C1P1H1.jpg", "healthy"),
            ("C1P2E2.jpg", "rust_level_2"),
            ("C1P3M1.jpg", "red_spider_mite"),  # should be skipped
        ],
    )

    manifest = build_manifest(bracol_root, rocole_root)

    # 3 from BRACOL + 2 from RoCoLe (red_spider_mite dropped) = 5
    assert len(manifest) == 5
    assert set(manifest["label"].unique()) == {"healthy", "leaf_rust", "leaf_miner"}
    assert set(manifest["variety"].unique()) == {"arabica", "robusta"}
    # red_spider_mite never made it in
    rocole_rows = manifest[manifest["source"] == "RoCoLe"]
    assert len(rocole_rows) == 2
    assert set(rocole_rows["label"].unique()) == {"healthy", "leaf_rust"}


def _write_bracol_leaf_fixture(bracol_root, rows):
    """Create a minimal BRACOL 'leaf' dataset: dataset.csv + images/."""
    import csv

    images_dir = bracol_root / "images"
    images_dir.mkdir(parents=True)
    csv_path = bracol_root / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "predominant_stress", "miner", "rust", "phoma", "cercospora", "severity"]
        )
        for img_id, stress in rows:
            (images_dir / f"{img_id}.jpg").touch()
            writer.writerow([img_id, stress, 0, 0, 0, 0, 0])


def test_build_bracol_leaf_manifest_reads_csv_and_skips_unknown(tmp_path):
    from src.preprocess import build_bracol_manifest

    bracol_root = tmp_path / "BRACOL_leaf"
    _write_bracol_leaf_fixture(
        bracol_root,
        [
            (1, 0),  # healthy
            (2, 1),  # miner
            (3, 2),  # rust
            (4, 5),  # unknown -> skipped
            (5, 4),  # cercospora
        ],
    )

    rows = build_bracol_manifest(bracol_root)

    assert len(rows) == 4  # the "5" row is dropped
    assert all(r["variety"] == "arabica" for r in rows)
    assert {r["label"] for r in rows} == {"healthy", "leaf_miner", "leaf_rust", "cercospora"}


def test_build_rocole_manifest_skips_red_spider_mite(tmp_path):
    from src.preprocess import build_rocole_manifest

    rocole_root = tmp_path / "RoCoLe"
    _write_rocole_fixture(
        rocole_root,
        [
            ("a.jpg", "healthy"),
            ("b.jpg", "rust_level_1"),
            ("c.jpg", "rust_level_3"),
            ("d.jpg", "red_spider_mite"),
            ("e.jpg", "red_spider_mite"),
        ],
    )

    rows = build_rocole_manifest(rocole_root)

    assert len(rows) == 3  # 2 mites dropped
    assert all(r["variety"] == "robusta" for r in rows)
    assert all(r["label"] in ("healthy", "leaf_rust") for r in rows)


def _fake_manifest(n_per_class=100):
    rows = []
    for label in ["healthy", "leaf_rust", "leaf_miner", "phoma", "cercospora"]:
        for variety in ["arabica", "robusta"]:
            for i in range(n_per_class // 2):
                rows.append(
                    {
                        "filepath": f"/fake/{label}_{variety}_{i}.jpg",
                        "label": label,
                        "variety": variety,
                        "source": "BRACOL" if variety == "arabica" else "RoCoLe",
                    }
                )
    return pd.DataFrame(rows)


def test_stratified_split_returns_train_test_and_5_folds():
    from src.preprocess import stratified_split

    manifest = _fake_manifest(n_per_class=200)
    result = stratified_split(manifest, test_size=0.15, n_folds=5, random_state=42)

    assert "test_idx" in result
    assert "folds" in result
    assert len(result["folds"]) == 5


def test_stratified_split_class_proportions_preserved():
    from src.preprocess import stratified_split

    manifest = _fake_manifest(n_per_class=200)
    result = stratified_split(manifest, test_size=0.15, n_folds=5, random_state=42)

    train_val = manifest.drop(index=result["test_idx"])
    test = manifest.loc[result["test_idx"]]

    for label in ["healthy", "leaf_rust", "leaf_miner", "phoma", "cercospora"]:
        train_pct = (train_val["label"] == label).mean()
        test_pct = (test["label"] == label).mean()
        assert abs(train_pct - test_pct) < 0.05, f"Class {label} not stratified"


def test_stratified_split_folds_dont_overlap():
    from src.preprocess import stratified_split

    manifest = _fake_manifest(n_per_class=200)
    result = stratified_split(manifest, test_size=0.15, n_folds=5, random_state=42)

    for train_idx, val_idx in result["folds"]:
        assert len(set(train_idx) & set(val_idx)) == 0


def test_stratified_split_test_set_size():
    from src.preprocess import stratified_split

    manifest = _fake_manifest(n_per_class=200)
    result = stratified_split(manifest, test_size=0.15, n_folds=5, random_state=42)

    expected_test_size = int(len(manifest) * 0.15)
    assert abs(len(result["test_idx"]) - expected_test_size) < 5
