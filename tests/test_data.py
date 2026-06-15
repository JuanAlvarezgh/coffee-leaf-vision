import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_image_path(tmp_path):
    img = Image.new("RGB", (256, 256), color=(120, 80, 40))
    p = tmp_path / "tiny.jpg"
    img.save(p)
    return p


def test_train_transform_returns_tensor_with_correct_shape(tiny_image_path):
    from src.data import build_train_transform

    transform = build_train_transform(image_size=224)
    img = np.array(Image.open(tiny_image_path))
    result = transform(image=img)["image"]
    assert result.shape == (3, 224, 224)
    assert result.is_floating_point()


def test_val_transform_is_deterministic(tiny_image_path):
    from src.data import build_val_transform

    transform = build_val_transform(image_size=224)
    img = np.array(Image.open(tiny_image_path))
    a = transform(image=img)["image"]
    b = transform(image=img)["image"]
    assert (a == b).all()


def test_dataset_returns_image_and_int_label(tmp_path, tiny_image_path):
    import pandas as pd

    from src.data import CoffeeLeafDataset, build_val_transform

    manifest = pd.DataFrame(
        {
            "filepath": [str(tiny_image_path)],
            "label": ["leaf_rust"],
            "variety": ["arabica"],
            "source": ["BRACOL"],
        }
    )
    transform = build_val_transform(image_size=224)
    ds = CoffeeLeafDataset(manifest, transform=transform)

    image, label = ds[0]
    assert image.shape == (3, 224, 224)
    assert label == 1  # leaf_rust index


def test_dataset_class_indices_match_config():
    from config import CLASS_LABELS
    from src.data import LABEL_TO_INDEX

    for i, label in enumerate(CLASS_LABELS):
        assert LABEL_TO_INDEX[label] == i
