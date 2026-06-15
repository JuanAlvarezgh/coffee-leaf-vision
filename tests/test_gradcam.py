import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_image():
    return Image.new("RGB", (256, 256), color=(140, 100, 60))


def test_generate_gradcam_returns_pil_image(tiny_image):
    from src.gradcam import generate_gradcam
    from src.models import build_model

    model = build_model("resnet50", num_classes=5, pretrained=False)
    overlay, raw = generate_gradcam(model, tiny_image, arch="resnet50", method="gradcam")
    assert isinstance(overlay, Image.Image)
    assert isinstance(raw, np.ndarray)
    assert overlay.size == tiny_image.size


def test_generate_gradcam_supported_methods(tiny_image):
    from src.gradcam import SUPPORTED_METHODS, generate_gradcam
    from src.models import build_model

    model = build_model("resnet50", num_classes=5, pretrained=False)
    for method in SUPPORTED_METHODS:
        overlay, _ = generate_gradcam(model, tiny_image, arch="resnet50", method=method)
        assert isinstance(overlay, Image.Image)


def test_generate_gradcam_unknown_method_raises(tiny_image):
    from src.gradcam import generate_gradcam
    from src.models import build_model

    model = build_model("resnet50", num_classes=5, pretrained=False)
    with pytest.raises(ValueError, match="Unsupported method"):
        generate_gradcam(model, tiny_image, arch="resnet50", method="unknown")
