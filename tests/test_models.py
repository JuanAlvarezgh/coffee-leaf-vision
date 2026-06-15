import pytest
import torch

ARCHITECTURES = [
    "resnet50",
    "efficientnetv2_s",
    "mobilenetv3_large",
    "vit_small",
]


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_build_model_returns_module_with_correct_output_shape(arch):
    from src.models import build_model

    model = build_model(arch, num_classes=5, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_build_model_unknown_arch_raises():
    from src.models import build_model

    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("nonexistent_arch", num_classes=5, pretrained=False)


def test_get_target_layer_returns_named_module():
    import torch.nn as nn

    from src.models import build_model, get_target_layer

    model = build_model("resnet50", num_classes=5, pretrained=False)
    target = get_target_layer(model, "resnet50")
    assert target is not None
    # Grad-CAM target layers are typically whole blocks (e.g. ResNet Bottleneck),
    # so we verify it's a valid nn.Module with learnable parameters, not a single
    # layer exposing a direct `.weight` attribute.
    assert isinstance(target, nn.Module)
    assert any(p.requires_grad for p in target.parameters())
