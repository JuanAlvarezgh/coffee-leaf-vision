"""Model factory wrapping timm for the 4 architectures compared in this project."""

import timm
import torch.nn as nn

ARCH_TIMM_NAMES = {
    "resnet50": "resnet50",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "vit_small": "vit_small_patch16_224",
}


def build_model(arch: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    if arch not in ARCH_TIMM_NAMES:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(ARCH_TIMM_NAMES)}")
    return timm.create_model(
        ARCH_TIMM_NAMES[arch],
        pretrained=pretrained,
        num_classes=num_classes,
    )


def get_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the layer to attach Grad-CAM to for each architecture."""
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "efficientnetv2_s":
        return model.blocks[-1][-1]
    if arch == "mobilenetv3_large":
        return model.blocks[-1][-1]
    if arch == "vit_small":
        return model.blocks[-1].norm1
    raise ValueError(f"No Grad-CAM target defined for {arch}")
