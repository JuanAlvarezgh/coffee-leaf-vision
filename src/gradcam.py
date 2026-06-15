"""Generate Grad-CAM overlays for predictions."""

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.models import get_target_layer

SUPPORTED_METHODS = ("gradcam", "gradcam++", "eigencam")


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ViT token output to a spatial feature map for Grad-CAM.

    ViT outputs (batch, num_tokens, features) where num_tokens = 1 CLS + H*W
    patch tokens. For ViT-Small/16 at 224x224: 1 + 14*14 = 197 tokens.
    Grad-CAM needs a 4D tensor (batch, features, H, W).
    """
    # Drop CLS token, reshape to (B, H, W, C), permute to (B, C, H, W)
    tokens = tensor[:, 1:, :]  # (B, 196, C)
    side = int(tokens.shape[1] ** 0.5)
    return tokens.reshape(tensor.size(0), side, side, tensor.size(-1)).permute(0, 3, 1, 2)


def _reshape_for(arch: str):
    """Return the reshape_transform function for arch, or None for CNNs."""
    if arch == "vit_small":
        return _vit_reshape_transform
    return None


def _build_cam(model: torch.nn.Module, arch: str, method: str):
    target_layer = get_target_layer(model, arch)
    reshape = _reshape_for(arch)
    kwargs = {"model": model, "target_layers": [target_layer]}
    if reshape is not None:
        kwargs["reshape_transform"] = reshape
    if method == "gradcam":
        return GradCAM(**kwargs)
    if method == "gradcam++":
        return GradCAMPlusPlus(**kwargs)
    if method == "eigencam":
        return EigenCAM(**kwargs)
    raise ValueError(f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}.")


def _preprocess_for_cam(pil_image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Resize to model input, normalize for tensor; also return un-normalized float image."""
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_uint8 = np.array(img)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (rgb_float - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, rgb_float


def generate_gradcam(
    model: torch.nn.Module,
    pil_image: Image.Image,
    arch: str,
    method: str = "gradcam",
    target_class: int | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Generate a Grad-CAM overlay PIL image plus the raw heatmap array."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}. Choose from {SUPPORTED_METHODS}.")

    model.eval()
    tensor, rgb_float = _preprocess_for_cam(pil_image)
    cam = _build_cam(model, arch, method)

    targets = None
    if method != "eigencam" and target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = [ClassifierOutputTarget(target_class)]

    heatmap = cam(input_tensor=tensor, targets=targets)[0]

    overlay_np = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    overlay_pil = Image.fromarray(overlay_np).resize(pil_image.size)
    return overlay_pil, heatmap
