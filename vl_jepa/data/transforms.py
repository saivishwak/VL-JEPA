from __future__ import annotations

from pathlib import Path

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: str | Path, image_size: int = 256) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Pillow is required for VL-JEPA image loading") from exc

    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size))
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = data.view(image_size, image_size, 3).permute(2, 0, 1).float().div(255.0)
    return normalize_image(tensor)


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def stack_frames(frames: list[torch.Tensor]) -> torch.Tensor:
    if not frames:
        raise ValueError("at least one frame is required")
    return torch.stack(frames, dim=0)
