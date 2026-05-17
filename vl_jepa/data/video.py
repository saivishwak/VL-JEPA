from __future__ import annotations

from pathlib import Path

import torch

from vl_jepa.data.transforms import load_image, stack_frames


def uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return [total_frames // 2]
    return torch.linspace(0, total_frames - 1, steps=num_frames).round().long().tolist()


def duplicate_image_as_video(
    path: str | Path,
    num_frames: int,
    image_size: int = 256,
) -> torch.Tensor:
    frame = load_image(path, image_size=image_size)
    return frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def load_frame_paths(paths: list[str], num_frames: int, image_size: int = 256) -> torch.Tensor:
    indices = uniform_indices(len(paths), num_frames)
    return stack_frames([load_image(paths[index], image_size=image_size) for index in indices])


def load_video(path: str | Path, num_frames: int, image_size: int = 256) -> torch.Tensor:
    video_path = Path(path)
    if video_path.is_dir():
        frames = sorted(
            str(item)
            for item in video_path.iterdir()
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        return load_frame_paths(frames, num_frames=num_frames, image_size=image_size)

    try:
        from torchvision.io import read_video
    except ImportError as exc:  # pragma: no cover - optional video dependency
        raise ImportError(
            "torchvision is required to decode video files; use frame_paths or a frame directory"
        ) from exc

    frames, _, _ = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"no frames decoded from {video_path}")
    indices = uniform_indices(frames.shape[0], num_frames)
    selected = []
    for index in indices:
        frame = frames[index].permute(2, 0, 1).float().div(255.0)
        frame = torch.nn.functional.interpolate(
            frame.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        from vl_jepa.data.transforms import normalize_image

        selected.append(normalize_image(frame))
    return stack_frames(selected)


def load_visual_tensor(
    *,
    visual_kind: str,
    visual_path: str | None,
    frame_paths: list[str] | None,
    num_frames: int,
    image_size: int = 256,
) -> torch.Tensor:
    if visual_kind == "image":
        if visual_path is None:
            raise ValueError("image sample requires visual_path")
        return duplicate_image_as_video(visual_path, num_frames=num_frames, image_size=image_size)
    if visual_kind == "frames":
        if not frame_paths:
            raise ValueError("frames sample requires frame_paths")
        return load_frame_paths(frame_paths, num_frames=num_frames, image_size=image_size)
    if visual_path is None:
        raise ValueError("video sample requires visual_path")
    return load_video(visual_path, num_frames=num_frames, image_size=image_size)
