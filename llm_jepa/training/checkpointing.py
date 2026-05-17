from __future__ import annotations

from pathlib import Path


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None
