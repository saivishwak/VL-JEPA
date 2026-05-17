from __future__ import annotations

from pathlib import Path

import torch

from vl_jepa.utils.config import load_config
from vl_jepa.utils.io import make_run_dir, read_jsonl, write_json, write_jsonl


def semantic_shift_points(embeddings: torch.Tensor, threshold: float) -> list[int]:
    if embeddings.shape[0] == 0:
        return []
    points = [0]
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    distances = 1.0 - (normalized[1:] * normalized[:-1]).sum(dim=-1)
    for index, distance in enumerate(distances.tolist(), start=1):
        if distance >= threshold:
            points.append(index)
    return points


def uniform_points(num_steps: int, interval: int) -> list[int]:
    if num_steps <= 0:
        return []
    return list(range(0, num_steps, max(1, interval)))


def evaluate_selective_decoding_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = make_run_dir(config.get("run_root", "runs"), config.get("name", "vl-jepa-selective"))
    records = read_jsonl(config["input_file"])
    predictions = []
    for record in records[: config.get("max_examples") or len(records)]:
        embeddings = torch.tensor(record["embeddings"], dtype=torch.float32)
        threshold = float(config.get("shift_threshold", 0.15))
        selected = semantic_shift_points(embeddings, threshold)
        predictions.append(
            {
                "id": record.get("id"),
                "selected_points": selected,
                "decode_count": len(selected),
                "total_points": int(embeddings.shape[0]),
            }
        )
    metrics = {
        "task": "selective_decoding",
        "examples": len(predictions),
        "avg_decode_count": (
            sum(item["decode_count"] for item in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
    }
    write_jsonl(predictions, run_dir / "predictions.jsonl")
    metrics_path = run_dir / "metrics.json"
    write_json(metrics, metrics_path)
    return metrics_path
