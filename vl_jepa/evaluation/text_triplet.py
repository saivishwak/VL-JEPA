from __future__ import annotations

from pathlib import Path

import torch

from vl_jepa.inference.infer import load_vl_jepa_checkpoint
from vl_jepa.utils.config import load_config
from vl_jepa.utils.io import make_run_dir, read_jsonl, write_json


def evaluate_text_triplets_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = make_run_dir(
        config.get("run_root", "runs"), config.get("name", "vl-jepa-text-triplet")
    )
    model = load_vl_jepa_checkpoint(config["checkpoint"])
    records = read_jsonl(config["input_file"])
    device = next(model.parameters()).device
    correct = []
    for record in records[: config.get("max_examples") or len(records)]:
        texts = [record["positive_1"], record["positive_2"], record["negative"]]
        encoded = model.target_tokenizer(
            texts,
            truncation=True,
            max_length=model.config.max_target_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = model.encode_target(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
        sims = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:], dim=-1)
        correct.append(bool(sims[0] > sims[1]))
    metrics = {
        "task": "text_triplet",
        "examples": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics, metrics_path)
    return metrics_path
