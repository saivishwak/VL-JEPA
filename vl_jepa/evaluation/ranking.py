from __future__ import annotations

from pathlib import Path

from vl_jepa.data.schemas import VLJEPASample
from vl_jepa.evaluation.metrics import recall_at_k, topk_accuracy
from vl_jepa.inference.infer import (
    load_vl_jepa_checkpoint,
    predict_vl_jepa_embedding,
    rank_candidates,
)
from vl_jepa.utils.config import load_config
from vl_jepa.utils.io import make_run_dir, read_jsonl, write_json, write_jsonl


def evaluate_ranking_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = make_run_dir(config.get("run_root", "runs"), config.get("name", "vl-jepa-eval"))
    model = load_vl_jepa_checkpoint(config["checkpoint"])
    records = [VLJEPASample.model_validate(record) for record in read_jsonl(config["input_file"])]
    predictions = []
    ranked_labels = []
    targets = []
    for record in records[: config.get("max_examples") or len(records)]:
        embedding = predict_vl_jepa_embedding(
            model,
            visual_path=record.visual_path or "",
            visual_kind=record.visual_kind,
            frame_paths=record.frame_paths,
            query=record.query,
        )
        candidates = [candidate.text for candidate in record.candidates]
        ranked = rank_candidates(model, embedding, candidates)
        labels = [item["text"] for item in ranked]
        target = record.target or next(
            (candidate.text for candidate in record.candidates if candidate.is_correct),
            "",
        )
        ranked_labels.append(labels)
        targets.append(target)
        predictions.append(
            {
                "id": record.id,
                "target": target,
                "ranked": ranked,
                "correct": bool(labels and labels[0] == target),
            }
        )

    task = str(config.get("task", "classification"))
    ks = config.get("recall_k" if task == "retrieval" else "top_k", [1])
    metric_fn = recall_at_k if task == "retrieval" else topk_accuracy
    metrics = {
        "task": task,
        "examples": len(predictions),
        **{
            f"{'recall' if task == 'retrieval' else 'top'}@{k}": metric_fn(
                ranked_labels, targets, int(k)
            )
            for k in ks
        },
    }
    write_jsonl(predictions, run_dir / "predictions.jsonl")
    metrics_path = run_dir / "metrics.json"
    write_json(metrics, metrics_path)
    return metrics_path
