from __future__ import annotations

from pathlib import Path

from vl_jepa.evaluation.ranking import evaluate_ranking_from_config
from vl_jepa.evaluation.selective_decoding import evaluate_selective_decoding_from_config
from vl_jepa.evaluation.text_triplet import evaluate_text_triplets_from_config
from vl_jepa.utils.config import load_config


def evaluate_vl_jepa_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    task = str(config.get("task", "classification"))
    if task == "selective_decoding":
        return evaluate_selective_decoding_from_config(config_path)
    if task == "text_triplet":
        return evaluate_text_triplets_from_config(config_path)
    return evaluate_ranking_from_config(config_path)
