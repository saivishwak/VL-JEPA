from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vl_jepa.data.schemas import VLJEPASample
from vl_jepa.data.video import load_visual_tensor
from vl_jepa.modeling import VLJEPA, VLJEPAConfig
from vl_jepa.utils.config import load_config
from vl_jepa.utils.io import read_jsonl, write_jsonl


def load_vl_jepa_checkpoint(
    checkpoint: str | Path, device: str | torch.device | None = None
) -> VLJEPA:
    checkpoint_path = Path(checkpoint)
    model_config = load_config(checkpoint_path / "model_config.yaml")
    model = VLJEPA(VLJEPAConfig(**model_config))
    state = torch.load(checkpoint_path / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(target_device).eval()


def predict_vl_jepa_embedding(
    model: VLJEPA,
    *,
    visual_path: str,
    query: str,
    visual_kind: str = "image",
    frame_paths: list[str] | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    pixel_values = load_visual_tensor(
        visual_kind=visual_kind,
        visual_path=visual_path,
        frame_paths=frame_paths,
        num_frames=model.config.num_frames,
        image_size=model.config.image_size,
    ).unsqueeze(0)
    query_tokens = model.query_tokenizer(
        query,
        truncation=True,
        max_length=model.config.max_query_length,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return model.predict_embedding(
            pixel_values.to(device),
            query_tokens["input_ids"].to(device),
            query_tokens["attention_mask"].to(device),
        )


def rank_candidates(
    model: VLJEPA,
    embedding: torch.Tensor,
    candidates: list[str],
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    encoded = model.target_tokenizer(
        candidates,
        truncation=True,
        max_length=model.config.max_target_length,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        scores = model.classify_candidates(
            embedding.to(device),
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )[0]
    order = torch.argsort(scores, descending=True).tolist()
    return [{"text": candidates[index], "score": float(scores[index].cpu())} for index in order]


def infer_vl_jepa_from_cli(
    checkpoint: str,
    visual_path: str,
    query: str,
    candidates: list[str] | None = None,
    output: str = "auto",
) -> str:
    model = load_vl_jepa_checkpoint(checkpoint)
    embedding = predict_vl_jepa_embedding(model, visual_path=visual_path, query=query)
    output = output.lower()
    if output not in {"auto", "embedding", "decode", "rank"}:
        raise ValueError("output must be one of: auto, embedding, decode, rank")
    if output == "rank" and not candidates:
        raise ValueError("--output rank requires at least one --candidate")
    if candidates and output in {"auto", "rank"}:
        ranked = rank_candidates(model, embedding, candidates)
        return "\n".join(f"{item['score']:.4f}\t{item['text']}" for item in ranked)
    if output == "decode":
        if not model.y_decoder.enabled:
            raise RuntimeError(
                "This checkpoint has no Y-Decoder configured, so it cannot generate text. "
                "Use --output embedding, pass --candidate labels for ranking, or train/configure "
                "a decoder_model before running --output decode."
            )
        return "\n".join(model.decode_embedding(embedding))
    if output == "auto" and model.y_decoder.enabled:
        return "\n".join(model.decode_embedding(embedding))
    if output == "auto":
        raise RuntimeError(
            "This checkpoint predicts embeddings, not text, because decoder_model is null. "
            "Use --output embedding to print the vector, or pass --candidate options to rank "
            "candidate captions/answers."
        )
    return " ".join(f"{value:.6f}" for value in embedding[0].detach().cpu().tolist())


def export_embeddings_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    model = load_vl_jepa_checkpoint(config["checkpoint"])
    records = [VLJEPASample.model_validate(record) for record in read_jsonl(config["input_file"])]
    output_path = Path(config.get("output_file", "runs/vl_jepa_embeddings.jsonl"))
    rows = []
    for record in records[: config.get("max_examples") or len(records)]:
        embedding = predict_vl_jepa_embedding(
            model,
            visual_path=record.visual_path or "",
            visual_kind=record.visual_kind,
            frame_paths=record.frame_paths,
            query=record.query,
        )
        rows.append({"id": record.id, "embedding": embedding[0].detach().cpu().tolist()})
    write_jsonl(rows, output_path)
    return output_path
