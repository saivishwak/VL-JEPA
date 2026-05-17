from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vl_jepa.data.schemas import VLJEPAManifest, validate_vl_jepa_jsonl
from vl_jepa.utils.io import write_json


@dataclass(frozen=True)
class PaperDatasetSpec:
    name: str
    stage: str
    task_type: str
    restricted: bool = False


PAPER_DATASETS: tuple[PaperDatasetSpec, ...] = (
    PaperDatasetSpec("datacomp", "pretrain_image_1f", "captioning"),
    PaperDatasetSpec("yfcc100m", "pretrain_image_1f", "captioning"),
    PaperDatasetSpec("action100m", "pretrain_video", "captioning", restricted=True),
    PaperDatasetSpec("plm_vqa_25m", "sft", "vqa", restricted=True),
    PaperDatasetSpec("plm_captioning_2_8m", "sft", "captioning", restricted=True),
    PaperDatasetSpec("plm_classification_1_8m", "sft", "classification", restricted=True),
    PaperDatasetSpec("ssv2", "eval", "classification"),
    PaperDatasetSpec("ek100", "eval", "classification"),
    PaperDatasetSpec("egoexo4d", "eval", "classification"),
    PaperDatasetSpec("kinetics400", "eval", "classification"),
    PaperDatasetSpec("coin_step", "eval", "classification"),
    PaperDatasetSpec("coin_task", "eval", "classification"),
    PaperDatasetSpec("crosstask_step", "eval", "classification"),
    PaperDatasetSpec("crosstask_task", "eval", "classification"),
    PaperDatasetSpec("msr_vtt", "eval", "retrieval"),
    PaperDatasetSpec("activitynet", "eval", "retrieval"),
    PaperDatasetSpec("didemo", "eval", "retrieval"),
    PaperDatasetSpec("msvd", "eval", "retrieval"),
    PaperDatasetSpec("youcook2", "eval", "retrieval"),
    PaperDatasetSpec("pvd_bench", "eval", "retrieval"),
    PaperDatasetSpec("dream_1k", "eval", "retrieval"),
    PaperDatasetSpec("vdc_1k", "eval", "retrieval"),
    PaperDatasetSpec("gqa", "eval", "vqa"),
    PaperDatasetSpec("tallyqa", "eval", "vqa"),
    PaperDatasetSpec("pope", "eval", "vqa"),
    PaperDatasetSpec("popev2", "eval", "vqa"),
    PaperDatasetSpec("worldprediction_wm", "eval", "world_prediction"),
    PaperDatasetSpec("epic_kitchens_100", "eval", "action_anticipation"),
    PaperDatasetSpec("sugarcrepe_pp", "eval", "text_triplet"),
    PaperDatasetSpec("visla", "eval", "text_triplet"),
)


def write_vl_jepa_manifest(
    *,
    name: str,
    split: str,
    task_type: str,
    file: str | Path,
    output_path: str | Path,
    source: str | None = None,
    restricted: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Path:
    examples = validate_vl_jepa_jsonl(file)
    manifest = VLJEPAManifest(
        name=name,
        split=split,
        task_type=task_type,  # type: ignore[arg-type]
        file=str(file),
        examples=examples,
        source=source,
        restricted=restricted,
        metadata=metadata or {},
    )
    target = Path(output_path)
    write_json(manifest.model_dump(mode="json"), target)
    return target


def require_manifest(path: str | Path, dataset_name: str) -> Path:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        known = ", ".join(spec.name for spec in PAPER_DATASETS)
        raise FileNotFoundError(
            f"VL-JEPA dataset '{dataset_name}' requires an explicit manifest at {manifest_path}. "
            f"Known paper dataset keys: {known}"
        )
    return manifest_path
