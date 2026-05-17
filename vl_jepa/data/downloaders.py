from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vl_jepa.data.manifests import write_vl_jepa_manifest
from vl_jepa.data.schemas import VLJEPASample, to_vl_jepa_record
from vl_jepa.utils.io import write_jsonl


@dataclass(frozen=True)
class OpenDatasetPreset:
    name: str
    hf_dataset: str
    task_type: str
    visual_kind: str
    image_column: str = "image"
    query_column: str | None = None
    target_column: str | None = None
    captions_column: str | None = None
    question_column: str | None = None
    answers_column: str | None = None
    candidates_column: str | None = None
    config_name: str | None = None
    revision: str | None = None
    split_aliases: dict[str, str] | None = None
    trust_remote_code: bool = False


OPEN_DATASET_PRESETS: dict[str, OpenDatasetPreset] = {
    "flickr30k": OpenDatasetPreset(
        name="flickr30k",
        hf_dataset="lmms-lab/flickr30k",
        task_type="captioning",
        visual_kind="image",
        image_column="image",
        captions_column="caption",
        config_name="default",
        revision="refs/convert/parquet",
    ),
    "coco_captions": OpenDatasetPreset(
        name="coco_captions",
        hf_dataset="lmms-lab/COCO-Caption2017",
        task_type="captioning",
        visual_kind="image",
        image_column="image",
        captions_column="captions",
        config_name="default",
        revision="refs/convert/parquet",
    ),
    "vqav2": OpenDatasetPreset(
        name="vqav2",
        hf_dataset="lmms-lab/VQAv2",
        task_type="vqa",
        visual_kind="image",
        image_column="image",
        question_column="question",
        answers_column="answers",
        config_name="default",
        revision="refs/convert/parquet",
        split_aliases={"val": "validation", "val2014": "validation"},
    ),
    "ok_vqa": OpenDatasetPreset(
        name="ok_vqa",
        hf_dataset="lmms-lab/OK-VQA",
        task_type="vqa",
        visual_kind="image",
        image_column="image",
        question_column="question",
        answers_column="answers",
        config_name="default",
        revision="refs/convert/parquet",
        split_aliases={"val": "validation", "val2014": "validation"},
    ),
}


def preset_names() -> list[str]:
    return sorted(OPEN_DATASET_PRESETS)


def resolve_split(preset: OpenDatasetPreset, split: str) -> str:
    return (preset.split_aliases or {}).get(split, split)


def _first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = _first_present(value, ["text", "answer", "caption", "raw"])
        return _coerce_text(text if text is not None else next(iter(value.values()), ""))
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _coerce_text(value[0])
    return str(value)


def _coerce_many_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("text", "answer", "answers", "caption", "captions", "raw"):
            if key in value:
                return _coerce_many_texts(value[key])
        return [_coerce_text(item) for item in value.values()]
    if isinstance(value, (list, tuple)):
        return [text for item in value if (text := _coerce_text(item))]
    return [_coerce_text(value)]


def _save_visual(value: Any, output_dir: Path, dataset_name: str, sample_id: str) -> str:
    image_dir = output_dir / dataset_name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "save"):
        target = image_dir / f"{sample_id}.jpg"
        value.convert("RGB").save(target)
        return str(target)
    path = Path(str(value)).expanduser()
    if path.is_file():
        target = image_dir / f"{sample_id}{path.suffix or '.jpg'}"
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        return str(target)
    raise ValueError(f"Cannot materialize visual input for sample {sample_id}: {value!r}")


def _candidate_records(candidates: list[str], target: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    records = []
    for candidate in [target, *candidates]:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        records.append({"text": candidate, "is_correct": candidate == target})
    return records


def record_to_vl_jepa_sample(
    record: dict[str, Any],
    *,
    preset: OpenDatasetPreset,
    output_dir: str | Path,
    split: str,
    index: int,
) -> VLJEPASample:
    sample_id = str(_first_present(record, ["id", "question_id", "image_id"]) or f"{split}-{index}")
    visual_value = record[preset.image_column]
    visual_path = _save_visual(visual_value, Path(output_dir), preset.name, sample_id)
    query = _coerce_text(record.get(preset.query_column or "")).strip()

    if preset.task_type == "captioning":
        captions = _coerce_many_texts(record.get(preset.captions_column or ""))
        target = captions[0] if captions else _coerce_text(record.get(preset.target_column or ""))
        query = query or "Caption the image."
        candidates: list[dict[str, Any]] = []
    elif preset.task_type == "vqa":
        query = query or _coerce_text(record.get(preset.question_column or ""))
        answers = _coerce_many_texts(record.get(preset.answers_column or ""))
        target = answers[0] if answers else _coerce_text(record.get(preset.target_column or ""))
        candidates = _candidate_records(
            _coerce_many_texts(record.get(preset.candidates_column or "")),
            target,
        )
    else:
        target = _coerce_text(record.get(preset.target_column or ""))
        candidates = []

    return VLJEPASample(
        id=f"{preset.name}-{sample_id}",
        task_type=preset.task_type,  # type: ignore[arg-type]
        visual_kind=preset.visual_kind,  # type: ignore[arg-type]
        visual_path=visual_path,
        query=query,
        target=target,
        candidates=candidates,
        source_dataset=preset.name,
        split=split,
        metadata={"hf_dataset": preset.hf_dataset},
    )


def download_open_dataset(
    *,
    dataset: str,
    split: str,
    output_dir: str | Path = "data/vl_jepa",
    max_examples: int | None = None,
    streaming: bool = False,
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Install the `datasets` package to download open VL datasets") from exc

    if dataset not in OPEN_DATASET_PRESETS:
        known = ", ".join(preset_names())
        raise ValueError(f"Unknown open dataset preset '{dataset}'. Available presets: {known}")
    preset = OPEN_DATASET_PRESETS[dataset]
    resolved_split = resolve_split(preset, split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        loaded = load_dataset(
            preset.hf_dataset,
            preset.config_name,
            split=resolved_split,
            streaming=streaming,
            revision=preset.revision,
            trust_remote_code=preset.trust_remote_code,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "This dataset resolved to an old Hugging Face dataset script. "
                "Use one of the script-free presets from `vl-jepa download-open --help`."
            ) from exc
        raise

    records = []
    for index, record in enumerate(loaded):
        if max_examples is not None and index >= max_examples:
            break
        sample = record_to_vl_jepa_sample(
            dict(record),
            preset=preset,
            output_dir=output,
            split=split,
            index=index,
        )
        records.append(to_vl_jepa_record(sample))

    data_file = output / f"{preset.name}_{resolved_split}.jsonl"
    write_jsonl(records, data_file)
    return write_vl_jepa_manifest(
        name=preset.name,
        split=resolved_split,
        task_type=preset.task_type,
        file=data_file,
        output_path=output / f"{preset.name}_{resolved_split}_manifest.json",
        source=preset.hf_dataset,
        restricted=False,
    )
