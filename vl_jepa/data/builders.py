from __future__ import annotations

from pathlib import Path

from vl_jepa.data.manifests import PAPER_DATASETS, require_manifest, write_vl_jepa_manifest
from vl_jepa.data.schemas import VLJEPASample, to_vl_jepa_record
from vl_jepa.utils.io import read_jsonl, write_jsonl


def prepare_vl_jepa_manifest(
    *,
    dataset: str,
    split: str,
    task_type: str | None,
    source: str | None,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized = dataset.replace("-", "_").lower()
    spec = next((item for item in PAPER_DATASETS if item.name == normalized), None)
    resolved_task = task_type or (spec.task_type if spec else "captioning")

    if source is None:
        require_manifest(output / f"{normalized}_{split}.jsonl", normalized)
    source_path = Path(source or "")
    records = [
        to_vl_jepa_record(VLJEPASample.model_validate(record)) for record in read_jsonl(source_path)
    ]
    data_file = output / f"{normalized}_{split}.jsonl"
    write_jsonl(records, data_file)
    return write_vl_jepa_manifest(
        name=normalized,
        split=split,
        task_type=resolved_task,
        file=data_file,
        output_path=output / f"{normalized}_{split}_manifest.json",
        source=source,
        restricted=bool(spec.restricted) if spec else False,
    )
