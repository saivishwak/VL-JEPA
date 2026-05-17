from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VLJEPATaskType = Literal[
    "captioning",
    "classification",
    "retrieval",
    "vqa",
    "selective_decoding",
    "text_triplet",
    "world_prediction",
    "action_anticipation",
]
VisualKind = Literal["image", "video", "frames"]


class CandidateText(BaseModel):
    id: str | None = None
    text: str
    label: str | None = None
    is_correct: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate text cannot be empty")
        return value


class VLJEPASample(BaseModel):
    """Paper-faithful VL-JEPA triplet: <X_V, X_Q, Y> plus optional candidates."""

    id: str
    task_type: VLJEPATaskType
    visual_kind: VisualKind
    visual_path: str | None = None
    frame_paths: list[str] | None = None
    query: str = ""
    target: str | None = None
    candidates: list[CandidateText] = Field(default_factory=list)
    timestamps: list[float] | None = None
    source_dataset: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_visual_and_target(self) -> VLJEPASample:
        if self.visual_kind == "frames":
            if not self.frame_paths:
                raise ValueError("frames samples require frame_paths")
        elif not self.visual_path:
            raise ValueError("image and video samples require visual_path")
        if self.task_type not in {"classification", "vqa", "world_prediction"} and not self.target:
            raise ValueError(f"{self.task_type} samples require target text")
        if self.task_type in {"classification", "vqa", "world_prediction"} and not (
            self.target or self.candidates
        ):
            raise ValueError(f"{self.task_type} samples require a target or candidates")
        return self


class VLJEPAManifest(BaseModel):
    name: str
    split: str
    task_type: VLJEPATaskType
    file: str
    examples: int | None = None
    source: str | None = None
    restricted: bool = False
    checksum: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_vl_jepa_jsonl(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                VLJEPASample.model_validate_json(line)
                count += 1
    return count


def to_vl_jepa_record(example: VLJEPASample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(example, VLJEPASample):
        return example.model_dump(mode="json", exclude_none=True)
    return VLJEPASample.model_validate(example).model_dump(mode="json", exclude_none=True)
