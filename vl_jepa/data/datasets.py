from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from vl_jepa.data.schemas import VLJEPASample
from vl_jepa.data.video import load_visual_tensor


class VLJEPAManifestDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        query_tokenizer: PreTrainedTokenizerBase,
        target_tokenizer: PreTrainedTokenizerBase,
        *,
        num_frames: int,
        image_size: int = 256,
        max_query_length: int = 512,
        max_target_length: int = 512,
        query_override: str | None = None,
        load_pixels: bool = True,
    ) -> None:
        self.path = Path(path)
        self.query_tokenizer = query_tokenizer
        self.target_tokenizer = target_tokenizer
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_query_length = max_query_length
        self.max_target_length = max_target_length
        self.query_override = query_override
        self.load_pixels = load_pixels
        with self.path.open("r", encoding="utf-8") as handle:
            self.records = [
                VLJEPASample.model_validate(json.loads(line)) for line in handle if line.strip()
            ]

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text: str,
        max_length: int,
    ) -> dict[str, Any]:
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.records[index]
        query_text = sample.query if self.query_override is None else self.query_override
        if query_text == "":
            query = {"input_ids": [], "attention_mask": []}
        else:
            query = self._tokenize(self.query_tokenizer, query_text, self.max_query_length)
        target_text = sample.target or next(
            (candidate.text for candidate in sample.candidates if candidate.is_correct),
            sample.candidates[0].text if sample.candidates else "",
        )
        target = self._tokenize(self.target_tokenizer, target_text, self.max_target_length)
        item: dict[str, Any] = {
            "id": sample.id,
            "task_type": sample.task_type,
            "query_input_ids": query["input_ids"],
            "query_attention_mask": query["attention_mask"],
            "target_input_ids": target["input_ids"],
            "target_attention_mask": target["attention_mask"],
            "target_text": target_text,
            "candidates": [candidate.model_dump(mode="json") for candidate in sample.candidates],
            "metadata": sample.metadata,
        }
        if self.load_pixels:
            item["pixel_values"] = load_visual_tensor(
                visual_kind=sample.visual_kind,
                visual_path=sample.visual_path,
                frame_paths=sample.frame_paths,
                num_frames=self.num_frames,
                image_size=self.image_size,
            )
        return item
