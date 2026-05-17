from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class VLJEPADataCollator:
    query_tokenizer: PreTrainedTokenizerBase
    target_tokenizer: PreTrainedTokenizerBase

    def _pad(self, values: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(value) for value in values)
        if max_len == 0:
            return torch.empty((len(values), 0), dtype=torch.long)
        padded = [value + [pad_value] * (max_len - len(value)) for value in values]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        query_pad = self.query_tokenizer.pad_token_id or 0
        target_pad = self.target_tokenizer.pad_token_id or 0
        query_ids = [item["query_input_ids"] for item in features]
        query_masks = [item["query_attention_mask"] for item in features]
        target_ids = [item["target_input_ids"] for item in features]
        target_masks = [item["target_attention_mask"] for item in features]
        batch: dict[str, Any] = {
            "query_input_ids": self._pad(query_ids, query_pad),
            "query_attention_mask": self._pad(query_masks, 0),
            "target_input_ids": self._pad(target_ids, target_pad),
            "target_attention_mask": self._pad(target_masks, 0),
            "ids": [item["id"] for item in features],
            "task_types": [item["task_type"] for item in features],
            "target_texts": [item["target_text"] for item in features],
            "candidates": [item.get("candidates", []) for item in features],
            "metadata": [item.get("metadata", {}) for item in features],
        }
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([item["pixel_values"] for item in features], dim=0)
        return batch
