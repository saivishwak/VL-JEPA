from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class JEPADataCollator:
    tokenizer: PreTrainedTokenizerBase

    def _pad(self, values: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(value) for value in values)
        padded = [value + [pad_value] * (max_len - len(value)) for value in values]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id or 0
        keys = features[0].keys()
        batch: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [feature[key] for feature in features]
            if key.startswith("labels"):
                batch[key] = self._pad(values, -100)
            elif key.startswith("attention_mask"):
                batch[key] = self._pad(values, 0)
            else:
                batch[key] = self._pad(values, pad_id)
        return batch
