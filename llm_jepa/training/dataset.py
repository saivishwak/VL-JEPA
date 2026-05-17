from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from llm_jepa.modeling.special_tokens import predictor_suffix
from llm_jepa.modeling.tokenizer_setup import format_messages


class ChatJEPADataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        predictors: int = 0,
        regular: bool = False,
        plain: bool = False,
        reverse_pred: bool = False,
        front_pred: bool = False,
        train_all: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.predictors = predictors
        self.regular = regular
        self.plain = plain
        self.reverse_pred = reverse_pred
        self.front_pred = front_pred
        self.train_all = train_all
        with Path(path).open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize(self, text: str) -> dict[str, list[int]]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=True,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def _labels_for_full(self, input_ids: list[int], prompt_ids: list[int]) -> list[int]:
        if self.train_all:
            return list(input_ids)
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return labels

    def _views(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        user_messages = [message for message in messages if message["role"] != "assistant"]
        assistant_message = messages[-1]
        assistant_messages = [assistant_message]
        if self.reverse_pred:
            user_messages, assistant_messages = assistant_messages, user_messages[-1:]

        suffix = predictor_suffix(self.predictors)
        if suffix and user_messages:
            user_messages = [dict(message) for message in user_messages]
            if self.front_pred:
                user_messages[-1]["content"] = suffix + user_messages[-1]["content"]
            else:
                user_messages[-1]["content"] += suffix
        return user_messages, assistant_messages

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        messages = record["messages"]
        prompt_messages = messages[:-1]
        full_text = format_messages(messages, self.tokenizer, plain=self.plain)
        prompt_text = format_messages(prompt_messages, self.tokenizer, plain=self.plain)
        full = self._tokenize(full_text)
        prompt = self._tokenize(prompt_text)
        item: dict[str, Any] = {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": self._labels_for_full(full["input_ids"], prompt["input_ids"]),
        }
        if self.regular:
            return item

        user_messages, assistant_messages = self._views(messages)
        user = self._tokenize(format_messages(user_messages, self.tokenizer, plain=self.plain))
        assistant = self._tokenize(format_messages(assistant_messages, self.tokenizer, plain=self.plain))
        item.update(
            {
                "input_ids_user": user["input_ids"],
                "attention_mask_user": user["attention_mask"],
                "labels_user": [-100] * len(user["input_ids"]),
                "input_ids_assistant": assistant["input_ids"],
                "attention_mask_assistant": assistant["attention_mask"],
                "labels_assistant": [-100] * len(assistant["input_ids"]),
            }
        )
        return item
