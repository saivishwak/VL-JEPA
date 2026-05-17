from __future__ import annotations

import torch
from transformers import Trainer

from llm_jepa.training.losses import jepa_loss
from llm_jepa.training.masks import block_diagonal_causal_mask, causal_additive_mask


class LLMJEPATrainer(Trainer):
    def __init__(
        self,
        *args,
        lbd: float = 1.0,
        gamma: float = 1.0,
        last_token: int = -1,
        loss_mode: str = "cosine",
        additive_mask: bool = False,
        jepa_ratio: float = -1.0,
        **kwargs,
    ) -> None:
        self.lbd = lbd
        self.gamma = gamma
        self.last_token = last_token
        self.loss_mode = loss_mode
        self.additive_mask = additive_mask
        self.jepa_ratio = jepa_ratio
        self._last_token_user: torch.Tensor | None = None
        self._last_token_assistant: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def _last_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1)
        return torch.clamp(lengths + self.last_token, min=0)

    def _pad_to_length(
        self,
        tensor: torch.Tensor,
        length: int,
        pad_value: int,
    ) -> torch.Tensor:
        if tensor.shape[1] >= length:
            return tensor[:, :length].clone()
        pad = torch.full(
            (tensor.shape[0], length - tensor.shape[1]),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=1)

    def _pad_token_id(self, model) -> int:
        pad_token_id = getattr(model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "eos_token_id", 0)
        return int(pad_token_id or 0)

    def _run_standard_forward(self, model, inputs):
        model_inputs = {
            "input_ids": torch.cat(
                [inputs["input_ids"], inputs["input_ids_user"], inputs["input_ids_assistant"]],
                dim=0,
            ),
            "attention_mask": torch.cat(
                [
                    inputs["attention_mask"],
                    inputs["attention_mask_user"],
                    inputs["attention_mask_assistant"],
                ],
                dim=0,
            ),
            "labels": torch.cat(
                [inputs["labels"], inputs["labels_user"], inputs["labels_assistant"]],
                dim=0,
            ),
        }
        outputs = model(**model_inputs, output_hidden_states=True)
        batch_size = inputs["input_ids"].shape[0]
        return outputs, outputs.hidden_states[-1][batch_size : batch_size * 2], outputs.hidden_states[-1][batch_size * 2 :]

    def _run_additive_forward(self, model, inputs):
        if self.jepa_ratio > 0.0 and torch.rand(1).item() > self.jepa_ratio:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                output_hidden_states=True,
            )
            return outputs, None, None

        batch_size, seq_length = inputs["input_ids"].shape
        device = inputs["input_ids"].device
        user_ids = self._pad_to_length(inputs["input_ids_user"], seq_length, self._pad_token_id(model))
        user_labels = self._pad_to_length(inputs["labels_user"], seq_length, -100)
        mask = torch.full(
            (batch_size * 2, 1, seq_length, seq_length),
            -torch.inf,
            device=device,
            dtype=torch.float32,
        )
        full_lengths = inputs["attention_mask"].long().sum(dim=1)
        user_lengths = inputs["attention_mask_user"].long().sum(dim=1)
        assistant_lengths = inputs["attention_mask_assistant"].long().sum(dim=1)
        full_last = torch.clamp(full_lengths + self.last_token, min=0, max=seq_length - 1)
        user_last = torch.clamp(user_lengths + self.last_token, min=0, max=seq_length - 1)
        assistant_last = torch.clamp(assistant_lengths + self.last_token, min=0)
        assistant_targets = torch.zeros_like(user_last)
        for idx in range(batch_size):
            full_len = min(int(full_lengths[idx].item()), seq_length)
            user_len = min(int(user_lengths[idx].item()), seq_length)
            assistant_len = min(int(assistant_lengths[idx].item()), inputs["input_ids_assistant"].shape[1])
            copy_len = min(assistant_len, seq_length - user_len)
            if copy_len > 0:
                user_ids[idx, user_len : user_len + copy_len] = inputs["input_ids_assistant"][idx, :copy_len]
                user_labels[idx, user_len : user_len + copy_len] = inputs["labels_assistant"][idx, :copy_len]
                assistant_targets[idx] = user_len + min(int(assistant_last[idx].item()), copy_len - 1)
            else:
                assistant_targets[idx] = user_len
            mask[idx, :, :full_len, :full_len] = causal_additive_mask(full_len, device)
            mask[idx + batch_size, 0] = block_diagonal_causal_mask(
                seq_length,
                [(0, user_len), (user_len, user_len + copy_len)],
                device,
            )
        self._last_token_user = user_last
        self._last_token_assistant = torch.clamp(assistant_targets, max=seq_length - 1)
        outputs = model(
            input_ids=torch.cat([inputs["input_ids"], user_ids], dim=0),
            attention_mask=mask,
            labels=torch.cat([inputs["labels"], user_labels], dim=0),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1][batch_size:]
        return outputs, hidden, hidden

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.additive_mask:
            outputs, user_hidden, assistant_hidden = self._run_additive_forward(model, inputs)
            if user_hidden is not None:
                user_idx = self._last_token_user
                assistant_idx = self._last_token_assistant
            else:
                user_idx = assistant_idx = None
        else:
            outputs, user_hidden, assistant_hidden = self._run_standard_forward(model, inputs)
            user_idx = self._last_indices(inputs["attention_mask_user"])
            assistant_idx = self._last_indices(inputs["attention_mask_assistant"])

        lm_loss = outputs.loss
        rep_loss = torch.zeros((), device=lm_loss.device, dtype=lm_loss.dtype)
        if user_hidden is not None and assistant_hidden is not None and user_idx is not None:
            rows = torch.arange(user_hidden.shape[0], device=user_hidden.device)
            user_embedding = user_hidden[rows, user_idx, :]
            assistant_embedding = assistant_hidden[rows, assistant_idx, :]
            rep_loss = jepa_loss(user_embedding, assistant_embedding, mode=self.loss_mode)
        total = self.gamma * lm_loss + self.lbd * rep_loss
        return (total, outputs) if return_outputs else total
