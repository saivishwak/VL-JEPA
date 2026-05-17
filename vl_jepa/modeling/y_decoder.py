from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vl_jepa.modeling.hf_auth import resolve_hf_token
from vl_jepa.modeling.tokenizer import SimpleTokenizer


class TinyCausalDecoder(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 64) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = type("TinyDecoderConfig", (), {"hidden_size": hidden_size})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        hidden, _ = self.gru(inputs_embeds)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("TinyCausalDecoderOutput", (), {"loss": loss, "logits": logits})()

    @torch.no_grad()
    def generate(
        self,
        *,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 64,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        batch = inputs_embeds.shape[0]
        token = torch.full(
            (batch, 1),
            eos_token_id if eos_token_id is not None else 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return token.repeat(1, max_new_tokens)


class YDecoder(nn.Module):
    """Optional readout module; it is intentionally outside the VL-JEPA training loss."""

    def __init__(
        self,
        model_name: str | None,
        *,
        embedding_dim: int,
        trust_remote_code: bool = True,
        hf_token: bool | str | None = True,
        torch_dtype: str | None = "auto",
    ) -> None:
        super().__init__()
        self.enabled = model_name is not None
        self.model = None
        self.tokenizer = None
        self.prefix_projection: nn.Linear | None = None
        if model_name is None:
            return
        if model_name == "tiny":
            self.model = TinyCausalDecoder(hidden_size=embedding_dim)
            self.tokenizer = SimpleTokenizer()
            self.prefix_projection = nn.Linear(embedding_dim, embedding_dim)
            return
        token = resolve_hf_token(hf_token)
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "token": token}
        if torch_dtype:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        hidden_size = int(getattr(self.model.config, "hidden_size", embedding_dim))
        self.prefix_projection = nn.Linear(embedding_dim, hidden_size)

    def input_embeddings(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("VL-JEPA Y-Decoder is not configured")
        embeddings = getattr(self.model, "get_input_embeddings", lambda: None)()
        if embeddings is None and hasattr(self.model, "model"):
            embeddings = self.model.model.embed_tokens
        if embeddings is None:
            raise RuntimeError("Could not locate decoder token embeddings")
        return embeddings

    def forward_loss(
        self,
        embedding: torch.Tensor,
        labels_input_ids: torch.Tensor,
        labels_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.model is None
            or self.tokenizer is None
            or self.prefix_projection is None
        ):
            raise RuntimeError("VL-JEPA Y-Decoder is not configured")
        prefix = self.prefix_projection(
            embedding.to(dtype=self.prefix_projection.weight.dtype)
        ).unsqueeze(1)
        token_embeds = self.input_embeddings()(labels_input_ids)
        prefix = prefix.to(dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones(
            labels_attention_mask.shape[0],
            1,
            dtype=labels_attention_mask.dtype,
            device=labels_attention_mask.device,
        )
        attention_mask = torch.cat([prefix_mask, labels_attention_mask], dim=1)
        labels = labels_input_ids.masked_fill(labels_attention_mask == 0, -100)
        labels = torch.cat(
            [torch.full_like(labels[:, :1], -100), labels],
            dim=1,
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        if outputs.loss is None:
            raise RuntimeError("Decoder did not return a language-modeling loss")
        return outputs.loss

    @torch.no_grad()
    def decode(self, embedding: torch.Tensor, *, max_new_tokens: int = 64) -> list[str]:
        if (
            not self.enabled
            or self.model is None
            or self.tokenizer is None
            or self.prefix_projection is None
        ):
            raise RuntimeError("VL-JEPA Y-Decoder is not configured")
        prefix = self.prefix_projection(
            embedding.to(dtype=self.prefix_projection.weight.dtype)
        ).unsqueeze(1)
        prefix = prefix.to(dtype=self.input_embeddings().weight.dtype)
        generated = self.model.generate(
            inputs_embeds=prefix,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]
