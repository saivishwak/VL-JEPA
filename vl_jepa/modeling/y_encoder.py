from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from vl_jepa.modeling.hf_auth import resolve_hf_token
from vl_jepa.modeling.tokenizer import SimpleTokenizer


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 64) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.config = type("TinyConfig", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.embed_tokens(input_ids))
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class YEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        output_dim: int,
        trust_remote_code: bool = True,
        hf_token: bool | str | None = True,
        torch_dtype: str | None = "auto",
        tiny: bool = False,
    ) -> None:
        super().__init__()
        if tiny:
            self.encoder = TinyTextEncoder()
            self.tokenizer = SimpleTokenizer()
        else:
            token = resolve_hf_token(hf_token)
            kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "token": token}
            if torch_dtype:
                kwargs["torch_dtype"] = torch_dtype
            self.encoder = AutoModel.from_pretrained(model_name, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, token=token
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        hidden_size = int(getattr(self.encoder.config, "hidden_size", output_dim))
        self.projection = nn.Linear(hidden_size, output_dim)

    @property
    def target_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, TinyTextEncoder):
            pooled = self.encoder(input_ids, attention_mask)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled.to(dtype=self.projection.weight.dtype))
