from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from vl_jepa.modeling.hf_auth import resolve_hf_token


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, hidden_size, kernel_size=(1, 16, 16), stride=(1, 16, 16))
        self.norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Input is [batch, frames, channels, height, width].
        x = pixel_values.permute(0, 2, 1, 3, 4)
        x = self.conv(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class XEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        output_dim: int,
        freeze: bool = True,
        trust_remote_code: bool = True,
        hf_token: bool | str | None = True,
        torch_dtype: str | None = "auto",
        tiny: bool = False,
    ) -> None:
        super().__init__()
        if tiny:
            self.encoder = TinyVisionEncoder()
            hidden_size = self.encoder.hidden_size
        else:
            kwargs: dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
                "token": resolve_hf_token(hf_token),
            }
            if torch_dtype:
                kwargs["torch_dtype"] = torch_dtype
            self.encoder = AutoModel.from_pretrained(model_name, **kwargs)
            hidden_size = int(getattr(self.encoder.config, "hidden_size", output_dim))
        self.projection = nn.Linear(hidden_size, output_dim)
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _run_encoder(self, pixel_values: torch.Tensor):
        signature = inspect.signature(self.encoder.forward)
        if "pixel_values_videos" in signature.parameters:
            return self.encoder(pixel_values_videos=pixel_values, output_hidden_states=True)
        if "pixel_values_images" in signature.parameters:
            return self.encoder(pixel_values_images=pixel_values, output_hidden_states=True)
        return self.encoder(pixel_values=pixel_values, output_hidden_states=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, TinyVisionEncoder):
            hidden = self.encoder(pixel_values)
        else:
            expects_video = (
                "pixel_values_videos" in inspect.signature(self.encoder.forward).parameters
            )
            flat_pixels = pixel_values
            if pixel_values.ndim == 5 and not expects_video:
                batch, frames, channels, height, width = pixel_values.shape
                flat_pixels = pixel_values.view(batch * frames, channels, height, width)
            outputs = self._run_encoder(flat_pixels)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs.hidden_states[-1]
            if pixel_values.ndim == 5 and not expects_video:
                hidden = hidden.view(batch, frames * hidden.shape[1], hidden.shape[-1])
        return self.projection(hidden.to(dtype=self.projection.weight.dtype))
