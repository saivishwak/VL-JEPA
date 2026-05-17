from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLJEPAConfig:
    vision_encoder: str = "facebook/vjepa2-vitl-fpc64-256"
    predictor_model: str = "meta-llama/Llama-3.2-1B"
    target_encoder: str = "google/embeddinggemma-300m"
    decoder_model: str | None = None
    embedding_dim: int = 1536
    predictor_layers: int = 8
    max_query_length: int = 512
    max_target_length: int = 512
    image_size: int = 256
    num_frames: int = 8
    freeze_x_encoder: bool = True
    y_encoder_lr_multiplier: float = 0.05
    temperature: float = 0.07
    trust_remote_code: bool = True
    hf_token: bool | str | None = True
    torch_dtype: str | None = "auto"
    tiny: bool = False
