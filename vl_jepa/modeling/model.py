from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vl_jepa.modeling.config import VLJEPAConfig
from vl_jepa.modeling.predictor import Predictor
from vl_jepa.modeling.x_encoder import XEncoder
from vl_jepa.modeling.y_decoder import YDecoder
from vl_jepa.modeling.y_encoder import YEncoder
from vl_jepa.training.losses import bidirectional_infonce_loss


class VLJEPA(nn.Module):
    def __init__(self, config: VLJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.x_encoder = XEncoder(
            config.vision_encoder,
            output_dim=config.embedding_dim,
            freeze=config.freeze_x_encoder,
            trust_remote_code=config.trust_remote_code,
            hf_token=config.hf_token,
            torch_dtype=config.torch_dtype,
            tiny=config.tiny,
        )
        self.predictor = Predictor(
            config.predictor_model,
            vision_dim=config.embedding_dim,
            output_dim=config.embedding_dim,
            num_layers=config.predictor_layers,
            trust_remote_code=config.trust_remote_code,
            hf_token=config.hf_token,
            torch_dtype=config.torch_dtype,
            tiny=config.tiny,
        )
        self.y_encoder = YEncoder(
            config.target_encoder,
            output_dim=config.embedding_dim,
            trust_remote_code=config.trust_remote_code,
            hf_token=config.hf_token,
            torch_dtype=config.torch_dtype,
            tiny=config.tiny,
        )
        self.y_decoder = YDecoder(
            config.decoder_model,
            embedding_dim=config.embedding_dim,
            trust_remote_code=config.trust_remote_code,
            hf_token=config.hf_token,
            torch_dtype=config.torch_dtype,
        )

    @property
    def query_tokenizer(self):
        return self.predictor.query_tokenizer

    @property
    def target_tokenizer(self):
        return self.y_encoder.target_tokenizer

    def predict_embedding(
        self,
        pixel_values: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_tokens = self.x_encoder(pixel_values)
        return self.predictor(visual_tokens, query_input_ids, query_attention_mask)

    def encode_target(
        self, target_input_ids: torch.Tensor, target_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.y_encoder(target_input_ids, target_attention_mask)

    def forward_embeddings(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predicted = self.predict_embedding(
            batch["pixel_values"],
            batch["query_input_ids"],
            batch["query_attention_mask"],
        )
        target = self.encode_target(batch["target_input_ids"], batch["target_attention_mask"])
        return {"predicted_embedding": predicted, "target_embedding": target}

    def forward_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = self.forward_embeddings(batch)
        predicted = embeddings["predicted_embedding"]
        target = embeddings["target_embedding"]
        loss = bidirectional_infonce_loss(predicted, target, temperature=self.config.temperature)
        return {"loss": loss, "predicted_embedding": predicted, "target_embedding": target}

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_loss(batch)

    @torch.no_grad()
    def classify_candidates(
        self,
        predicted_embedding: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = self.encode_target(candidate_input_ids, candidate_attention_mask)
        predicted_norm = F.normalize(predicted_embedding, p=2, dim=-1)
        candidate_norm = F.normalize(candidate_embeddings, p=2, dim=-1)
        return predicted_norm @ candidate_norm.T

    @torch.no_grad()
    def decode_embedding(self, embedding: torch.Tensor, *, max_new_tokens: int = 64) -> list[str]:
        return self.y_decoder.decode(embedding, max_new_tokens=max_new_tokens)


def build_vl_jepa_model(config: dict) -> VLJEPA:
    model_cfg = config.get("model", {})
    return VLJEPA(VLJEPAConfig(**model_cfg))
