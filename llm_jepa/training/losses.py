from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_loss(
    user_embedding: torch.Tensor,
    assistant_embedding: torch.Tensor,
    mode: str = "cosine",
    temperature: float = 0.07,
) -> torch.Tensor:
    if mode == "cosine":
        return 1.0 - F.cosine_similarity(user_embedding, assistant_embedding, dim=-1).mean()
    if mode == "mse":
        return F.mse_loss(user_embedding, assistant_embedding)
    if mode == "l2":
        return torch.linalg.norm(user_embedding - assistant_embedding, ord=2, dim=-1).mean()
    if mode == "infonce":
        user_norm = F.normalize(user_embedding, p=2, dim=-1)
        assistant_norm = F.normalize(assistant_embedding, p=2, dim=-1)
        logits = torch.mm(user_norm, assistant_norm.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
    raise ValueError(f"Unsupported JEPA loss mode: {mode}")
