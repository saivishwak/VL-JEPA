from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    predicted_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if predicted_embedding.size(0) < 2:
        raise ValueError("InfoNCE requires at least two samples in the contrastive batch")
    predicted_norm = F.normalize(predicted_embedding, p=2, dim=-1)
    target_norm = F.normalize(target_embedding, p=2, dim=-1)
    logits = torch.mm(predicted_norm, target_norm.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def _multi_positive_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must have the same shape as logits")
    if not positive_mask.any(dim=1).all():
        raise ValueError("each row must have at least one positive")
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probs = log_probs.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    return -torch.logsumexp(positive_log_probs, dim=1).mean()


def bidirectional_infonce_loss(
    predicted_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if predicted_embedding.size(0) < 2:
        raise ValueError("InfoNCE requires at least two samples in the contrastive batch")
    predicted_norm = F.normalize(predicted_embedding, p=2, dim=-1)
    target_norm = F.normalize(target_embedding, p=2, dim=-1)
    logits = torch.mm(predicted_norm, target_norm.T) / temperature
    if positive_mask is None:
        labels = torch.arange(logits.size(0), device=logits.device)
        forward = F.cross_entropy(logits, labels)
        backward = F.cross_entropy(logits.T, labels)
    else:
        forward = _multi_positive_nce(logits, positive_mask)
        backward = _multi_positive_nce(logits.T, positive_mask.T)
    return 0.5 * (forward + backward)
