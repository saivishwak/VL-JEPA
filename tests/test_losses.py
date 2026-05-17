import torch

from llm_jepa.training.losses import jepa_loss


def test_cosine_loss_is_small_for_matching_embeddings():
    values = torch.randn(4, 8)
    loss = jepa_loss(values, values, mode="cosine")
    assert loss.item() < 1e-6


def test_infonce_loss_is_finite():
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)
    assert torch.isfinite(jepa_loss(left, right, mode="infonce"))
