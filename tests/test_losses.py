import pytest
import torch

from vl_jepa.training.losses import bidirectional_infonce_loss, infonce_loss
from vl_jepa.training.train import _embedding_retrieval_metrics


def test_infonce_loss_is_finite():
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)
    assert torch.isfinite(infonce_loss(left, right))


def test_bidirectional_infonce_is_symmetric():
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)
    forward = bidirectional_infonce_loss(left, right)
    backward = bidirectional_infonce_loss(right, left)
    assert torch.allclose(forward, backward)


def test_infonce_rejects_singleton_batch():
    with pytest.raises(ValueError, match="at least two samples"):
        bidirectional_infonce_loss(torch.randn(1, 8), torch.randn(1, 8))


def test_bidirectional_infonce_accepts_multi_positive_mask():
    left = torch.randn(3, 8)
    right = torch.randn(3, 8)
    positive_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]
    )

    loss = bidirectional_infonce_loss(left, right, positive_mask=positive_mask)

    assert torch.isfinite(loss)


def test_embedding_retrieval_metrics_reward_correct_top1():
    embeddings = torch.eye(3)

    metrics = _embedding_retrieval_metrics(embeddings, embeddings)

    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0
