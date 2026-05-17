import pytest
import torch

from vl_jepa.training.losses import bidirectional_infonce_loss, infonce_loss


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
