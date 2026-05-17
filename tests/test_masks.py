import torch

from llm_jepa.training.masks import block_diagonal_causal_mask, causal_additive_mask


def test_causal_additive_mask_blocks_future_tokens():
    mask = causal_additive_mask(3)
    assert mask[0, 0] == 0
    assert torch.isinf(mask[0, 1])
    assert mask[2, 0] == 0


def test_block_diagonal_mask_blocks_cross_block_attention():
    mask = block_diagonal_causal_mask(5, [(0, 2), (2, 5)])
    assert torch.isinf(mask[0, 2])
    assert torch.isinf(mask[2, 1])
    assert mask[4, 2] == 0
