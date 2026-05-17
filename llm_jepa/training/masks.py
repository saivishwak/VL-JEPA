from __future__ import annotations

import torch


def causal_additive_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((length, length), dtype=torch.float32, device=device)
    upper = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
    mask[upper] = -torch.inf
    return mask


def block_diagonal_causal_mask(
    full_length: int,
    blocks: list[tuple[int, int]],
    device: torch.device | None = None,
) -> torch.Tensor:
    mask = torch.full((full_length, full_length), -torch.inf, dtype=torch.float32, device=device)
    for start, end in blocks:
        block_len = end - start
        if block_len <= 0:
            continue
        mask[start:end, start:end] = causal_additive_mask(block_len, device=device)
    return mask
