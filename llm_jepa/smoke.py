from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from llm_jepa.data.builders import prepare_dataset
from llm_jepa.training.losses import jepa_loss
from llm_jepa.training.masks import block_diagonal_causal_mask, causal_additive_mask


def run_smoke_test() -> None:
    with TemporaryDirectory() as tmp:
        manifest = prepare_dataset("synthetic", Path(tmp) / "data", 8, 2, seed=7)
        assert manifest.exists()
        assert causal_additive_mask(4).shape == (4, 4)
        assert torch.isinf(block_diagonal_causal_mask(6, [(0, 2), (3, 6)])[0, 3])
        left = torch.randn(4, 8)
        right = left + 0.01 * torch.randn(4, 8)
        loss = jepa_loss(left, right, mode="cosine")
        assert torch.isfinite(loss)
