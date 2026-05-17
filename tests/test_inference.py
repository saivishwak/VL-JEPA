import pytest
import torch

from vl_jepa.inference import infer as infer_module
from vl_jepa.inference.infer import infer_vl_jepa_from_cli


class DummyDecoder:
    enabled = False


class DummyModel:
    y_decoder = DummyDecoder()


def test_infer_auto_requires_decoder_or_candidates(monkeypatch):
    monkeypatch.setattr(infer_module, "load_vl_jepa_checkpoint", lambda checkpoint: DummyModel())
    monkeypatch.setattr(
        infer_module,
        "predict_vl_jepa_embedding",
        lambda model, visual_path, query: torch.zeros(1, 4),
    )

    with pytest.raises(RuntimeError, match="decoder_model is null"):
        infer_vl_jepa_from_cli("checkpoint", "image.jpg", "Caption the image.")


def test_infer_embedding_output_is_explicit(monkeypatch):
    monkeypatch.setattr(infer_module, "load_vl_jepa_checkpoint", lambda checkpoint: DummyModel())
    monkeypatch.setattr(
        infer_module,
        "predict_vl_jepa_embedding",
        lambda model, visual_path, query: torch.tensor([[1.0, -2.0]]),
    )

    output = infer_vl_jepa_from_cli(
        "checkpoint",
        "image.jpg",
        "Caption the image.",
        output="embedding",
    )

    assert output == "1.000000 -2.000000"
