import torch

from vl_jepa.modeling import VLJEPA, VLJEPAConfig
from vl_jepa.modeling.predictor import Predictor
from vl_jepa.modeling.x_encoder import XEncoder
from vl_jepa.training.train import _resolve_init_checkpoint, vl_jepa_optimizer


def test_tiny_vl_jepa_forward_loss_and_freeze():
    model = VLJEPA(
        VLJEPAConfig(
            vision_encoder="tiny",
            predictor_model="tiny",
            target_encoder="tiny",
            embedding_dim=64,
            image_size=32,
            num_frames=1,
            tiny=True,
        )
    )
    assert not any(parameter.requires_grad for parameter in model.x_encoder.encoder.parameters())
    batch = {
        "pixel_values": torch.randn(2, 1, 3, 32, 32),
        "query_input_ids": torch.tensor([[3, 1], [4, 1]]),
        "query_attention_mask": torch.ones(2, 2, dtype=torch.long),
        "target_input_ids": torch.tensor([[5, 1], [6, 1]]),
        "target_attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    outputs = model.forward_loss(batch)
    assert torch.isfinite(outputs["loss"])
    assert outputs["predicted_embedding"].shape == (2, 64)


def test_tiny_vl_jepa_forward_loss_without_query_tokens():
    model = VLJEPA(
        VLJEPAConfig(
            vision_encoder="tiny",
            predictor_model="tiny",
            target_encoder="tiny",
            embedding_dim=64,
            image_size=32,
            num_frames=1,
            tiny=True,
        )
    )
    batch = {
        "pixel_values": torch.randn(2, 1, 3, 32, 32),
        "query_input_ids": torch.empty((2, 0), dtype=torch.long),
        "query_attention_mask": torch.empty((2, 0), dtype=torch.long),
        "target_input_ids": torch.tensor([[5, 1], [6, 1]]),
        "target_attention_mask": torch.ones(2, 2, dtype=torch.long),
    }

    outputs = model.forward_loss(batch)

    assert torch.isfinite(outputs["loss"])


def test_vl_jepa_optimizer_uses_y_encoder_multiplier():
    config = VLJEPAConfig(
        vision_encoder="tiny",
        predictor_model="tiny",
        target_encoder="tiny",
        embedding_dim=64,
        tiny=True,
        y_encoder_lr_multiplier=0.05,
    )
    model = VLJEPA(config)
    optimizer = vl_jepa_optimizer(model, config, learning_rate=1e-3, weight_decay=0.0)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[1]["lr"] == 5e-5


def test_x_encoder_uses_vjepa2_video_argument():
    class FakeVideoEncoder(torch.nn.Module):
        config = type("Config", (), {"hidden_size": 5})()

        def forward(self, pixel_values_videos, output_hidden_states=False):
            assert output_hidden_states
            batch = pixel_values_videos.shape[0]
            hidden = torch.ones(batch, 3, 5)
            return type("Output", (), {"last_hidden_state": hidden})()

    encoder = XEncoder("tiny", output_dim=4, tiny=True)
    encoder.encoder = FakeVideoEncoder()
    encoder.projection = torch.nn.Linear(5, 4)

    output = encoder(torch.randn(2, 1, 3, 32, 32))

    assert output.shape == (2, 3, 4)


def test_x_encoder_casts_hidden_to_projection_dtype():
    class FakeBfloatEncoder(torch.nn.Module):
        config = type("Config", (), {"hidden_size": 5})()

        def forward(self, pixel_values_videos, output_hidden_states=False):
            batch = pixel_values_videos.shape[0]
            hidden = torch.ones(batch, 3, 5, dtype=torch.bfloat16)
            return type("Output", (), {"last_hidden_state": hidden})()

    encoder = XEncoder("tiny", output_dim=4, tiny=True)
    encoder.encoder = FakeBfloatEncoder()
    encoder.projection = torch.nn.Linear(5, 4)

    output = encoder(torch.randn(2, 1, 3, 32, 32))

    assert output.dtype == torch.float32


def test_predictor_bidirectional_attention_mask_has_no_causal_triangle():
    predictor = Predictor("tiny", vision_dim=64, output_dim=64, tiny=True)
    mask = predictor._bidirectional_attention_mask(
        torch.tensor([[1, 1, 0]], dtype=torch.long),
        dtype=torch.float32,
    )

    assert torch.all(mask[0, 0, :, :2] == 0)
    assert torch.all(mask[0, 0, :, 2] < -1e20)


def test_predictor_disables_causal_attention_flags():
    predictor = Predictor("tiny", vision_dim=64, output_dim=64, tiny=True)
    attention = torch.nn.Module()
    attention.is_causal = True
    predictor.backbone.self_attn = attention

    predictor._disable_causal_attention()

    assert predictor.backbone.self_attn.is_causal is False


def test_resolve_init_checkpoint_uses_latest_named_run(tmp_path):
    older = tmp_path / "20260101-000000-vl-jepa-pretrain-video-32f" / "checkpoint-final"
    newer = tmp_path / "20260102-000000-vl-jepa-pretrain-video-32f" / "checkpoint-final"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "model.pt").write_bytes(b"old")
    (newer / "model.pt").write_bytes(b"new")

    resolved = _resolve_init_checkpoint(
        {"init_checkpoint_run_name": "vl-jepa-pretrain-video-32f"},
        tmp_path,
    )

    assert resolved == newer / "model.pt"
