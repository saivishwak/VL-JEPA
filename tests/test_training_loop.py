import json

from PIL import Image

from vl_jepa.training.decoder import train_y_decoder_from_config
from vl_jepa.training.train import train_vl_jepa_from_config


def test_train_loop_runs_query_free_tiny_config(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_a)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_b)

    data_path = tmp_path / "train.jsonl"
    records = [
        {
            "id": "a",
            "task_type": "captioning",
            "visual_kind": "image",
            "visual_path": str(image_a),
            "query": "",
            "target": "red square",
        },
        {
            "id": "b",
            "task_type": "captioning",
            "visual_kind": "image",
            "visual_path": str(image_b),
            "query": "",
            "target": "green square",
        },
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: tiny-train
run_root: {tmp_path / "runs"}
model:
  vision_encoder: tiny
  predictor_model: tiny
  target_encoder: tiny
  decoder_model: null
  embedding_dim: 64
  predictor_layers: 2
  image_size: 32
  num_frames: 1
  tiny: true
data:
  train_file: {data_path}
  eval_file: {data_path}
  query_override: ""
training:
  seed: 42
  max_steps: 1
  batch_size: 2
  contrastive_accum_batches: 1
  grad_accum: 1
  learning_rate: 1.0e-4
  weight_decay: 0.0
  tensorboard: false
  checkpoint_steps: 1
  eval_steps: 1
  max_eval_batches: 1
""",
        encoding="utf-8",
    )

    run_dir = train_vl_jepa_from_config(config_path)

    assert (run_dir / "checkpoint-step-1" / "model.pt").is_file()
    assert (run_dir / "checkpoint-step-1" / "training_state.pt").is_file()
    assert (run_dir / "checkpoint-final" / "model.pt").is_file()


def test_decoder_train_loop_adds_tiny_decoder(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_a)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_b)

    data_path = tmp_path / "train.jsonl"
    records = [
        {
            "id": "a",
            "task_type": "captioning",
            "visual_kind": "image",
            "visual_path": str(image_a),
            "query": "",
            "target": "red square",
        },
        {
            "id": "b",
            "task_type": "captioning",
            "visual_kind": "image",
            "visual_path": str(image_b),
            "query": "",
            "target": "green square",
        },
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        f"""
name: tiny-base
run_root: {tmp_path / "runs"}
model:
  vision_encoder: tiny
  predictor_model: tiny
  target_encoder: tiny
  decoder_model: null
  embedding_dim: 64
  predictor_layers: 2
  image_size: 32
  num_frames: 1
  tiny: true
data:
  train_file: {data_path}
  eval_file: {data_path}
  query_override: ""
training:
  seed: 42
  max_steps: 1
  batch_size: 2
  contrastive_accum_batches: 1
  grad_accum: 1
  learning_rate: 1.0e-4
  weight_decay: 0.0
  tensorboard: false
  checkpoint_steps: 1
""",
        encoding="utf-8",
    )
    base_run = train_vl_jepa_from_config(base_config_path)

    decoder_config_path = tmp_path / "decoder.yaml"
    decoder_config_path.write_text(
        f"""
name: tiny-decoder
run_root: {tmp_path / "runs"}
checkpoint: {base_run / "checkpoint-final"}
decoder:
  model_name: tiny
  train_lm: false
  embedding_source: predicted
  max_length: 16
data:
  train_file: {data_path}
  eval_file: {data_path}
  query_override: "Caption the image."
training:
  seed: 42
  max_steps: 1
  batch_size: 2
  grad_accum: 1
  learning_rate: 1.0e-4
  weight_decay: 0.0
  tensorboard: false
  checkpoint_steps: 1
  eval_steps: 1
  max_eval_batches: 1
""",
        encoding="utf-8",
    )

    decoder_run = train_y_decoder_from_config(decoder_config_path)

    assert (decoder_run / "checkpoint-step-1" / "model.pt").is_file()
    assert (decoder_run / "checkpoint-step-1" / "training_state.pt").is_file()
    assert (decoder_run / "checkpoint-final" / "model.pt").is_file()
