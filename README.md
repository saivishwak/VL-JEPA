# VL-JEPA

VL-JEPA is a PyTorch/Hugging Face implementation scaffold for the paper
“VL-JEPA: Joint Embedding Predictive Architecture for Vision-language”.

The project is vision-language only. It predicts continuous target-text embeddings from
visual inputs and optional textual queries, then uses those embeddings for classification,
retrieval, VQA, selective decoding, or optional text readout.

## Features

- Paper-shaped `X_V, X_Q, Y` JSONL manifests for image/video captioning, classification,
  retrieval, VQA, text triplets, selective decoding, and world-prediction style tasks.
- Frozen visual X-Encoder wrapper, Llama-initialized predictor, EmbeddingGemma-style
  Y-Encoder, and optional inference-only Y-Decoder.
- Bidirectional InfoNCE objective in the shared target embedding space.
- VL-JEPA training loop with Y-Encoder learning-rate multiplier and checkpoint save/load.
- Candidate ranking for open-vocabulary classification, retrieval, and discriminative VQA.
- Selective-decoding utilities for embedding streams.
- Tiny model mode for local tests without downloading paper-scale checkpoints.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install a CUDA-specific PyTorch build first if your machine requires one.

For gated model checkpoints such as Llama, log in before training:

```bash
hf auth login
hf auth whoami
```

Configs set `hf_token: true`, which tells Hugging Face loaders to use your logged-in token.
For non-interactive jobs, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

## Quick Smoke Run

```bash
vl-jepa train --config configs/vl_jepa/smoke_tiny.yaml
```

## Prepare A Manifest

```bash
vl-jepa prepare \
  --dataset datacomp \
  --split train \
  --source data/vl_jepa/datacomp_train_source.jsonl \
  --output-dir data/vl_jepa
```

## Download Open Datasets

For public Hugging Face datasets, use the downloader to materialize images and write
VL-JEPA JSONL plus a manifest:

```bash
vl-jepa download-open --dataset flickr30k --split test --max-examples 100
```

The same downloader is also available as a script:

```bash
python scripts/download_open_datasets.py flickr30k --split test --max-examples 100
```

Available presets:

- `flickr30k`: image captioning.
- `coco_captions`: image captioning.
- `vqav2`: visual question answering.
- `ok_vqa`: visual question answering.

For `vqav2` and `ok_vqa`, use `--split validation`. The downloader also accepts
legacy aliases such as `val` and `val2014` and maps them to `validation`.

Restricted or unreleased paper datasets such as Action100M and the PLM mixture still require
separate access and should be registered with `vl-jepa prepare`.

## Train

```bash
vl-jepa train --config configs/vl_jepa/pretrain_image_1f.yaml
```

Train the optional Llama Y-Decoder readout after you have a VL-JEPA checkpoint:

```bash
vl-jepa train-decoder --config configs/vl_jepa/train_decoder_caption.yaml
```

The decoder stage freezes VL-JEPA and, by default, freezes the Llama weights too. It trains
the embedding-to-Llama prefix projection on the same captioning JSONL used for image
captioning. Set `decoder.train_lm: true` in the decoder config only if you want to finetune
the Llama decoder weights as well.

Long training runs can save periodic checkpoints with:

```yaml
training:
  checkpoint_steps: 500
```

This writes `checkpoint-step-500/`, `checkpoint-step-1000/`, etc. under the run directory,
plus `checkpoint-final/` at the end.

## Visualize Loss

Training writes TensorBoard events under each run directory by default:

```text
runs/<timestamp>-<run-name>/tensorboard/
```

Launch TensorBoard with:

```bash
tensorboard --logdir runs
```

Logged scalars:

- `train/loss`
- `train/lr`
- `train/y_encoder_lr`
- `train/epoch`

## Inference

```bash
vl-jepa infer \
  --checkpoint runs/example-decoder/checkpoint-final \
  --visual-path examples/image.jpg \
  --query "Caption the image."
```

`infer` returns text only when the checkpoint was configured with a `decoder_model`.
Pretraining checkpoints with `decoder_model: null` predict target embeddings. For those
checkpoints, print the embedding explicitly:

```bash
vl-jepa infer \
  --checkpoint runs/example/checkpoint-final \
  --visual-path examples/image.jpg \
  --query "Caption the image." \
  --output embedding
```

or rank candidate captions/answers:

```bash
vl-jepa infer \
  --checkpoint runs/example/checkpoint-final \
  --visual-path examples/image.jpg \
  --query "Caption the image." \
  --candidate "a dog on grass" \
  --candidate "a car on a road"
```

## Evaluation

```bash
vl-jepa eval --config configs/vl_jepa/eval_classification.yaml
```

Outputs are written under `runs/{timestamp}-{name}/` with the resolved config, checkpoints,
predictions, metrics, and manifests.

## Project Layout

```text
vl_jepa/
├── data/          # VL-JEPA schemas, manifests, visual transforms, and loaders
├── modeling/      # X-Encoder, Predictor, Y-Encoder, optional Y-Decoder
├── training/      # bidirectional InfoNCE and VL-JEPA training loop
├── inference/     # embedding prediction, candidate ranking, optional decoding
├── evaluation/    # classification, retrieval, VQA, text triplet, selective decoding
└── utils/         # config, IO, logging, reproducibility
```
