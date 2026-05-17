# LLM-JEPA

LLM-JEPA is a structured PyTorch/Hugging Face project for text-generation JEPA training.
It reproduces the core idea from `galilai-group/llm-jepa`: combine causal language-model
fine-tuning with a representation objective that aligns the hidden states of an input
view, such as a problem statement, with an output view, such as an answer, program, or label.

The first version focuses on LLM generation. The code is organized so a later multimodal
image/V-JEPA path can reuse dataset, training, and evaluation interfaces.

## Features

- Public dataset preparation for `synth`, `turk`, `gsm8k`, `spider`, `hellaswag`,
  `paraphrase`, `rotten_tomatoes`, and `yelp`.
- Synthetic JSONL generation for quick local smoke runs.
- Regular supervised fine-tuning and LLM-JEPA fine-tuning.
- Cosine, MSE, L2, and InfoNCE JEPA losses.
- Predictor tokens, LoRA, additive-mask single-forward JEPA mode, random JEPA-loss dropout,
  same-FLOP scheduling, mixed precision, gradient accumulation, and checkpoint resume.
- Batch and single-prompt inference.
- Generation metrics, Spider SQLite execution evaluation, embedding cosine analysis, and
  JSON/Markdown reports.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install a CUDA-specific PyTorch build first if your machine requires one.

## Quick Smoke Run

```bash
llm-jepa prepare-data --task synthetic --output-dir data/synthetic --train-size 32 --test-size 8
llm-jepa smoke-test
```

### Prepare dataset

```bash
llm-jepa prepare-data --task gsm8k --output-dir data/gsm8k
```

## Train

```bash
llm-jepa train --config configs/train/llm_jepa.yaml
```

For multi-GPU:

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/train/llm_jepa.yaml
```

## Inference

```bash
llm-jepa infer --checkpoint runs/example/checkpoint-final --prompt "What is 17 + 25?"
```

## Evaluation

```bash
llm-jepa eval --config configs/eval/gsm8k.yaml
```

### Latent-space plots

Generate a 2D PNG projection of token or contextual embeddings to inspect relationships such as
`king`, `man`, `queen`, and `woman`:

```bash
.venv/bin/python scripts/plot_latent_space.py \
  --checkpoint runs/example/checkpoint-final \
  --output runs/latent_space.png \
  --term king --term man --term queen --term woman
```

By default this plots the model's input token embedding space. To plot the JEPA latent space,
use `--mode jepa`; this plots paired input and target view hidden states at the same
representation positions used by the JEPA loss:

```bash
.venv/bin/python scripts/plot_latent_space.py \
  --checkpoint runs/example/checkpoint-final \
  --mode jepa \
  --data-file data/gsm8k/gsm8k_train.jsonl \
  --predictors 2 \
  --last-token -2 \
  --output runs/jepa_latent_space.png
```

Add `--mode contextual` to plot prompt-level hidden-state embeddings, or `--projection tsne`
for t-SNE instead of PCA.

Outputs are written under `runs/{timestamp}-{name}/` with the resolved config, checkpoints,
predictions, metrics, and manifests.

## Project Layout

```text
llm_jepa/
├── data/          # schemas, dataset builders, public task adapters
├── modeling/      # tokenizer setup, special tokens, LoRA helpers, Trainer subclass
├── training/      # collators, losses, additive masks, training entrypoint
├── inference/     # generation utilities
├── evaluation/    # generation, Spider, and embedding metrics
└── utils/         # config, IO, logging, reproducibility
```

See `docs/` for dataset, training, evaluation, and multimodal roadmap notes.
