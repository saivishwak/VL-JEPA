# Training

Use `llm-jepa train --config <config>` for local runs and `torchrun scripts/train.py` for
multi-GPU runs.

## Key Config Fields

- `jepa.regular`: use standard supervised fine-tuning when true.
- `jepa.lambda`: weight for the JEPA representation loss.
- `jepa.gamma`: weight for the language-modeling loss.
- `jepa.predictors`: number of predictor tokens appended to the input view.
- `jepa.last_token`: hidden-state position offset used for view embeddings.
- `jepa.loss`: `cosine`, `mse`, `l2`, or `infonce`.
- `jepa.additive_mask`: compute full and JEPA views in one model call using a 4D mask.
- `jepa.jepa_ratio`: apply JEPA loss to a random fraction of batches.
- `lora.enabled`: enable parameter-efficient fine-tuning.

## Multi-GPU

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/train/llm_jepa.yaml
```

Keep batch size per device small for 1B+ models and scale with `training.grad_accum`.
