# LLM-JEPA Tutorial

LLM-JEPA fine-tunes a causal language model with two objectives:

1. The standard language-modeling loss on the full prompt and target.
2. A representation loss between two views of the same example: the input view and the
   target view.

For text generation tasks, the input view is usually the user/problem text. The target view is
the assistant answer, code, SQL query, class label, or continuation. The default JEPA loss is
`1 - cosine_similarity(input_embedding, target_embedding)`.

## Prepare Data

```bash
llm-jepa prepare-data --task gsm8k --output-dir data/gsm8k
```

For quick local checks:

```bash
llm-jepa prepare-data --task synthetic --output-dir data/synthetic --train-size 128 --test-size 32
```

## Train

```bash
llm-jepa train --config configs/train/llm_jepa.yaml
```

The config controls regular versus JEPA training, predictor token count, LoRA, additive masks,
and the representation loss.

## Evaluate

```bash
llm-jepa eval --config configs/eval/gsm8k.yaml
```

The evaluator writes `predictions.jsonl`, `metrics.json`, and `report.md` into a run directory.

## Faithfulness Notes

This project preserves the load-bearing LLM-JEPA behavior from the reference implementation:

- separate input and output views;
- hidden-state extraction at configurable terminal token positions;
- joint language-model and representation objectives;
- predictor tokens;
- LoRA/full fine-tuning;
- additive-mask single-forward training;
- random JEPA-loss dropout;
- task-specific generation evaluation.

The code is intentionally factored into modules instead of a single research script so that data,
training, inference, and evaluation can be tested and extended independently.
