# Evaluation

The evaluator generates model responses and scores them with task-aware metrics.

```bash
llm-jepa eval --config configs/eval/gsm8k.yaml
```

Outputs:

- `predictions.jsonl`: one record per example with prediction, target, and correctness.
- `metrics.json`: machine-readable aggregate metrics.
- `report.md`: short human-readable summary.

## Metrics

- `gsm8k`: extracts the final `#### answer` when present.
- `spider`: executes predicted and target SQL against the SQLite database.
- `hellaswag`, `rotten_tomatoes`, `yelp`: prefix matching for compact labels.
- other tasks: normalized exact match.

Spider evaluation requires `spider_path` to point to the database directory.
