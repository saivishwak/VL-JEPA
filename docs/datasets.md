# Datasets

All training and evaluation data uses a chat JSONL schema:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Optional fields are `id`, `task`, and `metadata`.

## Supported Tasks

- `synthetic`: local arithmetic generation for smoke tests.
- `gsm8k`: Hugging Face `gsm8k/main`.
- `spider`: Hugging Face `spider`, with optional local SQLite database root for eval.
- `hellaswag`: Hugging Face `hellaswag`.
- `paraphrase`: Hugging Face `cestwc/paraphrase`.
- `rotten_tomatoes`: Hugging Face `cornell-movie-review-data/rotten_tomatoes`.
- `yelp`: Hugging Face `Yelp/yelp_review_full`.
- `synth` and `turk`: local reference JSONL directories matching `synth_train.jsonl` and
  `synth_test.jsonl` naming.

Every preparation run writes `manifest.json` with source, split sizes, seed, and output paths.
