# Faithfulness And Limitations

This implementation follows the reference LLM-JEPA behavior at the objective level while
turning the research scripts into a tested package.

## Preserved

- Causal language-model training on full prompt/target sequences.
- Separate input and target views for representation alignment.
- Last-token hidden-state extraction with model-family-specific offsets.
- Cosine JEPA loss by default, with MSE, L2, and InfoNCE ablations.
- Predictor tokens, reverse prediction, front predictor placement, and plain formatting.
- Additive-mask JEPA mode for reducing extra forward passes.
- Random JEPA-loss dropout via `jepa_ratio`.
- LoRA and full fine-tuning.
- Task-aware evaluation for generation datasets.

## Deliberate Differences

- Dataset preparation is explicit and manifest-driven instead of assuming files are already in
  the working directory.
- CLI and YAML configs replace experiment shell functions.
- Tests cover reusable invariants; large-model quality still depends on external compute,
  model access, and dataset availability.
- V2 multimodal support is documented through extension points but not implemented in the
  first text-generation release.
