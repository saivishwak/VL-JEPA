from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def apply_lora(
    model: PreTrainedModel,
    rank: int = 16,
    alpha: int | None = None,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> PreTrainedModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha or rank * 2,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_LORA_TARGETS,
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    return model
