from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig

from llm_jepa.modeling.tokenizer_setup import format_messages, load_model_and_tokenizer


def _latest_numbered_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.is_dir():
        return None
    checkpoints = []
    for path in checkpoints_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            checkpoints.append(path)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1]))


def resolve_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint).expanduser()
    if path.exists():
        return str(path)

    if path.name == "checkpoint" and path.parent.exists():
        final_checkpoint = path.parent / "checkpoint-final"
        if final_checkpoint.is_dir():
            return str(final_checkpoint)
        latest_checkpoint = _latest_numbered_checkpoint(path.parent / "checkpoints")
        if latest_checkpoint is not None:
            return str(latest_checkpoint)

    if checkpoint.count("/") > 1 or checkpoint.startswith((".", "~", "/")):
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint}. "
            "Use checkpoint-final or a directory under checkpoints/, such as checkpoint-500."
        )

    return checkpoint


def _input_device(model) -> torch.device:
    base_model = getattr(model, "base_model", model)
    embeddings = base_model.get_input_embeddings()
    return embeddings.weight.device


def generate_text(
    checkpoint: str,
    messages: list[dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device_map: str | dict[str, Any] | None = "auto",
) -> str:
    checkpoint = resolve_checkpoint(checkpoint)
    model, tokenizer = load_model_and_tokenizer(checkpoint, device_map=device_map)
    prompt = format_messages(messages, tokenizer, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    if device_map is None:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    encoded = {key: value.to(_input_device(model)) for key, value in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    generation_config = GenerationConfig(**generation_kwargs)
    if temperature <= 0:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    output = model.generate(**encoded, generation_config=generation_config)
    generated = output[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_from_cli(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    return generate_text(
        checkpoint,
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
