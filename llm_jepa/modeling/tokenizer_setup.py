from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_jepa.modeling.special_tokens import LLM_JEPA_SPECIAL_TOKENS


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def add_llm_jepa_tokens(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> int:
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [token for token in LLM_JEPA_SPECIAL_TOKENS if token not in existing_vocab]
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if added:
        model.resize_token_embeddings(len(tokenizer))
    return added


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str | None = "auto",
    trust_remote_code: bool = True,
    device_map: str | dict[str, Any] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype:
        kwargs["torch_dtype"] = torch_dtype
    if device_map:
        kwargs["device_map"] = device_map
    adapter_config_path = Path(model_name) / "adapter_config.json"
    if adapter_config_path.is_file():
        with adapter_config_path.open("r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_model_name = adapter_config["base_model_name_or_path"]
        kwargs["local_files_only"] = True
        model_config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=model_config,
            **kwargs,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, model_name)
        return model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    add_llm_jepa_tokens(model, tokenizer)
    return model, tokenizer


def format_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = False,
    plain: bool = False,
) -> str:
    if plain:
        return "\n".join(message["content"] for message in messages)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n".join(rendered)
