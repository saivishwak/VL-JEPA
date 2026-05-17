from __future__ import annotations

PREDICTOR_TOKENS = [f"<|predictor_{idx}|>" for idx in range(1, 11)]
PERCEPTION_TOKEN = "<|perception|>"
START_OF_TEXT_TOKEN = "<|startoftext|>"
CHAT_COMPAT_TOKENS = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

LLM_JEPA_SPECIAL_TOKENS = [
    *PREDICTOR_TOKENS,
    START_OF_TEXT_TOKEN,
    PERCEPTION_TOKEN,
    *CHAT_COMPAT_TOKENS,
]


def predictor_suffix(count: int) -> str:
    if count < 0 or count > len(PREDICTOR_TOKENS):
        raise ValueError(f"predictor count must be between 0 and {len(PREDICTOR_TOKENS)}")
    return "".join(PREDICTOR_TOKENS[:count])
