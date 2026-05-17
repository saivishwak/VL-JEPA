from __future__ import annotations

import re

GSM8K_ANSWER = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def exact_match(prediction: str, target: str) -> bool:
    return normalize_text(prediction) == normalize_text(target)


def startswith_match(prediction: str, target: str) -> bool:
    return normalize_text(prediction).startswith(normalize_text(target))


def gsm8k_match(prediction: str, target: str) -> bool:
    pred = GSM8K_ANSWER.search(prediction)
    gold = GSM8K_ANSWER.search(target)
    if pred and gold:
        return pred.group(1).replace(",", "") == gold.group(1).replace(",", "")
    return exact_match(prediction, target)


def accuracy(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0
