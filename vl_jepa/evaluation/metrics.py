from __future__ import annotations


def topk_accuracy(ranked_labels: list[list[str]], targets: list[str], k: int) -> float:
    if not targets:
        return 0.0
    correct = 0
    for labels, target in zip(ranked_labels, targets, strict=True):
        correct += target in labels[:k]
    return correct / len(targets)


def recall_at_k(ranked_ids: list[list[str]], positives: list[str], k: int) -> float:
    return topk_accuracy(ranked_ids, positives, k)
