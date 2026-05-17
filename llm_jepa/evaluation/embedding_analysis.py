from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_jepa.modeling.tokenizer_setup import format_messages


@torch.no_grad()
def sequence_embedding(model, tokenizer, messages: list[dict[str, str]], layer: int = -1) -> torch.Tensor:
    prompt = format_messages(messages, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**encoded, output_hidden_states=True)
    mask = encoded["attention_mask"].bool()
    hidden = outputs.hidden_states[layer][0]
    return hidden[mask[0]].mean(dim=0)


def cosine_stats(embeddings_a: list[torch.Tensor], embeddings_b: list[torch.Tensor]) -> dict[str, float]:
    if not embeddings_a:
        return {"mean": 0.0, "std": 0.0}
    sims = torch.stack(
        [F.cosine_similarity(a.float(), b.float(), dim=0) for a, b in zip(embeddings_a, embeddings_b)]
    )
    return {"mean": float(sims.mean().item()), "std": float(sims.std(unbiased=False).item())}
