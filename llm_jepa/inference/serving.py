from __future__ import annotations

from dataclasses import dataclass

from llm_jepa.inference.generate import generate_text


@dataclass
class GenerationService:
    checkpoint: str
    max_new_tokens: int = 128
    temperature: float = 0.0

    def generate(self, prompt: str) -> str:
        return generate_text(
            self.checkpoint,
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
