from __future__ import annotations

import typer

from llm_jepa.inference.generate import generate_from_cli


def main(checkpoint: str, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> None:
    typer.echo(generate_from_cli(checkpoint, prompt, max_new_tokens, temperature))


if __name__ == "__main__":
    typer.run(main)
