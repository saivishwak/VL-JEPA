from __future__ import annotations

import typer

from llm_jepa.evaluation.generation_eval import evaluate_from_config


def main(config: str = typer.Option(..., "--config", "-c")) -> None:
    metrics = evaluate_from_config(config)
    typer.echo(metrics)


if __name__ == "__main__":
    typer.run(main)
