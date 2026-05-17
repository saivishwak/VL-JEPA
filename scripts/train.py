from __future__ import annotations

import typer

from llm_jepa.training.train import train_from_config


def main(config: str = typer.Option(..., "--config", "-c")) -> None:
    train_from_config(config)


if __name__ == "__main__":
    typer.run(main)
