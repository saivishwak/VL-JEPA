from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from llm_jepa.data.builders import prepare_dataset
from llm_jepa.evaluation.generation_eval import evaluate_from_config
from llm_jepa.inference.generate import generate_from_cli
from llm_jepa.training.train import train_from_config

app = typer.Typer(help="LLM-JEPA training, inference, and evaluation CLI.")


@app.command("prepare-data")
def prepare_data(
    task: str = typer.Option(..., help="Task name or 'synthetic'."),
    output_dir: Path = typer.Option(Path("data/processed"), help="Directory for JSONL outputs."),
    train_size: int = typer.Option(1000, help="Maximum train examples or synthetic train size."),
    test_size: int = typer.Option(200, help="Maximum test examples or synthetic test size."),
    seed: int = typer.Option(42, help="Deterministic split/generation seed."),
    source: Optional[str] = typer.Option(None, help="Optional local source path or HF dataset name."),
) -> None:
    manifest = prepare_dataset(
        task=task,
        output_dir=output_dir,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        source=source,
    )
    typer.echo(f"Wrote dataset manifest: {manifest}")


@app.command("train")
def train(config: Path = typer.Option(..., "--config", "-c", help="Training config YAML.")) -> None:
    train_from_config(config)


@app.command("infer")
def infer(
    checkpoint: str = typer.Option(..., help="Model checkpoint or Hugging Face model id."),
    prompt: str = typer.Option(..., help="Prompt to generate from."),
    max_new_tokens: int = typer.Option(128, help="Maximum generated tokens."),
    temperature: float = typer.Option(0.0, help="Sampling temperature. Use 0 for greedy."),
) -> None:
    typer.echo(generate_from_cli(checkpoint, prompt, max_new_tokens, temperature))


@app.command("eval")
def eval_command(config: Path = typer.Option(..., "--config", "-c", help="Eval config YAML.")) -> None:
    metrics_path = evaluate_from_config(config)
    typer.echo(f"Wrote evaluation metrics: {metrics_path}")


@app.command("smoke-test")
def smoke_test() -> None:
    from llm_jepa.smoke import run_smoke_test

    run_smoke_test()
    typer.echo("Smoke test completed.")
