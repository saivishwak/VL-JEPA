from __future__ import annotations

from pathlib import Path

import typer

from vl_jepa.data.builders import prepare_vl_jepa_manifest
from vl_jepa.data.downloaders import download_open_dataset, preset_names
from vl_jepa.evaluation.eval import evaluate_vl_jepa_from_config
from vl_jepa.inference.infer import export_embeddings_from_config, infer_vl_jepa_from_cli
from vl_jepa.training.decoder import train_y_decoder_from_config
from vl_jepa.training.train import train_vl_jepa_from_config

app = typer.Typer(help="VL-JEPA training, inference, and evaluation CLI.")


@app.command("prepare")
def prepare(
    dataset: str = typer.Option(..., help="Paper dataset key, e.g. datacomp, action100m, gqa."),
    split: str = typer.Option("train", help="Dataset split name."),
    output_dir: Path = typer.Option(Path("data/vl_jepa"), help="Directory for VL-JEPA manifests."),
    source: str | None = typer.Option(None, help="Source JSONL already matching VLJEPASample."),
    task_type: str | None = typer.Option(None, help="Override task type if not inferred."),
) -> None:
    manifest = prepare_vl_jepa_manifest(
        dataset=dataset,
        split=split,
        task_type=task_type,
        source=source,
        output_dir=output_dir,
    )
    typer.echo(f"Wrote VL-JEPA manifest: {manifest}")


@app.command("download-open")
def download_open(
    dataset: str = typer.Option(
        ..., help=f"Open dataset preset. Choices: {', '.join(preset_names())}"
    ),
    split: str = typer.Option("train", help="Hugging Face split to download."),
    output_dir: Path = typer.Option(Path("data/vl_jepa"), help="Output directory."),
    max_examples: int | None = typer.Option(None, help="Optional cap for local smoke runs."),
    streaming: bool = typer.Option(False, help="Use Hugging Face streaming mode."),
) -> None:
    manifest = download_open_dataset(
        dataset=dataset,
        split=split,
        output_dir=output_dir,
        max_examples=max_examples,
        streaming=streaming,
    )
    typer.echo(f"Wrote VL-JEPA open-dataset manifest: {manifest}")


@app.command("train")
def train(config: Path = typer.Option(..., "--config", "-c", help="Training config YAML.")) -> None:
    train_vl_jepa_from_config(config)


@app.command("train-decoder")
def train_decoder(
    config: Path = typer.Option(..., "--config", "-c", help="Y-Decoder training config YAML."),
) -> None:
    train_y_decoder_from_config(config)


@app.command("infer")
def infer(
    checkpoint: str = typer.Option(..., help="VL-JEPA checkpoint-final directory."),
    visual_path: str = typer.Option(..., help="Image, video, or frame directory path."),
    query: str = typer.Option("", help="Textual query X_Q."),
    candidate: list[str] = typer.Option(None, "--candidate", help="Candidate answer/label text."),
    output: str = typer.Option(
        "auto",
        "--output",
        help="Output mode: auto, embedding, decode, or rank.",
    ),
) -> None:
    typer.echo(infer_vl_jepa_from_cli(checkpoint, visual_path, query, candidate, output))


@app.command("eval")
def eval_command(
    config: Path = typer.Option(..., "--config", "-c", help="VL-JEPA eval config YAML."),
) -> None:
    metrics_path = evaluate_vl_jepa_from_config(config)
    typer.echo(f"Wrote VL-JEPA evaluation metrics: {metrics_path}")


@app.command("selective-decode")
def selective_decode(
    config: Path = typer.Option(..., "--config", "-c", help="Selective decoding config YAML."),
) -> None:
    metrics_path = evaluate_vl_jepa_from_config(config)
    typer.echo(f"Wrote selective decoding metrics: {metrics_path}")


@app.command("export-embeddings")
def export_embeddings(
    config: Path = typer.Option(..., "--config", "-c", help="Embedding export config YAML."),
) -> None:
    output_path = export_embeddings_from_config(config)
    typer.echo(f"Wrote VL-JEPA embeddings: {output_path}")
