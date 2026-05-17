from __future__ import annotations

from pathlib import Path

import typer

from vl_jepa.data.downloaders import download_open_dataset, preset_names

app = typer.Typer(help="Download public vision-language datasets into VL-JEPA JSONL format.")


@app.command()
def main(
    dataset: str = typer.Argument(..., help=f"Dataset preset: {', '.join(preset_names())}"),
    split: str = typer.Option("train", help="Hugging Face split name."),
    output_dir: Path = typer.Option(Path("data/vl_jepa"), help="Output directory."),
    max_examples: int | None = typer.Option(None, help="Optional cap for testing."),
    streaming: bool = typer.Option(False, help="Use Hugging Face streaming mode."),
) -> None:
    manifest = download_open_dataset(
        dataset=dataset,
        split=split,
        output_dir=output_dir,
        max_examples=max_examples,
        streaming=streaming,
    )
    typer.echo(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    app()
