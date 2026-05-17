from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from llm_jepa.inference.generate import resolve_checkpoint
from llm_jepa.modeling.tokenizer_setup import format_messages, load_model_and_tokenizer
from llm_jepa.training.dataset import ChatJEPADataset


DEFAULT_TERMS = [
    "king",
    "queen",
    "man",
    "woman",
    "prince",
    "princess",
    "boy",
    "girl",
    "father",
    "mother",
    "royal",
    "person",
]


def _model_device(model) -> torch.device:
    base_model = getattr(model, "base_model", model)
    return base_model.get_input_embeddings().weight.device


def _mean_token_embedding(model, tokenizer, text: str) -> torch.Tensor:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"No tokens produced for term: {text!r}")
    embeddings = model.get_input_embeddings().weight.detach()
    ids = torch.tensor(token_ids, device=embeddings.device)
    return embeddings.index_select(0, ids).mean(dim=0)


@torch.no_grad()
def _contextual_embedding(model, tokenizer, text: str, layer: int) -> torch.Tensor:
    prompt = format_messages([{"role": "user", "content": f"Represent this concept: {text}"}], tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(_model_device(model)) for key, value in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    mask = encoded["attention_mask"][0].bool()
    return hidden[mask].mean(dim=0)


def _tensorize(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.long, device=device)


def _last_index(attention_mask: torch.Tensor, last_token: int) -> int:
    seq_length = attention_mask.shape[-1]
    length = int(attention_mask.long().sum().item())
    return max(0, min(seq_length - 1, length + last_token))


@torch.no_grad()
def _jepa_view_embedding(
    model,
    input_ids: list[int],
    attention_mask: list[int],
    layer: int,
    last_token: int,
) -> torch.Tensor:
    device = _model_device(model)
    encoded = {
        "input_ids": _tensorize(input_ids, device),
        "attention_mask": _tensorize(attention_mask, device),
    }
    outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    return hidden[_last_index(encoded["attention_mask"], last_token)]


def _preview(messages: list[dict[str, str]], max_chars: int) -> str:
    text = " ".join(message["content"].replace("\n", " ").strip() for message in messages)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _jepa_embeddings(
    model,
    tokenizer,
    data_file: Path,
    max_examples: int,
    max_length: int,
    predictors: int,
    last_token: int,
    layer: int,
    plain: bool,
    reverse_pred: bool,
    front_pred: bool,
    label_chars: int,
) -> tuple[list[str], list[np.ndarray], list[str], list[tuple[int, int]]]:
    dataset = ChatJEPADataset(
        data_file,
        tokenizer,
        max_length=max_length,
        predictors=predictors,
        plain=plain,
        reverse_pred=reverse_pred,
        front_pred=front_pred,
    )
    labels = []
    vectors = []
    kinds = []
    pairs = []
    for index in range(min(max_examples, len(dataset))):
        item = dataset[index]
        record = dataset.records[index]
        user_messages, assistant_messages = dataset._views(record["messages"])
        user_vector = _jepa_view_embedding(
            model,
            item["input_ids_user"],
            item["attention_mask_user"],
            layer,
            last_token,
        )
        assistant_vector = _jepa_view_embedding(
            model,
            item["input_ids_assistant"],
            item["attention_mask_assistant"],
            layer,
            last_token,
        )
        labels.extend(
            [
                f"{index} input: {_preview(user_messages, label_chars)}",
                f"{index} target: {_preview(assistant_messages, label_chars)}",
            ]
        )
        vectors.extend([user_vector.float().cpu().numpy(), assistant_vector.float().cpu().numpy()])
        kinds.extend(["input", "target"])
        pairs.append((len(vectors) - 2, len(vectors) - 1))
    return labels, vectors, kinds, pairs


def _project(embeddings: np.ndarray, method: Literal["pca", "tsne"]) -> np.ndarray:
    if len(embeddings) < 2:
        raise ValueError("Need at least two terms to make a 2D plot.")
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    perplexity = max(2, min(5, len(embeddings) - 1))
    return TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity).fit_transform(
        embeddings
    )


def _plot(
    points: np.ndarray,
    terms: list[str],
    output: Path,
    title: str,
    kinds: list[str] | None = None,
    pairs: list[tuple[int, int]] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save latent-space plots. "
            "Install it with `uv pip install -e .` or reinstall the project dependencies."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    if pairs:
        for start, end in pairs:
            plt.plot(
                [points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.45,
                zorder=1,
            )

    if kinds:
        colors = {"input": "#1f77b4", "target": "#ff7f0e"}
        markers = {"input": "o", "target": "s"}
        for kind in sorted(set(kinds)):
            indices = [index for index, value in enumerate(kinds) if value == kind]
            plt.scatter(
                points[indices, 0],
                points[indices, 1],
                s=70,
                c=colors.get(kind, "#2ca02c"),
                marker=markers.get(kind, "o"),
                label=kind,
                zorder=2,
            )
        plt.legend(title="JEPA view")
    else:
        plt.scatter(points[:, 0], points[:, 1], s=70, zorder=2)

    for term, (x_coord, y_coord) in zip(terms, points):
        plt.annotate(
            term,
            (x_coord, y_coord),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8 if kinds else 10,
        )
    plt.axhline(0, color="lightgray", linewidth=0.8)
    plt.axvline(0, color="lightgray", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main(
    checkpoint: Annotated[
        str,
        typer.Option(
            "--checkpoint",
            "-c",
            help="Model id or local checkpoint. The shorthand run/checkpoint resolves automatically.",
        ),
    ] = "Qwen/Qwen2.5-1.5B-Instruct",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="PNG path for the latent-space plot."),
    ] = Path("runs/latent_space.png"),
    terms: Annotated[
        list[str] | None,
        typer.Option("--term", help="Term to plot. Repeat this option for multiple terms."),
    ] = None,
    mode: Annotated[
        Literal["token", "contextual", "jepa"],
        typer.Option(
            help=(
                "Use static token embeddings, contextual hidden states, or JEPA input/target "
                "view embeddings from a dataset."
            )
        ),
    ] = "token",
    projection: Annotated[
        Literal["pca", "tsne"],
        typer.Option(help="2D projection method."),
    ] = "pca",
    layer: Annotated[
        int,
        typer.Option(help="Hidden-state layer for contextual and JEPA modes. -1 means final layer."),
    ] = -1,
    data_file: Annotated[
        Path,
        typer.Option("--data-file", help="JSONL dataset used when --mode jepa."),
    ] = Path("data/gsm8k/gsm8k_train.jsonl"),
    max_examples: Annotated[
        int,
        typer.Option(help="Maximum dataset examples to plot in JEPA mode."),
    ] = 16,
    max_length: Annotated[
        int,
        typer.Option(help="Tokenizer max length for JEPA dataset examples."),
    ] = 2048,
    predictors: Annotated[
        int,
        typer.Option(help="Predictor token count for JEPA input views."),
    ] = 0,
    last_token: Annotated[
        int,
        typer.Option(help="Representation token offset used by the JEPA loss."),
    ] = -1,
    plain: Annotated[
        bool,
        typer.Option(help="Use plain text formatting for JEPA dataset examples."),
    ] = False,
    reverse_pred: Annotated[
        bool,
        typer.Option(help="Plot reversed input/target JEPA views."),
    ] = False,
    front_pred: Annotated[
        bool,
        typer.Option(help="Place predictor tokens before the input view text."),
    ] = False,
    label_chars: Annotated[
        int,
        typer.Option(help="Maximum text characters per JEPA point label."),
    ] = 52,
) -> None:
    selected_terms = terms or DEFAULT_TERMS
    resolved_checkpoint = resolve_checkpoint(checkpoint)
    model, tokenizer = load_model_and_tokenizer(resolved_checkpoint, device_map=None)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if mode == "jepa":
        labels, vectors, kinds, pairs = _jepa_embeddings(
            model,
            tokenizer,
            data_file,
            max_examples,
            max_length,
            predictors,
            last_token,
            layer,
            plain,
            reverse_pred,
            front_pred,
            label_chars,
        )
    else:
        labels = selected_terms
        kinds = None
        pairs = None
        vectors = []
        for term in selected_terms:
            if mode == "token":
                vector = _mean_token_embedding(model, tokenizer, term)
            else:
                vector = _contextual_embedding(model, tokenizer, term, layer)
            vectors.append(vector.float().cpu().numpy())

    embeddings = np.vstack(vectors)
    points = _project(embeddings, projection)
    title = f"{mode.title()} latent space ({projection.upper()}): {Path(resolved_checkpoint).name}"
    _plot(points, labels, output, title, kinds=kinds, pairs=pairs)
    typer.echo(f"Wrote latent-space plot: {output}")


if __name__ == "__main__":
    typer.run(main)
