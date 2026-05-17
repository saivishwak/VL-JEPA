from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm_jepa.evaluation.metrics import accuracy, exact_match, gsm8k_match, startswith_match
from llm_jepa.evaluation.spider_eval import spider_execution_match
from llm_jepa.inference.generate import generate_text
from llm_jepa.utils.config import load_config
from llm_jepa.utils.io import make_run_dir, read_jsonl, write_json, write_jsonl


def _score(task: str, prediction: str, target: str, metadata: dict[str, Any], spider_path: str | None) -> bool:
    if task == "gsm8k":
        return gsm8k_match(prediction, target)
    if task == "spider" and spider_path:
        return spider_execution_match(spider_path, str(metadata["db_id"]), prediction, target)
    if task in {"rotten_tomatoes", "yelp", "hellaswag"}:
        return startswith_match(prediction, target)
    return exact_match(prediction, target)


def evaluate_generation(
    checkpoint: str,
    input_file: str | Path,
    output_dir: str | Path,
    task: str,
    max_examples: int | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    spider_path: str | None = None,
) -> Path:
    records = read_jsonl(input_file)
    if max_examples:
        records = records[:max_examples]
    predictions = []
    correct = []
    for record in tqdm(records, desc="Evaluating"):
        messages = record["messages"]
        prompt_messages = messages[:-1]
        target = messages[-1]["content"]
        prediction = generate_text(
            checkpoint,
            prompt_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        ok = _score(task, prediction, target, record.get("metadata", {}), spider_path)
        correct.append(ok)
        predictions.append(
            {
                "id": record.get("id"),
                "task": task,
                "prediction": prediction,
                "target": target,
                "correct": ok,
                "metadata": record.get("metadata", {}),
            }
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, out / "predictions.jsonl")
    metrics = {
        "task": task,
        "checkpoint": checkpoint,
        "examples": len(predictions),
        "accuracy": accuracy(correct),
    }
    metrics_path = out / "metrics.json"
    write_json(metrics, metrics_path)
    with (out / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# Evaluation Report\n\n- Task: `{task}`\n- Examples: {len(predictions)}\n")
        handle.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
    return metrics_path


def evaluate_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = make_run_dir(config.get("run_root", "runs"), config.get("name", "eval"))
    return evaluate_generation(
        checkpoint=config["checkpoint"],
        input_file=config["input_file"],
        output_dir=run_dir,
        task=config["task"],
        max_examples=config.get("max_examples"),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        temperature=float(config.get("temperature", 0.0)),
        spider_path=config.get("spider_path"),
    )
