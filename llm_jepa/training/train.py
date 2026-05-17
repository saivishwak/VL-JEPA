from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments

from llm_jepa.modeling.jepa_trainer import LLMJEPATrainer
from llm_jepa.modeling.lora import apply_lora
from llm_jepa.modeling.tokenizer_setup import load_model_and_tokenizer
from llm_jepa.training.collators import JEPADataCollator
from llm_jepa.training.dataset import ChatJEPADataset
from llm_jepa.utils.config import load_config, save_config
from llm_jepa.utils.io import make_run_dir
from llm_jepa.utils.seed import seed_everything


def _training_args(config: dict[str, Any], run_dir: Path) -> TrainingArguments:
    train_cfg = config.get("training", {})
    args: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": float(train_cfg.get("num_epochs", 1)),
        "per_device_train_batch_size": int(train_cfg.get("batch_size", 1)),
        "per_device_eval_batch_size": int(
            train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 1))
        ),
        "gradient_accumulation_steps": int(train_cfg.get("grad_accum", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-5)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_steps": int(train_cfg.get("save_steps", 500)),
        "eval_steps": int(train_cfg.get("eval_steps", 500)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", False)),
        "report_to": train_cfg.get("report_to", "none"),
        "remove_unused_columns": False,
        "seed": int(train_cfg.get("seed", 42)),
    }
    eval_strategy = "steps" if config.get("data", {}).get("eval_file") else "no"
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    args[strategy_key] = eval_strategy
    return TrainingArguments(**args)


def _processing_kwargs(tokenizer: Any) -> dict[str, Any]:
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def train_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed_everything(int(config.get("training", {}).get("seed", 42)))
    run_dir = make_run_dir(config.get("run_root", "runs"), config.get("name", "llm-jepa"))
    save_config(config, run_dir / "config.yaml")

    model_cfg = config.get("model", {})
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["name"],
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        model = apply_lora(
            model,
            rank=int(lora_cfg.get("rank", 16)),
            dropout=float(lora_cfg.get("dropout", 0.1)),
            target_modules=lora_cfg.get("target_modules"),
        )

    data_cfg = config.get("data", {})
    jepa_cfg = config.get("jepa", {})
    regular = bool(jepa_cfg.get("regular", False))
    train_dataset = ChatJEPADataset(
        data_cfg["train_file"],
        tokenizer,
        max_length=int(data_cfg.get("max_length", 2048)),
        predictors=int(jepa_cfg.get("predictors", 0)),
        regular=regular,
        plain=bool(data_cfg.get("plain", False)),
        reverse_pred=bool(jepa_cfg.get("reverse_pred", False)),
        front_pred=bool(jepa_cfg.get("front_pred", False)),
        train_all=bool(data_cfg.get("train_all", False)),
    )
    eval_dataset = None
    if data_cfg.get("eval_file"):
        eval_dataset = ChatJEPADataset(
            data_cfg["eval_file"],
            tokenizer,
            max_length=int(data_cfg.get("max_length", 2048)),
            predictors=int(jepa_cfg.get("predictors", 0)),
            regular=regular,
            plain=bool(data_cfg.get("plain", False)),
        )
    args = _training_args(config, run_dir)
    collator = JEPADataCollator(tokenizer)
    if regular:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            **_processing_kwargs(tokenizer),
        )
    else:
        trainer = LLMJEPATrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            **_processing_kwargs(tokenizer),
            lbd=float(jepa_cfg.get("lambda", 1.0)),
            gamma=float(jepa_cfg.get("gamma", 1.0)),
            last_token=int(jepa_cfg.get("last_token", -1)),
            loss_mode=str(jepa_cfg.get("loss", "cosine")),
            additive_mask=bool(jepa_cfg.get("additive_mask", False)),
            jepa_ratio=float(jepa_cfg.get("jepa_ratio", -1.0)),
        )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(run_dir / "checkpoint-final"))
    tokenizer.save_pretrained(str(run_dir / "checkpoint-final"))
    return run_dir
