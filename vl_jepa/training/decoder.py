from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vl_jepa.modeling import VLJEPA, VLJEPAConfig
from vl_jepa.training.train import (
    _build_loader,
    _device,
    _move_tensors,
    _scheduler,
    _summary_writer,
    save_training_checkpoint,
)
from vl_jepa.utils.config import load_config, save_config
from vl_jepa.utils.io import make_run_dir
from vl_jepa.utils.seed import seed_everything


def _load_vl_jepa_with_decoder(
    config: dict[str, Any], checkpoint: str | Path, device: torch.device
) -> VLJEPA:
    checkpoint_path = Path(checkpoint)
    base_config = load_config(checkpoint_path / "model_config.yaml")
    decoder_cfg = config.get("decoder", {})
    model_config = {
        **base_config,
        "decoder_model": decoder_cfg.get("model_name", "meta-llama/Llama-3.2-1B"),
        "trust_remote_code": decoder_cfg.get(
            "trust_remote_code", base_config.get("trust_remote_code", True)
        ),
        "hf_token": decoder_cfg.get("hf_token", base_config.get("hf_token", True)),
        "torch_dtype": decoder_cfg.get("torch_dtype", base_config.get("torch_dtype", "auto")),
    }
    model = VLJEPA(VLJEPAConfig(**model_config)).to(device)
    state = torch.load(checkpoint_path / "model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.config = VLJEPAConfig(**model_config)
    return model


def _freeze_for_decoder_training(model: VLJEPA, *, train_decoder_lm: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not model.y_decoder.enabled or model.y_decoder.prefix_projection is None:
        raise RuntimeError("Y-Decoder must be configured for decoder training")
    for parameter in model.y_decoder.prefix_projection.parameters():
        parameter.requires_grad = True
    if train_decoder_lm and model.y_decoder.model is not None:
        for parameter in model.y_decoder.model.parameters():
            parameter.requires_grad = True


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.no_grad()
def evaluate_decoder_loss(
    *,
    model: VLJEPA,
    loader: DataLoader,
    embedding_source: str,
    max_decoder_length: int,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    was_training = model.y_decoder.training
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = _move_tensors(batch, device)
        if embedding_source == "predicted":
            embedding = model.predict_embedding(
                batch["pixel_values"],
                batch["query_input_ids"],
                batch["query_attention_mask"],
            )
        elif embedding_source == "target":
            embedding = model.encode_target(
                batch["target_input_ids"],
                batch["target_attention_mask"],
            )
        else:
            raise ValueError("decoder.embedding_source must be 'predicted' or 'target'")
        decoder_tokens = model.y_decoder.tokenizer(
            batch["target_texts"],
            truncation=True,
            max_length=max_decoder_length,
            padding=True,
            return_tensors="pt",
        )
        loss = model.y_decoder.forward_loss(
            embedding,
            decoder_tokens["input_ids"].to(device),
            decoder_tokens["attention_mask"].to(device),
        )
        losses.append(float(loss.detach().cpu()))
    if was_training:
        model.y_decoder.train()
    return sum(losses) / len(losses) if losses else 0.0


def train_y_decoder_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    seed_everything(
        int(train_cfg.get("seed", 42)),
        deterministic=bool(train_cfg.get("deterministic", False)),
    )
    run_dir = make_run_dir(config.get("run_root", "runs"), config.get("name", "vl_jepa_decoder"))
    save_config(config, run_dir / "config.yaml")

    device = _device()
    checkpoint = config["checkpoint"]
    model = _load_vl_jepa_with_decoder(config, checkpoint, device)
    _freeze_for_decoder_training(
        model,
        train_decoder_lm=bool(config.get("decoder", {}).get("train_lm", False)),
    )

    model_cfg = model.config
    data_cfg = config.get("data", {})
    loader = _build_loader(
        data_file=data_cfg["train_file"],
        model=model,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = int(train_cfg.get("grad_accum", 1))
    max_steps = int(train_cfg.get("max_steps", 0))
    epochs = int(train_cfg.get("num_epochs", 1))
    log_steps = int(train_cfg.get("log_steps", 1))
    checkpoint_steps = int(train_cfg.get("checkpoint_steps", 0))
    eval_steps = int(train_cfg.get("eval_steps", 0))
    eval_loader = None
    if eval_steps and data_cfg.get("eval_file"):
        eval_loader = _build_loader(
            data_file=data_cfg["eval_file"],
            model=model,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            shuffle=False,
        )
    max_eval_batches = train_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    steps_per_epoch = max(1, ceil(len(loader) / max(1, grad_accum)))
    total_steps = max_steps or epochs * steps_per_epoch
    scheduler = _scheduler(optimizer, train_cfg, total_steps)
    writer = _summary_writer(train_cfg, run_dir)
    embedding_source = str(config.get("decoder", {}).get("embedding_source", "predicted"))
    max_decoder_length = int(
        config.get("decoder", {}).get("max_length", model_cfg.max_target_length)
    )

    global_step = 0
    micro_step = 0
    try:
        model.eval()
        model.y_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        epoch = 0
        while True:
            epoch += 1
            progress = tqdm(loader, desc=f"VL-JEPA decoder train epoch {epoch}")
            for batch in progress:
                micro_step += 1
                batch = _move_tensors(batch, device)
                with torch.no_grad():
                    if embedding_source == "predicted":
                        embedding = model.predict_embedding(
                            batch["pixel_values"],
                            batch["query_input_ids"],
                            batch["query_attention_mask"],
                        )
                    elif embedding_source == "target":
                        embedding = model.encode_target(
                            batch["target_input_ids"],
                            batch["target_attention_mask"],
                        )
                    else:
                        raise ValueError("decoder.embedding_source must be 'predicted' or 'target'")
                decoder_tokens = model.y_decoder.tokenizer(
                    batch["target_texts"],
                    truncation=True,
                    max_length=max_decoder_length,
                    padding=True,
                    return_tensors="pt",
                )
                loss = model.y_decoder.forward_loss(
                    embedding,
                    decoder_tokens["input_ids"].to(device),
                    decoder_tokens["attention_mask"].to(device),
                )
                loss_value = float(loss.detach().cpu())
                (loss / grad_accum).backward()
                if micro_step % grad_accum == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.set_postfix(loss=loss_value)
                    if writer is not None and global_step % log_steps == 0:
                        writer.add_scalar("decoder/loss", loss_value, global_step)
                        writer.add_scalar(
                            "decoder/lr", optimizer.param_groups[0]["lr"], global_step
                        )
                        writer.add_scalar("decoder/epoch", epoch, global_step)
                    if eval_loader is not None and global_step % eval_steps == 0:
                        eval_loss = evaluate_decoder_loss(
                            model=model,
                            loader=eval_loader,
                            embedding_source=embedding_source,
                            max_decoder_length=max_decoder_length,
                            device=device,
                            max_batches=max_eval_batches,
                        )
                        progress.set_postfix(loss=loss_value, eval_loss=eval_loss)
                        if writer is not None:
                            writer.add_scalar("decoder_eval/loss", eval_loss, global_step)
                    if checkpoint_steps and global_step % checkpoint_steps == 0:
                        save_training_checkpoint(
                            model=model,
                            checkpoint_dir=run_dir / f"checkpoint-step-{global_step}",
                            model_config=model.config.__dict__,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            epoch=epoch,
                        )
                    if max_steps and global_step >= max_steps:
                        break
            if max_steps and global_step >= max_steps:
                break
            if not max_steps and epoch >= epochs:
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    checkpoint_dir = run_dir / "checkpoint-final"
    save_training_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        model_config=model.config.__dict__,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=epoch,
    )
    if model.y_decoder.tokenizer is not None:
        model.y_decoder.tokenizer.save_pretrained(checkpoint_dir / "decoder_tokenizer")
    return run_dir
