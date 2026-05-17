from __future__ import annotations

import math
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vl_jepa.data.collators import VLJEPADataCollator
from vl_jepa.data.datasets import VLJEPAManifestDataset
from vl_jepa.modeling import VLJEPAConfig, build_vl_jepa_model
from vl_jepa.training.losses import bidirectional_infonce_loss
from vl_jepa.utils.config import load_config, save_config
from vl_jepa.utils.io import make_run_dir
from vl_jepa.utils.seed import seed_everything


def vl_jepa_optimizer(
    model: torch.nn.Module, config: VLJEPAConfig, learning_rate: float, weight_decay: float
):
    y_encoder_ids = {id(parameter) for parameter in model.y_encoder.parameters()}
    y_encoder = []
    base = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in y_encoder_ids:
            y_encoder.append(parameter)
        else:
            base.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": base, "lr": learning_rate, "weight_decay": weight_decay},
            {
                "params": y_encoder,
                "lr": learning_rate * config.y_encoder_lr_multiplier,
                "weight_decay": weight_decay,
            },
        ]
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _summary_writer(train_cfg: dict[str, Any], run_dir: Path) -> SummaryWriter | None:
    if not bool(train_cfg.get("tensorboard", True)):
        return None
    log_dir = Path(train_cfg.get("log_dir") or run_dir / "tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(train_cfg.get("scheduler", "constant")).lower()
    if scheduler_name in {"", "none", "constant"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=float(train_cfg.get("min_learning_rate", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _resolve_init_checkpoint(train_cfg: dict[str, Any], run_root: str | Path) -> Path | None:
    init_checkpoint = train_cfg.get("init_checkpoint")
    if init_checkpoint:
        checkpoint_path = Path(init_checkpoint)
        return checkpoint_path if checkpoint_path.is_file() else checkpoint_path / "model.pt"

    run_name = train_cfg.get("init_checkpoint_run_name")
    if run_name:
        candidates = sorted(Path(run_root).glob(f"*-{run_name}/checkpoint-final/model.pt"))
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(
            f"No checkpoint found for init_checkpoint_run_name={run_name!r} under {run_root!r}"
        )

    if bool(train_cfg.get("require_init_checkpoint", False)):
        raise ValueError("This config requires init_checkpoint or init_checkpoint_run_name")
    return None


def _positive_mask_from_texts(texts: list[str], device: torch.device) -> torch.Tensor:
    normalized = [text.strip().lower() for text in texts]
    return torch.tensor(
        [[left == right for right in normalized] for left in normalized],
        dtype=torch.bool,
        device=device,
    )


def _embedding_retrieval_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    predicted_norm = torch.nn.functional.normalize(predicted, p=2, dim=-1)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=-1)
    scores = predicted_norm @ target_norm.T
    labels = torch.arange(scores.size(0), device=scores.device)
    order = torch.argsort(scores, dim=1, descending=True)
    strict_top1 = (order[:, 0] == labels).float().mean()
    strict_ranks = (order == labels[:, None]).float().argmax(dim=1) + 1
    strict_mrr = (1.0 / strict_ranks.float()).mean()
    if positive_mask is None:
        positive_mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
    else:
        positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    ordered_positive = torch.gather(positive_mask, 1, order)
    positive_top1 = ordered_positive[:, 0].float().mean()
    first_positive = ordered_positive.float().argmax(dim=1) + 1
    positive_mrr = (1.0 / first_positive.float()).mean()
    positives_per_query = positive_mask.float().sum(dim=1).mean()
    return {
        "top1": float(strict_top1.detach().cpu()),
        "mrr": float(strict_mrr.detach().cpu()),
        "positive_top1": float(positive_top1.detach().cpu()),
        "positive_mrr": float(positive_mrr.detach().cpu()),
        "positives_per_query": float(positives_per_query.detach().cpu()),
    }


def save_training_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_dir: Path,
    model_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    global_step: int | None = None,
    epoch: int | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    save_config(model_config, checkpoint_dir / "model_config.yaml")
    query_tokenizer = getattr(model, "query_tokenizer", None)
    target_tokenizer = getattr(model, "target_tokenizer", None)
    if query_tokenizer is not None:
        query_tokenizer.save_pretrained(checkpoint_dir / "query_tokenizer")
    if target_tokenizer is not None:
        target_tokenizer.save_pretrained(checkpoint_dir / "target_tokenizer")
    training_state: dict[str, Any] = {}
    if optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()
    if global_step is not None:
        training_state["global_step"] = global_step
    if epoch is not None:
        training_state["epoch"] = epoch
    if training_state:
        torch.save(training_state, checkpoint_dir / "training_state.pt")


def _build_loader(
    *,
    data_file: str,
    model: torch.nn.Module,
    model_cfg: VLJEPAConfig,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    dataset = VLJEPAManifestDataset(
        data_file,
        model.query_tokenizer,
        model.target_tokenizer,
        num_frames=int(model_cfg.num_frames),
        image_size=int(model_cfg.image_size),
        max_query_length=int(model_cfg.max_query_length),
        max_target_length=int(model_cfg.max_target_length),
        query_override=data_cfg.get("query_override"),
    )
    collator = VLJEPADataCollator(model.query_tokenizer, model.target_tokenizer)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )


@torch.no_grad()
def evaluate_vl_jepa_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    model_cfg: VLJEPAConfig,
    train_cfg: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    contrastive_accum_batches = int(train_cfg.get("eval_contrastive_accum_batches", 0)) or int(
        train_cfg.get("contrastive_accum_batches", 1)
    )
    pending_predicted: list[torch.Tensor] = []
    pending_targets: list[torch.Tensor] = []
    pending_target_texts: list[str] = []
    losses = []
    top1_scores = []
    mrr_scores = []
    positive_top1_scores = []
    positive_mrr_scores = []
    positives_per_query_scores = []
    contrastive_batch_sizes = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = _move_tensors(batch, device)
        embeddings = model.forward_embeddings(batch)
        pending_predicted.append(embeddings["predicted_embedding"])
        pending_targets.append(embeddings["target_embedding"])
        pending_target_texts.extend(batch.get("target_texts", []))
        if len(pending_predicted) < contrastive_accum_batches:
            continue
        predicted = torch.cat(pending_predicted, dim=0)
        target = torch.cat(pending_targets, dim=0)
        pending_predicted.clear()
        pending_targets.clear()
        positive_mask = None
        if bool(train_cfg.get("multi_positive_by_text", False)):
            positive_mask = _positive_mask_from_texts(pending_target_texts, predicted.device)
        pending_target_texts.clear()
        loss = bidirectional_infonce_loss(
            predicted,
            target,
            temperature=float(model_cfg.temperature),
            positive_mask=positive_mask,
        )
        retrieval_metrics = _embedding_retrieval_metrics(predicted, target, positive_mask)
        losses.append(float(loss.detach().cpu()))
        top1_scores.append(retrieval_metrics["top1"])
        mrr_scores.append(retrieval_metrics["mrr"])
        positive_top1_scores.append(retrieval_metrics["positive_top1"])
        positive_mrr_scores.append(retrieval_metrics["positive_mrr"])
        positives_per_query_scores.append(retrieval_metrics["positives_per_query"])
        contrastive_batch_sizes.append(predicted.size(0))
    if was_training:
        model.train()
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    mean_batch_size = (
        sum(contrastive_batch_sizes) / len(contrastive_batch_sizes)
        if contrastive_batch_sizes
        else 0.0
    )
    return {
        "loss": mean_loss,
        "top1": sum(top1_scores) / len(top1_scores) if top1_scores else 0.0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "positive_top1": (
            sum(positive_top1_scores) / len(positive_top1_scores) if positive_top1_scores else 0.0
        ),
        "positive_mrr": (
            sum(positive_mrr_scores) / len(positive_mrr_scores) if positive_mrr_scores else 0.0
        ),
        "positives_per_query": (
            sum(positives_per_query_scores) / len(positives_per_query_scores)
            if positives_per_query_scores
            else 0.0
        ),
        "random_baseline_loss": math.log(mean_batch_size) if mean_batch_size > 0 else 0.0,
        "contrastive_batch_size": mean_batch_size,
    }


def train_vl_jepa_from_config(config_path: str | Path) -> Path:
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    seed_everything(
        int(train_cfg.get("seed", 42)),
        deterministic=bool(train_cfg.get("deterministic", False)),
    )
    run_dir = make_run_dir(config.get("run_root", "runs"), config.get("name", "vl_jepa"))
    save_config(config, run_dir / "config.yaml")

    model = build_vl_jepa_model(config)
    model_cfg = VLJEPAConfig(**config.get("model", {}))
    data_cfg = config.get("data", {})
    loader = _build_loader(
        data_file=data_cfg["train_file"],
        model=model,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        shuffle=True,
    )
    device = _device()
    model.to(device)
    state_path = _resolve_init_checkpoint(train_cfg, config.get("run_root", "runs"))
    if state_path is not None:
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)
    optimizer = vl_jepa_optimizer(
        model,
        model_cfg,
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = int(train_cfg.get("grad_accum", 1))
    contrastive_accum_batches = int(train_cfg.get("contrastive_accum_batches", 1))
    if int(train_cfg.get("batch_size", 1)) * contrastive_accum_batches < 2:
        raise ValueError(
            "VL-JEPA InfoNCE needs an actual contrastive batch of at least 2. "
            "Increase batch_size or contrastive_accum_batches."
        )
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
    steps_per_epoch = max(1, ceil(len(loader) / max(1, grad_accum * contrastive_accum_batches)))
    total_steps = max_steps or epochs * steps_per_epoch
    scheduler = _scheduler(optimizer, train_cfg, total_steps)
    writer = _summary_writer(train_cfg, run_dir)
    global_step = 0
    contrastive_step = 0
    ema_loss: float | None = None
    pending_predicted: list[torch.Tensor] = []
    pending_targets: list[torch.Tensor] = []
    pending_target_texts: list[str] = []
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch = 0
        while True:
            epoch += 1
            progress = tqdm(loader, desc=f"VL-JEPA train epoch {epoch}")
            for batch in progress:
                batch = _move_tensors(batch, device)
                embeddings = model.forward_embeddings(batch)
                pending_predicted.append(embeddings["predicted_embedding"])
                pending_targets.append(embeddings["target_embedding"])
                pending_target_texts.extend(batch.get("target_texts", []))
                if len(pending_predicted) < contrastive_accum_batches:
                    continue

                predicted = torch.cat(pending_predicted, dim=0)
                target = torch.cat(pending_targets, dim=0)
                pending_predicted.clear()
                pending_targets.clear()
                positive_mask = None
                if bool(train_cfg.get("multi_positive_by_text", False)):
                    positive_mask = _positive_mask_from_texts(
                        pending_target_texts,
                        predicted.device,
                    )
                pending_target_texts.clear()
                loss = bidirectional_infonce_loss(
                    predicted,
                    target,
                    temperature=float(model_cfg.temperature),
                    positive_mask=positive_mask,
                )
                loss_value = float(loss.detach().cpu())
                ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value
                loss = loss / grad_accum
                loss.backward()
                contrastive_step += 1
                if contrastive_step % grad_accum == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.set_postfix(loss=loss_value)
                    if writer is not None and global_step % log_steps == 0:
                        writer.add_scalar("train/loss", loss_value, global_step)
                        writer.add_scalar("train/loss_ema", ema_loss, global_step)
                        writer.add_scalar(
                            "train/random_baseline_loss", math.log(predicted.size(0)), global_step
                        )
                        writer.add_scalar(
                            "train/contrastive_batch_size", predicted.size(0), global_step
                        )
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                        writer.add_scalar(
                            "train/y_encoder_lr",
                            optimizer.param_groups[1]["lr"],
                            global_step,
                        )
                        writer.add_scalar("train/epoch", epoch, global_step)
                    if eval_loader is not None and global_step % eval_steps == 0:
                        eval_metrics = evaluate_vl_jepa_loss(
                            model=model,
                            loader=eval_loader,
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            device=device,
                            max_batches=max_eval_batches,
                        )
                        progress.set_postfix(loss=loss_value, eval_loss=eval_metrics["loss"])
                        if writer is not None:
                            writer.add_scalar("eval/loss", eval_metrics["loss"], global_step)
                            writer.add_scalar("eval/top1", eval_metrics["top1"], global_step)
                            writer.add_scalar("eval/mrr", eval_metrics["mrr"], global_step)
                            writer.add_scalar(
                                "eval/positive_top1",
                                eval_metrics["positive_top1"],
                                global_step,
                            )
                            writer.add_scalar(
                                "eval/positive_mrr",
                                eval_metrics["positive_mrr"],
                                global_step,
                            )
                            writer.add_scalar(
                                "eval/positives_per_query",
                                eval_metrics["positives_per_query"],
                                global_step,
                            )
                            writer.add_scalar(
                                "eval/random_baseline_loss",
                                eval_metrics["random_baseline_loss"],
                                global_step,
                            )
                            writer.add_scalar(
                                "eval/contrastive_batch_size",
                                eval_metrics["contrastive_batch_size"],
                                global_step,
                            )
                    if checkpoint_steps and global_step % checkpoint_steps == 0:
                        save_training_checkpoint(
                            model=model,
                            checkpoint_dir=run_dir / f"checkpoint-step-{global_step}",
                            model_config=config.get("model", {}),
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

    save_training_checkpoint(
        model=model,
        checkpoint_dir=run_dir / "checkpoint-final",
        model_config=config.get("model", {}),
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=epoch,
    )
    return run_dir
