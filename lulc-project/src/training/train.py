"""
Training loop for the Dual-Branch EfficientNet-B0 ensemble.

Kept as a single plain-PyTorch script rather than a PyTorch-Lightning
Module: for a two-branch model with a custom loss (sum of two per-branch
losses) and a post-hoc calibration step, a plain loop is *more* legible
than fighting Lightning's callback hooks for something this small. If the
project later needs multi-GPU or multi-run experiment orchestration,
wrapping this same `train_one_epoch` / `evaluate` pair in a LightningModule
is a small, mechanical change -- deferred until it's actually needed
(YAGNI), per the project's simplicity-first engineering guidelines.

Progressive resizing (128 -> 160 -> 224) follows the original EfficientNet
paper's own training recipe: it speeds up early epochs (small images process
faster, so more gradient steps per wall-clock hour while the model is far
from convergence) and acts as a mild regularizer (the model can't overfit
to fine-grained pixel patterns before it has learned coarse shape/texture
features). It is training-time only and adds zero inference-time cost,
which is why it survived the "does this hurt the efficiency story"
filter that other candidate techniques (e.g. knowledge distillation) did
not -- see docs/BLUEPRINT.md.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.losses.label_smoothing_torch import build_criterion


@dataclass
class TrainConfig:
    num_classes: int = 10
    num_spectral_indices: int = 1
    epochs_per_stage: tuple[int, ...] = (5, 5, 10)     # one entry per progressive-resizing stage
    image_sizes: tuple[int, ...] = (128, 160, 224)
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    seed: int = 42
    branch_loss_weight_rgb: float = 1.0
    branch_loss_weight_spectral: float = 1.0
    # Which stages a checkpoint may be selected from. Progressive resizing
    # validates each stage at a DIFFERENT resolution, so comparing val accuracy
    # across stages is not apples-to-apples: a 96px-stage model can outscore the
    # 128px-stage one and be saved, then get evaluated at 128px (a train/eval
    # resolution mismatch that cost ~5pp in the first smoke test). "final_stage"
    # restricts selection to the last stage, whose resolution matches evaluation.
    # "global" restores the old cross-stage behaviour for comparison.
    checkpoint_selection: str = "final_stage"   # final_stage | global
    # If set, best-so-far weights are written here every time they improve, so an
    # interrupted run does not lose everything (see fit()).
    checkpoint_path: str | None = None
    # If set, the *latest* epoch's weights are written here every epoch, regardless
    # of checkpoint_selection eligibility. Distinct from checkpoint_path, which only
    # ever holds the best *eligible* (final-stage) weights: with
    # checkpoint_selection="final_stage", a crash during the final stage's first
    # validation would otherwise discard every warmup epoch with nothing on disk.
    last_checkpoint_path: str | None = None


def _atomic_save(state, path: Path) -> None:
    """Write-then-rename, so an interrupted save never leaves a truncated .pt behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer_and_scheduler(model: nn.Module, cfg: TrainConfig, total_epochs: int):
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - cfg.warmup_epochs, 1))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
    return optimizer, scheduler


def compute_loss(outputs: dict, labels: torch.Tensor, criterion: nn.Module, cfg: TrainConfig) -> torch.Tensor:
    """
    Each branch is supervised directly on its own logits (see
    DualBranchEfficientNet docstring for why), and the two branch losses
    are combined with fixed weights. The fused-probability output is used
    only for prediction/evaluation, not directly optimized -- it is a
    deterministic function of the two branches' outputs, so supervising
    it separately would be redundant with supervising the branches.
    """
    loss_rgb = criterion(outputs["logits_rgb"], labels)
    loss_spectral = criterion(outputs["logits_spectral"], labels)
    return cfg.branch_loss_weight_rgb * loss_rgb + cfg.branch_loss_weight_spectral * loss_spectral


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, cfg: TrainConfig, device, scaler=None) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        rgb_plus_idx = batch.get("spectral", batch.get("rgb_plus_indices")).to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        if cfg.mixed_precision and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(rgb, rgb_plus_idx)
                loss = compute_loss(outputs, labels, criterion, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(rgb, rgb_plus_idx)
            loss = compute_loss(outputs, labels, criterion, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, cfg: TrainConfig, device) -> dict:
    model.eval()
    running_loss, n_batches, n_correct, n_total = 0.0, 0, 0, 0
    all_logits_rgb, all_logits_spectral, all_fused_probs, all_labels = [], [], [], []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        rgb_plus_idx = batch.get("spectral", batch.get("rgb_plus_indices")).to(device)
        labels = batch["label"].to(device)

        outputs = model(rgb, rgb_plus_idx)
        loss = compute_loss(outputs, labels, criterion, cfg)
        running_loss += loss.item()
        n_batches += 1

        preds = outputs["fused_probs"].argmax(dim=-1)
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

        all_logits_rgb.append(outputs["logits_rgb"].cpu())
        all_logits_spectral.append(outputs["logits_spectral"].cpu())
        all_fused_probs.append(outputs["fused_probs"].cpu())
        all_labels.append(labels.cpu())

    return {
        "loss": running_loss / max(n_batches, 1),
        "accuracy": n_correct / max(n_total, 1),
        "logits_rgb": torch.cat(all_logits_rgb),
        "logits_spectral": torch.cat(all_logits_spectral),
        "fused_probs": torch.cat(all_fused_probs),
        "labels": torch.cat(all_labels),
    }


def fit(model, train_loader_fn, val_loader_fn, cfg: TrainConfig, device) -> nn.Module:
    """
    Args:
        train_loader_fn / val_loader_fn: callables (image_size: int) ->
            DataLoader, so each progressive-resizing stage can rebuild the
            dataloader at a new resolution without this function needing
            to know anything about the dataset implementation.
    """
    set_seed(cfg.seed)
    criterion = build_criterion(cfg.label_smoothing)
    total_epochs = sum(cfg.epochs_per_stage)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_epochs)
    scaler = torch.cuda.amp.GradScaler() if (cfg.mixed_precision and device.type == "cuda") else None

    if cfg.checkpoint_selection not in ("final_stage", "global"):
        raise ValueError(
            f"checkpoint_selection must be 'final_stage' or 'global', got {cfg.checkpoint_selection!r}"
        )

    best_val_acc = -1.0        # best among *eligible* epochs -> what gets checkpointed
    best_state = None
    best_seen_acc = -1.0       # best at the current resolution -> early-stopping signal only
    epochs_without_improvement = 0
    final_stage_idx = len(cfg.epochs_per_stage) - 1

    for stage_idx, (stage_epochs, image_size) in enumerate(zip(cfg.epochs_per_stage, cfg.image_sizes)):
        train_loader = train_loader_fn(image_size)
        val_loader = val_loader_fn(image_size)
        # Patience is per-stage: a resolution change shifts the val-accuracy scale,
        # so carrying a stale non-improvement count across stages would stop runs early.
        best_seen_acc = -1.0
        epochs_without_improvement = 0
        eligible = (cfg.checkpoint_selection == "global") or (stage_idx == final_stage_idx)
        print(f"[stage {stage_idx}] image_size={image_size}, epochs={stage_epochs}"
              f"{'' if eligible else '  (warmup stage: not eligible for checkpoint selection)'}")

        for epoch in range(stage_epochs):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, device, scaler)
            val_metrics = evaluate(model, val_loader, criterion, cfg, device)
            scheduler.step()
            elapsed = time.time() - start
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"val_acc={val_metrics['accuracy']:.4f} ({elapsed:.1f}s)")

            if cfg.last_checkpoint_path is not None:
                _atomic_save(model.state_dict(), Path(cfg.last_checkpoint_path))

            if val_metrics["accuracy"] > best_seen_acc:
                best_seen_acc = val_metrics["accuracy"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if eligible and val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                # Persist immediately: best_state is otherwise only in memory until
                # every stage finishes, so a crash/kill mid-run loses the whole run.
                if cfg.checkpoint_path is not None:
                    _atomic_save(best_state, Path(cfg.checkpoint_path))
                    print(f"    best checkpoint saved (val_acc={best_val_acc:.4f})")

            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"  early stopping (no improvement for {cfg.early_stopping_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # Only reachable if the final stage ran zero epochs. Say so rather than
        # silently returning whatever weights happen to be in memory.
        print("  WARNING: no eligible checkpoint was selected; returning final-epoch weights as-is")
    return model
