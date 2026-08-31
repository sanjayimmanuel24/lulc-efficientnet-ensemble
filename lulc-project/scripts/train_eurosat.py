"""
End-to-end training runner for the Dual-Branch EfficientNet-B0 ensemble on
real EuroSAT-MS data.

Usage (from inside lulc-project/, after extracting EuroSAT-MS so that
TIF_ROOT below points at the folder containing all 10 class subfolders):

    python scripts/train_eurosat.py

Quick smoke-test run (few minutes, to check everything is wired correctly
before committing to a full run):

    python scripts/train_eurosat.py --epochs-per-stage 1 1 1 --image-sizes 64 96 128

Full run (closer to what docs/BLUEPRINT.md Section 8 specifies):

    python scripts/train_eurosat.py --epochs-per-stage 5 5 10 --image-sizes 128 160 224

This script runs a *single* stratified 70/15/15 split, saved to
data/eurosat_split.json on first run and reused on every later run (see
src/data/splits.py docstring for why that matters). This is the first
real training run to confirm the pipeline works end-to-end on real data --
it is not yet the full k=5 cross-validation protocol from Section 11,
which a separate ablation-runner script will build on top of this same
split-building code.

Note on progressive resizing here specifically: EuroSAT-MS patches are
natively 64x64 pixels. Resizing up to 224x224 is upsampling (no new pixel
information is created), done to match the ImageNet-pretrained backbone's
expected input scale -- standard practice for transfer learning on small
satellite patches, but worth knowing it's not "revealing more detail" the
way progressive resizing does on naturally high-resolution datasets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Running this file directly (python scripts\train_eurosat.py) only puts
# scripts\ on sys.path, not the project root -- so `import src...` fails
# with ModuleNotFoundError. Insert the project root explicitly, the same
# fix conftest.py applies for pytest, so this script works regardless of
# how it's invoked (direct, `python -m scripts.train_eurosat`, an IDE
# "run" button, etc.).
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, config_defaults, unmapped_keys, dump_resolved_config
from src.data.splits import build_or_load_split, CLASS_NAMES
from src.data.dataset import EuroSATMSDataset
from src.data.transforms import EuroSATTransform
from src.models.ensemble import DualBranchEfficientNet
from src.training.train import TrainConfig, fit, evaluate
from src.losses.label_smoothing_torch import build_criterion
from src.calibration.temperature_scaling import fit_temperature, expected_calibration_error
from src.evaluation.metrics import compute_all_metrics

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)
SPLIT_PATH = PROJECT_ROOT / "data" / "eurosat_split.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "dual_branch_best.pt"
LAST_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "dual_branch_last.pt"
RESULTS_PATH = PROJECT_ROOT / "results" / "first_run_metrics.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs-per-stage", type=int, nargs="+", default=[3, 3, 6])
    p.add_argument("--image-sizes", type=int, nargs="+", default=[128, 160, 224])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 is the safe default on Windows; DataLoader workers>0 need "
                        "the __main__ guard this script already has, but can still be "
                        "slow to spin up on Windows -- raise this only if training is CPU-bound.")
    p.add_argument("--spectral-indices", type=str, nargs="+", default=["ndvi"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--attention", choices=["eca", "none"], default="eca",
                   help="Channel attention in each branch. 'none' is the C3 ablation "
                        "(src/models/ensemble.py).")
    p.add_argument("--pin-memory", action="store_true",
                   help="Pin host memory for faster H2D copies. Off by default: pinned pages are "
                        "non-pageable and count against the Windows commit limit, which on a "
                        "small-RAM machine shows up as 'Couldn't open shared file mapping ... 1455'.")
    p.add_argument("--checkpoint-selection", choices=["final_stage", "global"], default="final_stage",
                   help="Which progressive-resizing stages a checkpoint may be selected from. "
                        "'final_stage' (default) only considers the last stage, whose resolution "
                        "matches the evaluation resolution; 'global' compares across all stages, "
                        "which is not apples-to-apples (see src/training/train.py).")
    # config.yaml supplies defaults; an explicit flag on the command line still wins.
    cfg_file = load_config()
    p.set_defaults(**config_defaults(cfg_file))
    args = p.parse_args()
    args._config_defaults_applied = sorted(config_defaults(cfg_file))
    args._config_keys_not_consumed = unmapped_keys(cfg_file)
    return args


def make_loader(split_data, image_size, batch_size, num_workers, train, spectral_indices,
                pin_memory=False):
    ds = EuroSATMSDataset(
        file_paths=split_data["files"],
        labels=split_data["labels"],
        indices=tuple(spectral_indices),
        transform=EuroSATTransform(image_size, train=train),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(), drop_last=train,
    )


def main():
    args = parse_args()

    if len(args.epochs_per_stage) != len(args.image_sizes):
        raise ValueError(
            f"--epochs-per-stage ({len(args.epochs_per_stage)} values) and "
            f"--image-sizes ({len(args.image_sizes)} values) must have the same length."
        )

    if not TIF_ROOT.exists():
        raise FileNotFoundError(
            f"EuroSAT-MS tif folder not found at:\n  {TIF_ROOT}\n"
            "Double-check the extraction path, or edit TIF_ROOT at the top of this script "
            "if your folder structure differs."
        )

    print(f"Data root: {TIF_ROOT}")
    print("Building / loading train-val-test split...")
    split = build_or_load_split(TIF_ROOT, SPLIT_PATH)
    print(f"  train={len(split['train']['files'])}  "
          f"val={len(split['val']['files'])}  "
          f"test={len(split['test']['files'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args._config_keys_not_consumed:
        print("note: config.yaml keys no runner consumes: " + ", ".join(args._config_keys_not_consumed))
    print("resolved config -> " + str(dump_resolved_config(args, RESULTS_PATH.parent, {"device": str(device)})))
    if device.type == "cpu":
        print("  WARNING: no GPU detected -- training will be slow. Consider a smaller "
              "smoke-test run first: --epochs-per-stage 1 1 1 --image-sizes 64 96 128")

    num_indices = len(args.spectral_indices)
    cfg = TrainConfig(
        num_classes=len(CLASS_NAMES),
        num_spectral_indices=num_indices,
        epochs_per_stage=tuple(args.epochs_per_stage),
        image_sizes=tuple(args.image_sizes),
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_selection=args.checkpoint_selection,
        checkpoint_path=str(CHECKPOINT_PATH),   # persist best weights every improvement
        last_checkpoint_path=str(LAST_CHECKPOINT_PATH),  # and the latest epoch, always
    )

    def train_loader_fn(image_size):
        return make_loader(split["train"], image_size, cfg.batch_size, args.num_workers,
                            train=True, spectral_indices=args.spectral_indices,
                            pin_memory=args.pin_memory)

    def val_loader_fn(image_size):
        return make_loader(split["val"], image_size, cfg.batch_size, args.num_workers,
                            train=False, spectral_indices=args.spectral_indices,
                            pin_memory=args.pin_memory)

    model = DualBranchEfficientNet(
        num_classes=cfg.num_classes, num_spectral_indices=num_indices, pretrained=True,
        attention=args.attention,
    ).to(device)

    print("\n=== Training ===")
    start = time.time()
    model = fit(model, train_loader_fn, val_loader_fn, cfg, device)
    print(f"Training finished in {(time.time() - start) / 60:.1f} minutes")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved best checkpoint to {CHECKPOINT_PATH}")

    # --- Fit calibration (C4) on validation logits at final resolution ---
    print("\n=== Fitting temperature scaling on validation split ===")
    final_size = cfg.image_sizes[-1]
    criterion = build_criterion(cfg.label_smoothing)
    val_loader = val_loader_fn(final_size)
    val_metrics = evaluate(model, val_loader, criterion, cfg, device)

    t_rgb = fit_temperature(val_metrics["logits_rgb"].numpy(), val_metrics["labels"].numpy())
    t_spectral = fit_temperature(val_metrics["logits_spectral"].numpy(), val_metrics["labels"].numpy())
    model.set_temperatures(t_rgb, t_spectral)
    print(f"  fitted T_rgb={t_rgb:.3f}  T_spectral={t_spectral:.3f}")
    print("  (T > 1 means the branch was overconfident pre-calibration, T < 1 underconfident)")

    # --- Final test-set evaluation (model now applies calibrated fusion internally) ---
    print("\n=== Evaluating on held-out test set ===")
    test_loader = make_loader(split["test"], final_size, cfg.batch_size, args.num_workers,
                               train=False, spectral_indices=args.spectral_indices,
                            pin_memory=args.pin_memory)
    test_metrics = evaluate(model, test_loader, criterion, cfg, device)

    labels = test_metrics["labels"].numpy()
    fused_probs = test_metrics["fused_probs"].numpy()
    preds = fused_probs.argmax(axis=-1)

    metrics = compute_all_metrics(labels, preds, fused_probs)
    ece = expected_calibration_error(fused_probs, labels)

    print("\n--- Test set results (calibrated, confidence-fused) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  ece: {ece:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            **{k: float(v) for k, v in metrics.items()},
            "ece": ece,
            "t_rgb": t_rgb,
            "t_spectral": t_spectral,
            "spectral_indices": args.spectral_indices,
            "image_sizes": args.image_sizes,
            "epochs_per_stage": args.epochs_per_stage,
        }, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    # Required on Windows: DataLoader with num_workers>0 uses spawn-based
    # multiprocessing, which re-imports this module in each worker process.
    # Without this guard, that re-import would re-trigger training itself
    # in every worker, not just parse function definitions.
    main()
