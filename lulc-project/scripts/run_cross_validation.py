"""
k-fold cross-validation runner for the dual-branch ensemble (docs/BLUEPRINT.md
Sections 10-11).

Unlike scripts/train_eurosat.py (one split, one model), this trains one model per
fold on a fixed set of stratified folds shared by every variant, and records
per-sample predictions so variants can be compared with paired tests later
(scripts/compare_variants.py).

Usage, from inside lulc-project/:

    # blueprint architecture (spectral branch sees RGB + NDVI)
    python scripts/run_cross_validation.py --tag default --spectral-branch-mode rgb_plus_indices

    # branch-input ablation (spectral branch sees NIR + SWIR + NDVI, no RGB overlap)
    python scripts/run_cross_validation.py --tag nonrgb --spectral-branch-mode nonrgb

Resumable: a fold whose metrics file already exists is skipped, so an interrupted
sweep continues where it stopped rather than restarting. Delete that fold's JSON
to force a re-run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, config_defaults, unmapped_keys, dump_resolved_config
from src.data.splits import (build_or_load_kfolds, materialize_fold, subsample_split,
                             split_in_two, CLASS_NAMES)
from src.data.dataset import EuroSATMSDataset, spectral_in_chans, SPECTRAL_BRANCH_MODES
from src.data.transforms import EuroSATTransform
from src.models.ensemble import DualBranchEfficientNet
from src.models.baseline import SingleBackboneBaseline
from src.models.fusion import confidence_weighted_fusion
from src.training.train import TrainConfig, fit, evaluate
from src.losses.label_smoothing_torch import build_criterion
from src.calibration.temperature_scaling import (
    fit_temperature, softmax, expected_calibration_error,
    adaptive_expected_calibration_error, brier_score,
)
from src.evaluation.metrics import compute_all_metrics

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True,
                   help="Name for this variant; results land in results/cv/<tag>/.")
    p.add_argument("--spectral-branch-mode", choices=list(SPECTRAL_BRANCH_MODES),
                   default="rgb_plus_indices",
                   help="What the spectral branch receives (see src/data/dataset.py).")
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None,
                   help="Run a single fold index and exit (for splitting work across sessions).")
    p.add_argument("--epochs-per-stage", type=int, nargs="+", default=[5, 5, 10])
    p.add_argument("--image-sizes", type=int, nargs="+", default=[128, 160, 224])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2,
                   help="2 is the safe default here; 4 exhausted the Windows commit limit at 224px.")
    p.add_argument("--spectral-indices", type=str, nargs="+", default=["ndvi"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--baseline-backbone", type=str, default=None,
                   help="Train a SINGLE-backbone RGB baseline instead of the dual-branch model "
                        "(e.g. resnet50, efficientnet_b0). Gives the literature comparison "
                        "(Helber et al. 2019 ResNet-50) on OUR folds rather than theirs.")
    p.add_argument("--seed", type=int, default=42,
                   help="Training seed. Vary it (with a distinct --tag) to measure run-to-run "
                        "variance: fold std alone captures data variance, not init/ordering, so "
                        "sub-0.001 differences between variants are otherwise uninterpretable.")
    p.add_argument("--augment", action="store_true",
                   help="Dihedral augmentation (flips + 90-degree rotations) on the training "
                        "loader. Off by default so pre-existing runs stay reproducible.")
    p.add_argument("--calibration-split-frac", type=float, default=0.0,
                   help="Fraction of VALIDATION held out solely for temperature scaling. 0 (default) "
                        "reuses val for both model selection and calibration, which biases reported "
                        "ECE optimistically; 0.5 gives calibration its own unseen data.")
    p.add_argument("--attention", choices=["eca", "none"], default="eca",
                   help="Channel attention in each branch. 'none' is the C3 ablation "
                        "(src/models/ensemble.py).")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--train-subset-frac", type=float, default=1.0,
                   help="Stratified fraction of the TRAINING split to keep (val/test stay full). "
                        "Low-data ablation: C1 has no headroom at full data (branches agree ~98%%, "
                        "mean fusion weight 0.4995), so starving training is how we test whether "
                        "its failure is saturation-specific.")
    # config.yaml supplies defaults; an explicit flag on the command line still wins.
    cfg_file = load_config()
    p.set_defaults(**config_defaults(cfg_file))
    args = p.parse_args()
    args._config_defaults_applied = sorted(config_defaults(cfg_file))
    args._config_keys_not_consumed = unmapped_keys(cfg_file)
    return args


def make_loader(split_data, image_size, args, train, mode):
    ds = EuroSATMSDataset(
        file_paths=split_data["files"], labels=split_data["labels"],
        indices=tuple(args.spectral_indices),
        transform=EuroSATTransform(image_size, train=train,
                                    augment=getattr(args, "augment", False)),
        spectral_branch_mode=mode,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.num_workers,
                      pin_memory=args.pin_memory and torch.cuda.is_available(),
                      drop_last=train)


def run_fold(fold_idx, folds, args, device, out_dir, ckpt_dir):
    metrics_path = out_dir / "fold_{}.json".format(fold_idx)
    if metrics_path.exists():
        print("[fold {}] already done -> {}, skipping".format(fold_idx, metrics_path.name))
        with open(metrics_path) as f:
            return json.load(f)

    split = materialize_fold(folds, fold_idx)
    if args.train_subset_frac < 1.0:
        n_before = len(split["train"]["files"])
        # seed varies per fold so folds don't all see the same subsample pattern
        split["train"] = subsample_split(split["train"], args.train_subset_frac,
                                          seed=42 + fold_idx)
        print("  training subset: {} -> {} images ({:.0%})".format(
            n_before, len(split["train"]["files"]), args.train_subset_frac))
    calib_split = None
    if args.calibration_split_frac > 0.0:
        split["val"], calib_split = split_in_two(split["val"], args.calibration_split_frac,
                                                  seed=args.seed + fold_idx)
        print("  calibration split: {} val kept for selection, {} held out for temperature".format(
            len(split["val"]["files"]), len(calib_split["files"])))
    mode = args.spectral_branch_mode
    n_idx = len(args.spectral_indices)
    spec_chans = spectral_in_chans(mode, n_idx)
    print("")
    print("=== fold {} | mode={} | spectral branch in_chans={} ===".format(fold_idx, mode, spec_chans))
    print("  train={} val={} test={}".format(
        len(split["train"]["files"]), len(split["val"]["files"]), len(split["test"]["files"])))

    cfg = TrainConfig(
        num_classes=len(CLASS_NAMES), num_spectral_indices=n_idx,
        epochs_per_stage=tuple(args.epochs_per_stage), image_sizes=tuple(args.image_sizes),
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        checkpoint_path=str(ckpt_dir / "fold_{}_best.pt".format(fold_idx)),
        last_checkpoint_path=str(ckpt_dir / "fold_{}_last.pt".format(fold_idx)),
    )
    if args.baseline_backbone:
        # Single backbone: zero the spectral loss weight, or compute_loss counts the
        # same logits twice and doubles this run's learning signal vs the dual runs.
        cfg.branch_loss_weight_spectral = 0.0
        model = SingleBackboneBaseline(num_classes=cfg.num_classes,
                                        backbone_name=args.baseline_backbone,
                                        pretrained=True).to(device)
        print("  baseline: single {} (spectral loss weight forced to 0)".format(args.baseline_backbone))
    else:
        model = DualBranchEfficientNet(num_classes=cfg.num_classes, num_spectral_indices=n_idx,
                                       pretrained=True, spectral_in_chans=spec_chans,
                                       attention=args.attention).to(device)

    start = time.time()
    model = fit(model,
                lambda s: make_loader(split["train"], s, args, True, mode),
                lambda s: make_loader(split["val"], s, args, False, mode),
                cfg, device)
    train_minutes = (time.time() - start) / 60
    print("  trained in {:.1f} min".format(train_minutes))

    final_size = cfg.image_sizes[-1]
    criterion = build_criterion(cfg.label_smoothing)

    # fit temperature on the held-out calibration split when one exists, else on val
    calib_source = calib_split if calib_split is not None else split["val"]
    val = evaluate(model, make_loader(calib_source, final_size, args, False, mode),
                   criterion, cfg, device)
    t_rgb = fit_temperature(val["logits_rgb"].numpy(), val["labels"].numpy())
    t_spec = fit_temperature(val["logits_spectral"].numpy(), val["labels"].numpy())
    model.set_temperatures(t_rgb, t_spec)
    print("  T_rgb={:.3f}  T_spectral={:.3f}".format(t_rgb, t_spec))

    test = evaluate(model, make_loader(split["test"], final_size, args, False, mode),
                    criterion, cfg, device)
    y = test["labels"].numpy()
    p_rgb = softmax(test["logits_rgb"].numpy(), t_rgb)
    p_spec = softmax(test["logits_spectral"].numpy(), t_spec)
    p_fused, w_rgb, _ = confidence_weighted_fusion(p_rgb, p_spec)
    p_avg = 0.5 * p_rgb + 0.5 * p_spec

    variants = {"rgb_branch_only": p_rgb, "spectral_branch_only": p_spec,
                "fixed_average_50_50": p_avg, "c1_confidence_fusion": p_fused}
    rows = {}
    for name, probs in variants.items():
        pred = probs.argmax(axis=-1)
        m = compute_all_metrics(y, pred, probs)
        m["ece"] = float(expected_calibration_error(probs, y))
        m["ece_adaptive"] = float(adaptive_expected_calibration_error(probs, y))
        m["brier"] = float(brier_score(probs, y))
        rows[name] = {kk: float(vv) for kk, vv in m.items()}
        print("  {:<24} acc={:.4f} macroF1={:.4f} ECE={:.4f}".format(
            name, m["accuracy"], m["macro_f1"], m["ece"]))

    # per-sample predictions: required for paired McNemar between variants later
    np.savez_compressed(
        out_dir / "fold_{}_preds.npz".format(fold_idx),
        y=y, test_index=np.array(folds["folds"][fold_idx]["test"]),
        pred_rgb=p_rgb.argmax(-1), pred_spectral=p_spec.argmax(-1),
        pred_avg=p_avg.argmax(-1), pred_c1=p_fused.argmax(-1),
        weight_rgb=w_rgb.astype(np.float32),
    )

    payload = {"fold": fold_idx, "tag": args.tag, "spectral_branch_mode": mode,
               "spectral_in_chans": spec_chans, "attention": args.attention,
               "baseline_backbone": args.baseline_backbone,
               "seed": args.seed, "augment": getattr(args, "augment", False),
               "calibration_split_frac": args.calibration_split_frac, "variants": rows,
               "t_rgb": float(t_rgb), "t_spectral": float(t_spec),
               "weight_rgb_mean": float(w_rgb.mean()), "weight_rgb_std": float(w_rgb.std()),
               "branch_agreement_rate": float((p_rgb.argmax(-1) == p_spec.argmax(-1)).mean()),
               "train_minutes": train_minutes,
               "train_subset_frac": args.train_subset_frac,
               "n_train": len(split["train"]["files"]),
               "epochs_per_stage": args.epochs_per_stage, "image_sizes": args.image_sizes}
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("  saved " + metrics_path.name)
    return payload


def aggregate(results, out_dir, args):
    if not results:
        return
    names = list(results[0]["variants"].keys())
    summary = {}
    print("")
    print("=== {}-fold summary (tag={}) ===".format(len(results), args.tag))
    hdr = "{:<24} {:>19} {:>19} {:>9}".format("variant", "accuracy", "macro_f1", "ECE")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        accs = np.array([r["variants"][name]["accuracy"] for r in results])
        f1s = np.array([r["variants"][name]["macro_f1"] for r in results])
        eces = np.array([r["variants"][name]["ece"] for r in results])
        acc_sd = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0
        f1_sd = float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0
        summary[name] = {"accuracy_mean": float(accs.mean()), "accuracy_std": acc_sd,
                         "macro_f1_mean": float(f1s.mean()), "macro_f1_std": f1_sd,
                         "ece_mean": float(eces.mean()),
                         "per_fold_accuracy": [float(a) for a in accs]}
        print("{:<24} {:>10.4f} +- {:.4f} {:>10.4f} +- {:.4f} {:>9.4f}".format(
            name, accs.mean(), acc_sd, f1s.mean(), f1_sd, eces.mean()))

    path = out_dir / "summary.json"
    with open(path, "w") as f:
        json.dump({"tag": args.tag, "spectral_branch_mode": args.spectral_branch_mode,
                   "n_folds": len(results), "variants": summary}, f, indent=2)
    print("")
    print("Saved " + str(path))


def main():
    args = parse_args()
    if len(args.epochs_per_stage) != len(args.image_sizes):
        raise ValueError("--epochs-per-stage and --image-sizes must have the same length")
    if not TIF_ROOT.exists():
        raise FileNotFoundError("EuroSAT-MS tif folder not found at:\n  {}".format(TIF_ROOT))

    folds_path = PROJECT_ROOT / "data" / "eurosat_folds_k{}.json".format(args.folds)
    folds = build_or_load_kfolds(TIF_ROOT, folds_path, k=args.folds)
    print("folds: {} ({} folds over {} files)".format(folds_path.name, folds["k"], len(folds["files"])))

    out_dir = PROJECT_ROOT / "results" / "cv" / args.tag
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "cv" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))
    if args._config_keys_not_consumed:
        print("note: config.yaml keys no runner consumes: " + ", ".join(args._config_keys_not_consumed))
    print("resolved config -> " + str(dump_resolved_config(args, out_dir, {"device": str(device)})))

    fold_indices = [args.only_fold] if args.only_fold is not None else list(range(args.folds))
    for i in fold_indices:
        run_fold(i, folds, args, device, out_dir, ckpt_dir)

    # aggregate over whatever folds exist on disk, so a resumed sweep still summarises fully
    all_results = []
    for i in range(args.folds):
        fp = out_dir / "fold_{}.json".format(i)
        if fp.exists():
            with open(fp) as f:
                all_results.append(json.load(f))
    aggregate(all_results, out_dir, args)


if __name__ == "__main__":
    main()
