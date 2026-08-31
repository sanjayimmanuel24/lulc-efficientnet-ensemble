"""
Reliability diagrams for C4 (BLUEPRINT Section 12 / calibration reporting).

The CV runs saved per-sample PREDICTIONS but not probabilities, so calibration
could only ever be reported as a single ECE number with no figure behind it.
This re-runs evaluation from the saved checkpoints (no retraining) to recover
probabilities, then draws reliability diagrams.

Two things it makes visible that a scalar ECE hides:
  * whether the model is over- or under-confident, and in which confidence range;
  * that both fused variants are WORSE calibrated than either single branch in
    absolute terms, which is the honest caveat on C4's claim.

Equal-mass bins are used for the diagram, matching
adaptive_expected_calibration_error: at ~99% accuracy nearly every sample lands
in the top equal-width bin, so an equal-width diagram is one tall bar and nine
empty ones.

    python scripts/make_calibration_figures.py --tag default
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.calibration.temperature_scaling import (
    adaptive_expected_calibration_error, brier_score, expected_calibration_error,
    fit_temperature, softmax,
)
from src.data.dataset import EuroSATMSDataset, spectral_in_chans
from src.data.splits import build_or_load_kfolds, materialize_fold, CLASS_NAMES
from src.data.transforms import EuroSATTransform
from src.models.ensemble import DualBranchEfficientNet
from src.models.fusion import confidence_weighted_fusion
from src.training.train import TrainConfig, evaluate
from src.losses.label_smoothing_torch import build_criterion

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)
OUT_DIR = PROJECT_ROOT / "results" / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--spectral-branch-mode", default="rgb_plus_indices")
    p.add_argument("--attention", choices=["eca", "none"], default="eca")
    p.add_argument("--n-bins", type=int, default=12)
    return p.parse_args()


def loader(split_data, args, mode):
    ds = EuroSATMSDataset(split_data["files"], split_data["labels"], indices=("ndvi",),
                          transform=EuroSATTransform(args.image_size, train=False),
                          spectral_branch_mode=mode)
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)


def equal_mass_bins(conf, acc, n_bins):
    """Returns (bin_confidence, bin_accuracy, bin_weight) for a reliability diagram."""
    edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
    xs, ys, ws = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(acc[m].mean())
        ws.append(m.mean())
    return np.array(xs), np.array(ys), np.array(ws)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = build_or_load_kfolds(TIF_ROOT, PROJECT_ROOT / "data" / "eurosat_folds_k5.json", k=args.folds)
    chans = spectral_in_chans(args.spectral_branch_mode, 1)
    cfg = TrainConfig(num_classes=len(CLASS_NAMES), image_sizes=(args.image_size,),
                      batch_size=args.batch_size)
    criterion = build_criterion(cfg.label_smoothing)

    pooled = {k: [] for k in ("y", "rgb", "spectral", "avg", "c1", "rgb_uncal", "c1_uncal")}
    for fold in range(args.folds):
        split = materialize_fold(folds, fold)
        model = DualBranchEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False,
                                        spectral_in_chans=chans, attention=args.attention).to(device)
        ckpt = PROJECT_ROOT / "checkpoints" / "cv" / args.tag / "fold_{}_best.pt".format(fold)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        val = evaluate(model, loader(split["val"], args, args.spectral_branch_mode), criterion, cfg, device)
        t_rgb = fit_temperature(val["logits_rgb"].numpy(), val["labels"].numpy())
        t_spec = fit_temperature(val["logits_spectral"].numpy(), val["labels"].numpy())

        test = evaluate(model, loader(split["test"], args, args.spectral_branch_mode), criterion, cfg, device)
        y = test["labels"].numpy()
        lr, ls = test["logits_rgb"].numpy(), test["logits_spectral"].numpy()
        p_rgb, p_spec = softmax(lr, t_rgb), softmax(ls, t_spec)
        p_fused, _, _ = confidence_weighted_fusion(p_rgb, p_spec)
        # uncalibrated (T=1) counterparts, to show what temperature scaling actually did
        pu_rgb, pu_spec = softmax(lr, 1.0), softmax(ls, 1.0)
        pu_fused, _, _ = confidence_weighted_fusion(pu_rgb, pu_spec)

        pooled["y"].append(y)
        pooled["rgb"].append(p_rgb)
        pooled["spectral"].append(p_spec)
        pooled["avg"].append(0.5 * p_rgb + 0.5 * p_spec)
        pooled["c1"].append(p_fused)
        pooled["rgb_uncal"].append(pu_rgb)
        pooled["c1_uncal"].append(pu_fused)
        print("fold {}: T_rgb={:.3f} T_spec={:.3f}".format(fold, t_rgb, t_spec))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y = np.concatenate(pooled["y"])
    probs = {k: np.concatenate(v) for k, v in pooled.items() if k != "y"}

    np.savez_compressed(PROJECT_ROOT / "results" / "cv" / args.tag / "pooled_test_probs.npz",
                        y=y, **probs)

    stats = {}
    print("")
    print("{:<16} {:>9} {:>12} {:>9} {:>10}".format("variant", "ECE", "ECE(adapt)", "Brier", "mean conf"))
    print("-" * 60)
    for name in ("rgb", "spectral", "avg", "c1", "rgb_uncal", "c1_uncal"):
        p = probs[name]
        stats[name] = {"ece": float(expected_calibration_error(p, y)),
                       "ece_adaptive": float(adaptive_expected_calibration_error(p, y)),
                       "brier": float(brier_score(p, y)),
                       "mean_confidence": float(p.max(-1).mean()),
                       "accuracy": float((p.argmax(-1) == y).mean())}
        s = stats[name]
        print("{:<16} {:>9.4f} {:>12.4f} {:>9.4f} {:>10.4f}".format(
            name, s["ece"], s["ece_adaptive"], s["brier"], s["mean_confidence"]))

    # ---- figure: reliability diagrams ----
    panels = [("rgb", "RGB branch"), ("spectral", "Spectral branch"),
              ("avg", "Fixed 50/50"), ("c1", "C1 fusion"),
              ("c1_uncal", "C1 fusion (uncalibrated)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        p = probs[key]
        conf = p.max(-1)
        acc = (p.argmax(-1) == y).astype(float)
        xs, ys, ws = equal_mass_bins(conf, acc, args.n_bins)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(xs, ys, "o-", ms=4, lw=1.5)
        for x, yy in zip(xs, ys):
            ax.vlines(x, min(x, yy), max(x, yy), colors="tab:red", alpha=0.4, lw=1)
        lo = min(xs.min(), ys.min()) - 0.02
        ax.set_xlim(lo, 1.002)
        ax.set_ylim(lo, 1.002)
        ax.set_title("{}\nECE {:.4f} / adapt {:.4f}".format(
            title, stats[key]["ece"], stats[key]["ece_adaptive"]), fontsize=9)
        ax.set_xlabel("confidence", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("accuracy", fontsize=8)
    fig.suptitle("Reliability diagrams, equal-mass bins, pooled over {} folds (tag={})".format(
        args.folds, args.tag), fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "reliability_{}.png".format(args.tag)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("")
    print("saved " + str(out))

    with open(PROJECT_ROOT / "results" / "cv" / args.tag / "calibration_pooled.json", "w") as f:
        json.dump({"tag": args.tag, "n": int(len(y)), "n_bins": args.n_bins, "variants": stats}, f, indent=2)


if __name__ == "__main__":
    main()
