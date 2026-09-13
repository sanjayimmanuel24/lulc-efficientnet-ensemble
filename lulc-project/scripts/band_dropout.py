"""
Band-dropout robustness: what happens when the NIR band fails?

Operational motivation: NDVI needs a real near-infrared band. In deployment that
band can be unavailable -- sensor fault, heavy cloud, a cheaper 3-band payload,
or a scene where B08 is saturated. A model whose only input is RGB+NDVI stacked
into one tensor has nothing to fall back on. A DUAL-branch model does: branch A
never saw NDVI at all.

This corrupts ONLY the spectral branch's index channel at evaluation time. The
RGB branch input is untouched, which is the realistic failure mode (RGB comes
from different bands than NIR).

The second question is the interesting one. On clean EuroSAT the two branches are
equally reliable, so confidence-weighted fusion (C1) has nothing to arbitrate and
provably reduces to averaging (mean weight 0.4995 -- see docs/C1_FINDINGS.md).
Band dropout breaks that symmetry deliberately: one branch is degraded, the other
is not. If entropy-based confidence can detect ANYTHING, it should detect this and
shift weight toward RGB. If it cannot even here, the mechanism is dead in general.

    python scripts/band_dropout.py --tag augmented
    python scripts/band_dropout.py --tag augmented --only-fold 0   # quick look
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

from src.calibration.temperature_scaling import softmax
from src.data.dataset import EuroSATMSDataset, spectral_in_chans
from src.data.splits import (build_or_load_kfolds, default_folds_path,
                             materialize_fold, CLASS_NAMES)
from src.data.transforms import EuroSATTransform
from src.models.ensemble import DualBranchEfficientNet
from src.models.fusion import confidence_weighted_fusion

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)
OUT_FIG = PROJECT_ROOT / "results" / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="augmented")
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None)
    p.add_argument("--severities", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--corruption", choices=["noise", "zero", "shuffle"], default="noise",
                   help="noise: blend the index channel toward gaussian noise; "
                        "zero: blank it (band absent); "
                        "shuffle: keep the marginal distribution but destroy correspondence")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--attention", choices=["eca", "none"], default="eca")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def corrupt_spectral(spectral, severity, kind, gen):
    """
    Degrade the index channel(s) of the spectral branch input.

    Channels 0-2 of this tensor are RGB and are left alone: only the NIR-derived
    index is damaged, which is what a NIR sensor failure actually does.
    """
    if severity <= 0.0:
        return spectral
    x = spectral.clone()
    idx = x[:, 3:]
    if kind == "zero":
        x[:, 3:] = idx * (1.0 - severity)
    elif kind == "shuffle":
        perm = torch.randperm(idx.shape[0], generator=gen, device=idx.device)
        keep = (torch.rand(idx.shape[0], 1, 1, 1, generator=gen, device=idx.device) >= severity)
        x[:, 3:] = torch.where(keep, idx, idx[perm])
    else:  # noise
        noise = torch.randn(idx.shape, generator=gen, device=idx.device) * idx.std().clamp_min(1e-6)
        x[:, 3:] = (1.0 - severity) * idx + severity * noise
    return x


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chans = spectral_in_chans("rgb_plus_indices", 1)
    folds = build_or_load_kfolds(TIF_ROOT, default_folds_path(args.folds), k=args.folds)
    fold_ids = [args.only_fold] if args.only_fold is not None else list(range(args.folds))

    # per-fold temperatures fitted on clean data -- calibration is NOT refitted under
    # corruption, because in deployment you cannot recalibrate on a broken sensor.
    temps = {}
    for f in fold_ids:
        m = json.loads((PROJECT_ROOT / "results" / "cv" / args.tag / "fold_{}.json".format(f)).read_text())
        temps[f] = (m["t_rgb"], m["t_spectral"])

    variants = ["rgb_branch_only", "spectral_branch_only", "fixed_average_50_50", "c1_confidence_fusion"]
    acc = {v: {s: [] for s in args.severities} for v in variants}
    wrgb = {s: [] for s in args.severities}

    for f in fold_ids:
        split = materialize_fold(folds, f)["test"]
        model = DualBranchEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False,
                                        spectral_in_chans=chans, attention=args.attention).to(device)
        model.load_state_dict(torch.load(
            PROJECT_ROOT / "checkpoints" / "cv" / args.tag / "fold_{}_best.pt".format(f),
            map_location=device))
        model.eval()
        t_rgb, t_spec = temps[f]

        ds = EuroSATMSDataset(split["files"], split["labels"], indices=("ndvi",),
                              transform=EuroSATTransform(args.image_size, train=False),
                              spectral_branch_mode="rgb_plus_indices")
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

        for sev in args.severities:
            gen = torch.Generator(device=device).manual_seed(args.seed + f)
            lr_all, ls_all, y_all = [], [], []
            with torch.no_grad():
                for batch in loader:
                    rgb = batch["rgb"].to(device)
                    spec = corrupt_spectral(batch["spectral"].to(device), sev, args.corruption, gen)
                    out = model(rgb, spec)
                    lr_all.append(out["logits_rgb"].cpu())
                    ls_all.append(out["logits_spectral"].cpu())
                    y_all.append(batch["label"])
            y = torch.cat(y_all).numpy()
            p_rgb = softmax(torch.cat(lr_all).numpy(), t_rgb)
            p_spec = softmax(torch.cat(ls_all).numpy(), t_spec)
            p_fused, w, _ = confidence_weighted_fusion(p_rgb, p_spec)
            p_avg = 0.5 * p_rgb + 0.5 * p_spec

            acc["rgb_branch_only"][sev].append(float((p_rgb.argmax(-1) == y).mean()))
            acc["spectral_branch_only"][sev].append(float((p_spec.argmax(-1) == y).mean()))
            acc["fixed_average_50_50"][sev].append(float((p_avg.argmax(-1) == y).mean()))
            acc["c1_confidence_fusion"][sev].append(float((p_fused.argmax(-1) == y).mean()))
            wrgb[sev].append(float(w.mean()))
            print("fold {} severity {:.2f}: rgb={:.4f} spec={:.4f} avg={:.4f} c1={:.4f} w_rgb={:.4f}".format(
                f, sev, acc["rgb_branch_only"][sev][-1], acc["spectral_branch_only"][sev][-1],
                acc["fixed_average_50_50"][sev][-1], acc["c1_confidence_fusion"][sev][-1], w.mean()))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("")
    print("Band dropout ({} corruption of the NDVI channel), tag={}, {} folds".format(
        args.corruption, args.tag, len(fold_ids)))
    hdr = "{:<24}".format("variant") + "".join(" {:>16}".format("sev {:.2f}".format(s))
                                                for s in args.severities)
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for v in variants:
        line = "{:<24}".format(v)
        summary[v] = {}
        for s in args.severities:
            a = np.array(acc[v][s])
            sd = a.std(ddof=1) if len(a) > 1 else 0.0
            summary[v][str(s)] = {"mean": float(a.mean()), "std": float(sd),
                                  "per_fold": [float(x) for x in a]}
            line += " {:>9.4f}+-{:.4f}".format(a.mean(), sd)
        print(line)

    line = "{:<24}".format("mean w_rgb")
    for s in args.severities:
        line += " {:>16.4f}".format(np.mean(wrgb[s]))
    print(line)

    # The decisive comparison: does C1 beat the fixed average once the branches are
    # genuinely unequal? On clean data it provably cannot (weights sit at 0.5).
    print("")
    print("C1 minus fixed-average accuracy, by severity:")
    from src.evaluation.statistical_tests import paired_ttest
    deltas = {}
    for s in args.severities:
        a1 = np.array(acc["c1_confidence_fusion"][s])
        a0 = np.array(acc["fixed_average_50_50"][s])
        d = a1 - a0
        p = paired_ttest(a1, a0)[1] if len(a1) > 1 else float("nan")
        deltas[str(s)] = {"delta": float(d.mean()), "paired_t_p": float(p),
                          "folds_better": int((d > 0).sum())}
        print("  severity {:.2f}: {:+.4f}  ({}/{} folds, paired-t p={:.4f})".format(
            s, d.mean(), int((d > 0).sum()), len(d), p))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in variants:
        m = [np.mean(acc[v][s]) for s in args.severities]
        sd = [np.std(acc[v][s], ddof=1) if len(acc[v][s]) > 1 else 0.0 for s in args.severities]
        axes[0].errorbar(args.severities, m, yerr=sd, marker="o", ms=4, capsize=3, lw=1.6, label=v)
    axes[0].set_xlabel("NDVI channel corruption severity")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Degradation when the NIR-derived band fails")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(args.severities, [np.mean(wrgb[s]) for s in args.severities], "o-", lw=1.8)
    axes[1].axhline(0.5, color="k", ls="--", lw=1, label="0.5 = no preference")
    axes[1].set_xlabel("NDVI channel corruption severity")
    axes[1].set_ylabel("mean fusion weight on the RGB branch")
    axes[1].set_title("Does confidence weighting notice?")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Band-dropout robustness (tag={}, {} corruption)".format(args.tag, args.corruption))
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "band_dropout_{}_{}.png".format(args.tag, args.corruption)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("")
    print("saved " + str(out))

    res = PROJECT_ROOT / "results" / "cv" / args.tag / "band_dropout_{}.json".format(args.corruption)
    with open(res, "w") as f:
        json.dump({"tag": args.tag, "corruption": args.corruption, "folds": fold_ids,
                   "severities": args.severities, "accuracy": summary,
                   "mean_weight_rgb": {str(s): float(np.mean(wrgb[s])) for s in args.severities},
                   "c1_minus_average": deltas}, f, indent=2)
    print("saved " + str(res))


if __name__ == "__main__":
    main()
