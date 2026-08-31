"""
Live prediction demo: classify real Sentinel-2 tiles with the trained model.

Everything else in this repo does batch evaluation over folds. This is the
one-image path -- what you run to SHOW the model working, rather than to measure
it. It reports what makes this architecture interesting: not just the fused
prediction, but what each branch said independently and how the fusion weighted
them.

Runs in seconds on CPU, so it needs no GPU and no training.

    # 8 random held-out tiles, printed table + a figure
    python scripts/predict.py --random 8

    # one specific tile
    python scripts/predict.py --image data/eurosat/.../Forest/Forest_1.tif

    # force CPU (e.g. demoing on a laptop with no GPU)
    python scripts/predict.py --random 8 --device cpu

Samples are drawn from fold 0's TEST split -- images the model never trained on.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.dataset import EuroSATMSDataset, spectral_in_chans
from src.data.splits import build_or_load_kfolds, materialize_fold, CLASS_NAMES
from src.data.transforms import EuroSATTransform
from src.models.ensemble import DualBranchEfficientNet
from src.models.fusion import confidence_weighted_fusion

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)
FOLDS_PATH = PROJECT_ROOT / "data" / "eurosat_folds_k5.json"

GREEN, RED, DIM, BOLD, OFF = "\033[92m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"


def parse_args():
    p = argparse.ArgumentParser(description="Classify Sentinel-2 tiles with the trained model.")
    p.add_argument("--image", type=str, default=None, help="path to a single 13-band .tif")
    p.add_argument("--random", type=int, default=8, help="how many random held-out tiles to classify")
    p.add_argument("--tag", default="augmented", help="which trained run to load (default: best)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--spectral-branch-mode", default="rgb_plus_indices")
    p.add_argument("--attention", choices=["eca", "none"], default="eca")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=None, help="fix which random tiles are drawn")
    p.add_argument("--figure", action="store_true", help="also save a figure of the predictions")
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def load_model(args, device):
    chans = spectral_in_chans(args.spectral_branch_mode, 1)
    model = DualBranchEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False,
                                    spectral_in_chans=chans, attention=args.attention).to(device)
    ckpt = PROJECT_ROOT / "checkpoints" / "cv" / args.tag / "fold_{}_best.pt".format(args.fold)
    if not ckpt.exists():
        raise FileNotFoundError(
            "No checkpoint at {}\nTrain one first, or pass --tag/--fold for a run you have.".format(ckpt))
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # Checkpoints are written DURING training, before temperature scaling is fitted,
    # so their temperature buffers are still 1.0 and the model reads as ~90% confident
    # when it is really ~99% accurate (that underconfidence is the C4 finding). Restore
    # the fitted temperatures from the fold's metrics so the demo shows calibrated numbers.
    metrics = PROJECT_ROOT / "results" / "cv" / args.tag / "fold_{}.json".format(args.fold)
    calibrated = False
    if metrics.exists():
        m = json.loads(metrics.read_text())
        if "t_rgb" in m and "t_spectral" in m:
            model.set_temperatures(m["t_rgb"], m["t_spectral"])
            calibrated = True
    model.eval()
    return model, ckpt, calibrated


def main():
    args = parse_args()
    if args.no_color:
        global GREEN, RED, DIM, BOLD, OFF
        GREEN = RED = DIM = BOLD = OFF = ""

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    model, ckpt, calibrated = load_model(args, device)

    # Which tiles to classify: either the one asked for, or random held-out ones.
    if args.image:
        files = [args.image]
        labels = [CLASS_NAMES.index(Path(args.image).parent.name)
                  if Path(args.image).parent.name in CLASS_NAMES else -1]
    else:
        folds = build_or_load_kfolds(TIF_ROOT, FOLDS_PATH, k=5)
        split = materialize_fold(folds, args.fold)["test"]
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(split["files"]), size=min(args.random, len(split["files"])), replace=False)
        files = [split["files"][i] for i in idx]
        labels = [split["labels"][i] for i in idx]

    ds = EuroSATMSDataset(files, labels, indices=("ndvi",),
                          transform=EuroSATTransform(args.image_size, train=False),
                          spectral_branch_mode=args.spectral_branch_mode)

    print("")
    print("{}Dual-Branch EfficientNet-B0 - Sentinel-2 land cover classification{}".format(BOLD, OFF))
    print("{}checkpoint: {}/{}  |  device: {}  |  {} held-out tiles  |  {}{}".format(
        DIM, args.tag, ckpt.name, device, len(files),
        "temperature-calibrated" if calibrated else "UNCALIBRATED (no fold metrics found)", OFF))
    print("")
    hdr = "{:<22} {:<22} {:>7}   {:<20} {:<20} {:>7}".format(
        "true class", "PREDICTED", "conf", "rgb branch", "spectral branch", "w_rgb")
    print(hdr)
    print("-" * len(hdr))

    n_correct = 0
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        with torch.no_grad():
            out = model(s["rgb"].unsqueeze(0).to(device), s["spectral"].unsqueeze(0).to(device))
        p_rgb = torch.softmax(out["logits_rgb"] / model.temperature_rgb, -1).cpu().numpy()
        p_spec = torch.softmax(out["logits_spectral"] / model.temperature_spectral, -1).cpu().numpy()
        fused, w_rgb, _ = confidence_weighted_fusion(p_rgb, p_spec)

        pred = int(fused.argmax(-1)[0])
        conf = float(fused.max(-1)[0])
        pr, ps = int(p_rgb.argmax(-1)[0]), int(p_spec.argmax(-1)[0])
        true = labels[i]
        ok = (true == pred)
        n_correct += int(ok)
        rows.append((true, pred, conf, pr, ps, float(w_rgb[0]), files[i]))

        mark = GREEN if ok else RED
        agree = "" if pr == ps else "   <-- branches disagree, fusion arbitrates"
        print("{:<22} {}{:<22}{} {:>6.2f}%   {:<20} {:<20} {:>7.3f}{}".format(
            CLASS_NAMES[true] if true >= 0 else "(unknown)",
            mark, CLASS_NAMES[pred], OFF, conf * 100,
            CLASS_NAMES[pr], CLASS_NAMES[ps], float(w_rgb[0]), agree))

    if labels[0] >= 0:
        print("")
        print("{}{}/{} correct on this sample{}".format(
            BOLD, n_correct, len(ds), OFF))
        n_dis = sum(1 for r in rows if r[3] != r[4])
        print("{}branches disagreed on {} of {} tiles (fold-wide rate is ~1.4%){}".format(
            DIM, n_dis, len(rows), OFF))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        n = len(ds)
        cols = min(n, 4)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 3.4 * rows_n), squeeze=False)
        for i in range(rows_n * cols):
            ax = axes[i // cols][i % cols]
            ax.axis("off")
            if i >= n:
                continue
            x = ds[i]["rgb"].numpy() * std + mean
            x = x.transpose(1, 2, 0)
            lo, hi = np.percentile(x, 2), np.percentile(x, 98)
            ax.imshow(np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1))
            true, pred, conf = rows[i][0], rows[i][1], rows[i][2]
            ok = true == pred
            ax.set_title("{} {:.1f}%\n{}".format(
                CLASS_NAMES[pred], conf * 100,
                "correct" if ok else "true: " + CLASS_NAMES[true]),
                fontsize=9, color=("green" if ok else "red"))
        fig.suptitle("Predictions on held-out Sentinel-2 tiles", fontsize=12)
        fig.tight_layout()
        out = PROJECT_ROOT / "results" / "figures" / "predictions_demo.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("")
        print("saved " + str(out))


if __name__ == "__main__":
    main()
