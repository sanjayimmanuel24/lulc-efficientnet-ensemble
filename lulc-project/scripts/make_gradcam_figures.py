"""
Grad-CAM figures for docs/BLUEPRINT.md Section 12.

The Grad-CAM code has been implemented and unit-tested since the start of the
project but had never been run on real imagery, so Section 12 had no figures.
This produces them from the trained CV checkpoints -- no retraining needed.

Per the module docstring in src/explainability/gradcam.py, CAMs are produced
per BRANCH rather than on the fused output: the point is to show what each
modality attends to, which a single fused heatmap would hide.

Two figure sets are produced:
  * correct cases  -- one per class, both branches agreeing
  * disagreement cases -- where the branches predicted DIFFERENT classes, which
    is where fusion actually does its work (76% of available headroom, per
    scripts/analyze_errors.py). These are the interesting figures: they show
    the two modalities attending to different evidence on the same patch.

    python scripts/make_gradcam_figures.py --tag default --fold 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")           # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import EuroSATMSDataset, spectral_in_chans
from src.data.splits import build_or_load_kfolds, materialize_fold, CLASS_NAMES
from src.data.transforms import EuroSATTransform
from src.explainability.gradcam import GradCAM
from src.models.ensemble import DualBranchEfficientNet

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)
OUT_DIR = PROJECT_ROOT / "results" / "figures"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--spectral-branch-mode", default="rgb_plus_indices")
    p.add_argument("--attention", choices=["eca", "none"], default="eca")
    p.add_argument("--max-disagreements", type=int, default=6)
    return p.parse_args()


def denorm_rgb(t, low=2.0, high=98.0):
    """
    Undo ImageNet normalisation, then percentile-stretch for display.

    Sentinel-2 surface reflectance for land cover sits around 0.05-0.20, so the
    raw values render almost black. The stretch is a DISPLAY transform only --
    the model is fed the normalised tensor, not this. Without it the figures are
    unusable in print.
    """
    x = t.cpu().numpy()
    x = x * _IMAGENET_STD + _IMAGENET_MEAN
    x = x.transpose(1, 2, 0)
    lo, hi = np.percentile(x, low), np.percentile(x, high)
    if hi - lo < 1e-6:
        return np.clip(x, 0, 1)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay(ax, rgb, cam, title):
    ax.imshow(rgb)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def branch_cam(model, branch, target_layer, x, class_idx):
    cam = GradCAM(branch, target_layer)
    try:
        heat = cam(x.unsqueeze(0).clone().requires_grad_(True), class_idx=class_idx,
                   forward_fn=lambda t: branch(t))
        return heat.detach().cpu().numpy()
    finally:
        cam.remove_hooks()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folds = build_or_load_kfolds(TIF_ROOT, PROJECT_ROOT / "data" / "eurosat_folds_k5.json", k=5)
    split = materialize_fold(folds, args.fold)
    chans = spectral_in_chans(args.spectral_branch_mode, 1)

    model = DualBranchEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False,
                                    spectral_in_chans=chans, attention=args.attention).to(device)
    ckpt = PROJECT_ROOT / "checkpoints" / "cv" / args.tag / "fold_{}_best.pt".format(args.fold)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print("loaded " + str(ckpt))

    ds = EuroSATMSDataset(split["test"]["files"], split["test"]["labels"], indices=("ndvi",),
                          transform=EuroSATTransform(args.image_size, train=False),
                          spectral_branch_mode=args.spectral_branch_mode)

    # attention module is the post-backbone feature map in each branch (see test_models_torch.py)
    layer_rgb = model.branch_rgb.attention
    layer_spec = model.branch_spectral.attention

    # ---- pass 1: predictions for the whole fold, to pick interesting samples ----
    preds_r, preds_s, labels = [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=2)
    with torch.no_grad():
        for batch in loader:
            out = model(batch["rgb"].to(device), batch["spectral"].to(device))
            preds_r.append(out["logits_rgb"].argmax(-1).cpu())
            preds_s.append(out["logits_spectral"].argmax(-1).cpu())
            labels.append(batch["label"])
    preds_r = torch.cat(preds_r).numpy()
    preds_s = torch.cat(preds_s).numpy()
    labels = torch.cat(labels).numpy()
    print("fold {}: {} test samples, branches disagree on {}".format(
        args.fold, len(labels), int((preds_r != preds_s).sum())))

    # ---- figure 1: one correct example per class, both branches ----
    fig, axes = plt.subplots(3, len(CLASS_NAMES), figsize=(2.0 * len(CLASS_NAMES), 6.4))
    for c, name in enumerate(CLASS_NAMES):
        idx = np.where((labels == c) & (preds_r == c) & (preds_s == c))[0]
        if len(idx) == 0:
            for r in range(3):
                axes[r, c].axis("off")
            continue
        i = int(idx[0])
        s = ds[i]
        rgb_img = denorm_rgb(s["rgb"])
        cam_r = branch_cam(model, model.branch_rgb, layer_rgb, s["rgb"].to(device), c)
        cam_s = branch_cam(model, model.branch_spectral, layer_spec, s["spectral"].to(device), c)
        overlay(axes[0, c], rgb_img, None, name)
        overlay(axes[1, c], rgb_img, cam_r, "RGB branch")
        overlay(axes[2, c], rgb_img, cam_s, "spectral branch")
    fig.suptitle("Grad-CAM per branch, correctly classified examples (fold {})".format(args.fold),
                 fontsize=11)
    fig.tight_layout()
    p1 = OUT_DIR / "gradcam_per_class_fold{}.png".format(args.fold)
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved " + str(p1))

    # ---- figure 2: branch disagreements -- where fusion earns its keep ----
    # Spread across distinct true classes: the test file list is sorted by class,
    # so taking the first N disagreements would draw them all from one class.
    all_dis = np.where(preds_r != preds_s)[0]
    picked, seen = [], set()
    for i in all_dis:                       # first pass: one per unseen class
        c = int(labels[i])
        if c not in seen:
            seen.add(c)
            picked.append(i)
        if len(picked) >= args.max_disagreements:
            break
    for i in all_dis:                       # top up if fewer classes than slots
        if len(picked) >= args.max_disagreements:
            break
        if i not in picked:
            picked.append(i)
    dis = np.array(picked[:args.max_disagreements])
    if len(dis):
        print("  disagreement figure covers classes: " + ", ".join(
            sorted({CLASS_NAMES[int(labels[i])] for i in dis})))
    if len(dis):
        fig, axes = plt.subplots(3, len(dis), figsize=(2.2 * len(dis), 6.8), squeeze=False)
        for col, i in enumerate(dis):
            i = int(i)
            s = ds[i]
            rgb_img = denorm_rgb(s["rgb"])
            true_c, pr, ps = int(labels[i]), int(preds_r[i]), int(preds_s[i])
            cam_r = branch_cam(model, model.branch_rgb, layer_rgb, s["rgb"].to(device), pr)
            cam_s = branch_cam(model, model.branch_spectral, layer_spec, s["spectral"].to(device), ps)
            overlay(axes[0][col], rgb_img, None, "true: {}".format(CLASS_NAMES[true_c]))
            overlay(axes[1][col], rgb_img, cam_r,
                    "RGB -> {}{}".format(CLASS_NAMES[pr], " OK" if pr == true_c else " X"))
            overlay(axes[2][col], rgb_img, cam_s,
                    "spec -> {}{}".format(CLASS_NAMES[ps], " OK" if ps == true_c else " X"))
        fig.suptitle("Branch disagreements: the samples fusion has to arbitrate (fold {})".format(
            args.fold), fontsize=11)
        fig.tight_layout()
        p2 = OUT_DIR / "gradcam_disagreements_fold{}.png".format(args.fold)
        fig.savefig(p2, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("saved " + str(p2))


if __name__ == "__main__":
    main()
