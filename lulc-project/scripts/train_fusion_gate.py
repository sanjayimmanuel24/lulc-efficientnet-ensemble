"""
A learned fusion gate, trained to survive band dropout.

WHY THIS EXISTS
---------------
C1's entropy-weighted fusion does not work, and docs/BAND_DROPOUT.md shows why in
the cleanest possible setting: with one branch fed pure noise (14.5% accuracy,
chance is 10%), the fusion weight moves only 0.4982 -> 0.5973. It should approach
1.0. The fused model scores 0.9208 where the intact RGB branch alone scores
0.9864 -- 6.5pp left on the table.

The mechanism is specific, and it suggests the fix. Normalised predictive entropy
is computed from the SOFTMAX, which divides out logit magnitude. A network fed
noise still produces a peaked distribution, so it is confidently wrong and entropy
cannot see it. Logit MAGNITUDE is not scale-invariant: max-logit and logsumexp
(the energy score, Liu et al. 2020) collapse when a branch receives garbage.

So the gate is given magnitude features that entropy structurally cannot access,
and is trained with band-dropout augmentation so it learns to route around a
degraded branch.

DESIGN
------
* Features are CLASS-AGNOSTIC (confidence, entropy, margin, max-logit, logsumexp
  per branch, plus joint agreement terms). The gate never sees class identity, so
  it cannot memorise "NDVI helps for Forest" -- it can only learn to judge
  reliability, which is what should transfer.
* Trained per fold on that fold's VALIDATION logits only; evaluated on test.
* Training data includes corrupted copies of validation, which is the whole point:
  a gate trained only on clean data has never seen a broken branch.
* Backbones are frozen. This trains ~600 parameters on cached logits in seconds.

    python scripts/train_fusion_gate.py --tag augmented           # dump logits + train + eval
    python scripts/train_fusion_gate.py --tag augmented --cached  # reuse cached logits
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
import torch.nn as nn
from torch.utils.data import DataLoader

from src.calibration.temperature_scaling import softmax
from src.data.dataset import EuroSATMSDataset, spectral_in_chans
from src.data.splits import build_or_load_kfolds, materialize_fold, CLASS_NAMES
from src.data.transforms import EuroSATTransform
from src.evaluation.statistical_tests import paired_ttest
from src.models.ensemble import DualBranchEfficientNet
from src.models.fusion import confidence_weighted_fusion

PROJECT_ROOT = Path(__file__).parent.parent
TIF_ROOT = (
    PROJECT_ROOT / "data" / "eurosat" / "EuroSATallBands" / "ds" / "images"
    / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
)
OUT_FIG = PROJECT_ROOT / "results" / "figures"
_EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="augmented")
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("--train-severities", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                   help="corruption levels the GATE sees during training (on validation)")
    p.add_argument("--eval-severities", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--attention", choices=["eca", "none"], default="eca")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cached", action="store_true", help="reuse cached logits, skip all GPU work")
    p.add_argument("--gate-split-frac", type=float, default=0.3,
                   help="Fraction of VALIDATION held out from gate training, used only to select "
                        "the gate's stopping point. 0 disables (the original fixed-epoch fit).")
    return p.parse_args()


# ---------------------------------------------------------------- corruption
def corrupt_spectral(spectral, severity, gen):
    """Blend the index channel(s) toward gaussian noise. RGB channels untouched."""
    if severity <= 0.0:
        return spectral
    x = spectral.clone()
    idx = x[:, 3:]
    noise = torch.randn(idx.shape, generator=gen, device=idx.device) * idx.std().clamp_min(1e-6)
    x[:, 3:] = (1.0 - severity) * idx + severity * noise
    return x


# ---------------------------------------------------------------- features
def gate_features(lr, ls):
    """
    Class-agnostic reliability features from the two branches' LOGITS.

    The magnitude terms (max logit, logsumexp) are the point: softmax-based entropy
    divides them out, which is exactly why C1 cannot detect a corrupted branch.
    """
    def per_branch(lg):
        p = softmax(lg, 1.0)
        srt = np.sort(p, axis=-1)
        ent = -(p * np.log(np.clip(p, _EPS, 1.0))).sum(-1) / np.log(p.shape[-1])
        return np.stack([
            p.max(-1),                                   # confidence
            1.0 - ent,                                   # entropy-based confidence (what C1 uses)
            srt[:, -1] - srt[:, -2],                     # top-1 vs top-2 margin
            lg.max(-1),                                  # max logit  <- magnitude, invisible to entropy
            np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1),   # logsumexp (energy)
            lg.std(-1),                                  # logit spread
        ], axis=-1)

    a, b = per_branch(lr), per_branch(ls)
    pa, pb = softmax(lr, 1.0), softmax(ls, 1.0)
    tv = 0.5 * np.abs(pa - pb).sum(-1)
    agree = (pa.argmax(-1) == pb.argmax(-1)).astype(np.float32)
    joint = np.stack([tv, agree, np.abs(a[:, 1] - b[:, 1]), a[:, 4] - b[:, 4]], axis=-1)
    return np.concatenate([a, b, joint], axis=-1).astype(np.float32)


class FusionGate(nn.Module):
    """Tiny MLP -> scalar weight in [0,1] for the RGB branch."""

    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, f):
        return torch.sigmoid(self.net(f)).squeeze(-1)


# ---------------------------------------------------------------- logit cache
def dump_logits(args, device):
    """Forward passes are the only expensive part; cache them so the gate can be re-trained free."""
    cache_dir = PROJECT_ROOT / "results" / "cv" / args.tag / "gate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chans = spectral_in_chans("rgb_plus_indices", 1)
    folds = build_or_load_kfolds(TIF_ROOT, PROJECT_ROOT / "data" / "eurosat_folds_k5.json", k=args.folds)
    severities = sorted(set(args.train_severities) | set(args.eval_severities))

    for f in range(args.folds):
        out = cache_dir / "fold_{}.npz".format(f)
        if out.exists():
            print("fold {}: cached".format(f))
            continue
        split = materialize_fold(folds, f)
        model = DualBranchEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False,
                                        spectral_in_chans=chans, attention=args.attention).to(device)
        model.load_state_dict(torch.load(
            PROJECT_ROOT / "checkpoints" / "cv" / args.tag / "fold_{}_best.pt".format(f),
            map_location=device))
        model.eval()

        store = {}
        for part in ("val", "test"):
            ds = EuroSATMSDataset(split[part]["files"], split[part]["labels"], indices=("ndvi",),
                                  transform=EuroSATTransform(args.image_size, train=False),
                                  spectral_branch_mode="rgb_plus_indices")
            loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
            for sev in severities:
                gen = torch.Generator(device=device).manual_seed(args.seed + f)
                LR, LS, Y = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        rgb = batch["rgb"].to(device)
                        spec = corrupt_spectral(batch["spectral"].to(device), sev, gen)
                        o = model(rgb, spec)
                        LR.append(o["logits_rgb"].cpu().numpy())
                        LS.append(o["logits_spectral"].cpu().numpy())
                        Y.append(batch["label"].numpy())
                key = "{}_{:.2f}".format(part, sev)
                store[key + "_lr"] = np.concatenate(LR)
                store[key + "_ls"] = np.concatenate(LS)
                store[key + "_y"] = np.concatenate(Y)
                print("  fold {} {} sev {:.2f}: {} samples".format(f, part, sev, len(store[key + "_y"])))
        np.savez_compressed(out, **store)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cache_dir


# ---------------------------------------------------------------- train + eval
def fit_gate(feats, lr_logits, ls_logits, y, args, rng):
    """
    Fit the gate on a TRAIN portion of validation,selects its stopping point on a
    disjoint HELD-OUT portion. Without the held-out part the gate runs a fixed
    number of epochs with no way to notice it is overfitting -- and an overfitted
    gate would still look fine on its own training data.

    The split is by sample index and is shared across corruption severities, so a
    tile never appears in both halves under different corruptions.
    """
    torch.manual_seed(args.seed)
    f = torch.from_numpy(feats)
    pa = torch.from_numpy(softmax(lr_logits, 1.0))
    pb = torch.from_numpy(softmax(ls_logits, 1.0))
    t = torch.from_numpy(y).long()

    n = len(y)
    if args.gate_split_frac > 0:
        n_hold = int(round(args.gate_split_frac * n))
        perm = rng.permutation(n)
        hold_idx, tr_idx = perm[:n_hold], perm[n_hold:]
    else:
        tr_idx = np.arange(n)
        hold_idx = np.arange(0)

    tr = torch.from_numpy(tr_idx)
    mu, sd = f[tr].mean(0, keepdim=True), f[tr].std(0, keepdim=True).clamp_min(1e-6)
    fn = (f - mu) / sd

    gate = FusionGate(feats.shape[1], args.hidden)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr, weight_decay=1e-4)

    def loss_on(idx):
        w = gate(fn[idx]).unsqueeze(-1)
        fused = w * pa[idx] + (1 - w) * pb[idx]
        return nn.functional.nll_loss(torch.log(fused.clamp_min(_EPS)), t[idx])

    best, best_state, best_epoch = float("inf"), None, 0
    ho = torch.from_numpy(hold_idx) if len(hold_idx) else None
    for ep in range(args.epochs):
        gate.train()
        opt.zero_grad(set_to_none=True)
        loss_on(tr).backward()
        opt.step()
        if ho is not None and (ep + 1) % 5 == 0:
            gate.eval()
            with torch.no_grad():
                hl = float(loss_on(ho))
            if hl < best:
                best, best_epoch = hl, ep + 1
                best_state = {k: v.clone() for k, v in gate.state_dict().items()}
    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    return gate, mu, sd, best_epoch, best


def apply_gate(gate, mu, sd, feats, lr_logits, ls_logits):
    with torch.no_grad():
        fn = (torch.from_numpy(feats) - mu) / sd
        w = gate(fn).numpy()
    pa, pb = softmax(lr_logits, 1.0), softmax(ls_logits, 1.0)
    fused = w[:, None] * pa + (1 - w[:, None]) * pb
    return fused, w


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = PROJECT_ROOT / "results" / "cv" / args.tag / "gate_cache"
    if not args.cached:
        cache_dir = dump_logits(args, device)

    methods = ["fixed_average_50_50", "c1_entropy_fusion", "learned_gate", "rgb_branch_only"]
    acc = {m: {s: [] for s in args.eval_severities} for m in methods}
    gate_w = {s: [] for s in args.eval_severities}

    for f in range(args.folds):
        d = np.load(cache_dir / "fold_{}.npz".format(f))
        m = json.loads((PROJECT_ROOT / "results" / "cv" / args.tag /
                        "fold_{}.json".format(f)).read_text())
        t_rgb, t_spec = m["t_rgb"], m["t_spectral"]

        # --- train the gate on validation only, pooled over the training severities ---
        F, LRs, LSs, Ys = [], [], [], []
        for sev in args.train_severities:
            k = "val_{:.2f}".format(sev)
            lr, ls, y = d[k + "_lr"], d[k + "_ls"], d[k + "_y"]
            F.append(gate_features(lr, ls)); LRs.append(lr); LSs.append(ls); Ys.append(y)
        rng = np.random.default_rng(args.seed + f)
        gate, mu, sd, best_ep, best_hl = fit_gate(np.concatenate(F), np.concatenate(LRs),
                                                   np.concatenate(LSs), np.concatenate(Ys), args, rng)

        # --- evaluate on test at every severity ---
        for sev in args.eval_severities:
            k = "test_{:.2f}".format(sev)
            lr, ls, y = d[k + "_lr"], d[k + "_ls"], d[k + "_y"]
            # Baselines get their FITTED temperatures (their best configuration, and what
            # every other script in this repo reports). The gate takes raw logits because
            # its magnitude features are exactly what temperature scaling would rescale.
            pa, pb = softmax(lr, t_rgb), softmax(ls, t_spec)
            p_c1, _, _ = confidence_weighted_fusion(pa, pb)
            p_gate, w = apply_gate(gate, mu, sd, gate_features(lr, ls), lr, ls)

            acc["fixed_average_50_50"][sev].append(float(((0.5 * pa + 0.5 * pb).argmax(-1) == y).mean()))
            acc["c1_entropy_fusion"][sev].append(float((p_c1.argmax(-1) == y).mean()))
            acc["learned_gate"][sev].append(float((p_gate.argmax(-1) == y).mean()))
            acc["rgb_branch_only"][sev].append(float((pa.argmax(-1) == y).mean()))
            gate_w[sev].append(float(w.mean()))
        print("fold {}: gate trained ({} params), stopped at epoch {} (held-out loss {:.4f})".format(
            f, sum(p.numel() for p in gate.parameters()), best_ep, best_hl))

    print("")
    print("Learned fusion gate vs entropy fusion, tag={}, k={}".format(args.tag, args.folds))
    hdr = "{:<24}".format("method") + "".join(" {:>16}".format("sev {:.2f}".format(s))
                                               for s in args.eval_severities)
    print(hdr); print("-" * len(hdr))
    summary = {}
    for m in methods:
        line = "{:<24}".format(m)
        summary[m] = {}
        for s in args.eval_severities:
            a = np.array(acc[m][s])
            summary[m][str(s)] = {"mean": float(a.mean()),
                                  "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                                  "per_fold": [float(x) for x in a]}
            line += " {:>9.4f}+-{:.4f}".format(a.mean(), summary[m][str(s)]["std"])
        print(line)
    line = "{:<24}".format("mean gate weight (rgb)")
    for s in args.eval_severities:
        line += " {:>16.4f}".format(np.mean(gate_w[s]))
    print(line)

    print("")
    print("learned gate minus entropy fusion:")
    deltas = {}
    for s in args.eval_severities:
        a1, a0 = np.array(acc["learned_gate"][s]), np.array(acc["c1_entropy_fusion"][s])
        dd = a1 - a0
        p = paired_ttest(a1, a0)[1] if len(a1) > 1 else float("nan")
        deltas[str(s)] = {"delta": float(dd.mean()), "paired_t_p": float(p),
                          "folds_better": int((dd > 0).sum())}
        print("  severity {:.2f}: {:+.4f}  ({}/{} folds, paired-t p={:.4f})".format(
            s, dd.mean(), int((dd > 0).sum()), len(dd), p))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in methods:
        mm = [np.mean(acc[m][s]) for s in args.eval_severities]
        sdv = [np.std(acc[m][s], ddof=1) if len(acc[m][s]) > 1 else 0.0 for s in args.eval_severities]
        axes[0].errorbar(args.eval_severities, mm, yerr=sdv, marker="o", ms=4, capsize=3, lw=1.6, label=m)
    axes[0].set_xlabel("NDVI channel corruption severity")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Learned gate vs entropy fusion under band dropout")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].plot(args.eval_severities, [np.mean(gate_w[s]) for s in args.eval_severities],
                 "o-", lw=1.8, label="learned gate")
    axes[1].axhline(0.5, color="k", ls="--", lw=1, label="0.5 = no preference")
    axes[1].axhline(1.0, color="g", ls=":", lw=1, label="1.0 = ignore spectral branch")
    axes[1].set_xlabel("NDVI channel corruption severity")
    axes[1].set_ylabel("mean weight on the RGB branch")
    axes[1].set_title("Does the gate route around the broken branch?")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.suptitle("Learned fusion gate (tag={})".format(args.tag))
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "fusion_gate_{}.png".format(args.tag)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("")
    print("saved " + str(out))

    res = PROJECT_ROOT / "results" / "cv" / args.tag / "fusion_gate.json"
    with open(res, "w") as f:
        json.dump({"tag": args.tag, "train_severities": args.train_severities,
                   "eval_severities": args.eval_severities, "accuracy": summary,
                   "mean_gate_weight_rgb": {str(s): float(np.mean(gate_w[s])) for s in args.eval_severities},
                   "gate_minus_entropy": deltas}, f, indent=2)
    print("saved " + str(res))


if __name__ == "__main__":
    main()
