"""
Per-class error analysis pooled across CV folds (BLUEPRINT Section 10's
error-analysis requirement).

Uses only the per-sample predictions each CV run already saved
(results/cv/<tag>/fold_*_preds.npz), so this needs no GPU and no retraining.

Answers the questions a reviewer asks about a 99%-accuracy result:
  * which classes actually fail, and are they the ones the method targets?
  * where does fusion help -- is it resolving branch disagreements, or just
    inheriting one branch's predictions?
  * does the spectral (NDVI) branch help on the vegetation classes it was
    introduced for, which is C2's whole premise?

    python scripts/analyze_errors.py --tag default
    python scripts/analyze_errors.py --tag default --compare baseline_resnet50
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.splits import CLASS_NAMES
from src.evaluation.metrics import compute_confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
CV_DIR = PROJECT_ROOT / "results" / "cv"

# Classes NDVI is supposed to disambiguate -- C2's stated motivation.
VEGETATION = {"AnnualCrop", "Forest", "HerbaceousVegetation", "Pasture", "PermanentCrop"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("--compare", default=None, help="second tag to compare per-class against")
    p.add_argument("-k", "--folds", type=int, default=5)
    return p.parse_args()


def load_pooled(tag, k):
    """Concatenate per-sample predictions over all folds (each sample tested exactly once)."""
    out = {key: [] for key in ("y", "pred_rgb", "pred_spectral", "pred_avg", "pred_c1", "weight_rgb")}
    for i in range(k):
        f = CV_DIR / tag / "fold_{}_preds.npz".format(i)
        if not f.exists():
            raise FileNotFoundError("missing {} -- run the sweep for tag '{}' first".format(f, tag))
        d = np.load(f)
        for key in out:
            out[key].append(d[key])
    return {key: np.concatenate(v) for key, v in out.items()}


def per_class_table(y, pred, title):
    print("")
    print(title)
    print("{:<24} {:>7} {:>8} {:>8} {:>8}".format("class", "n", "recall", "prec", "F1"))
    print("-" * 58)
    rows = {}
    cm = compute_confusion_matrix(y, pred, len(CLASS_NAMES))
    for c, name in enumerate(CLASS_NAMES):
        n = int(cm[c].sum())
        tp = int(cm[c, c])
        recall = tp / n if n else 0.0
        pred_c = int(cm[:, c].sum())
        prec = tp / pred_c if pred_c else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        rows[name] = {"n": n, "recall": recall, "precision": prec, "f1": f1}
        print("{:<24} {:>7} {:>8.4f} {:>8.4f} {:>8.4f}".format(name, n, recall, prec, f1))
    return rows, cm


def top_confusions(cm, k=6):
    print("")
    print("Most frequent confusions (true -> predicted):")
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), CLASS_NAMES[i], CLASS_NAMES[j]))
    for count, a, b in sorted(pairs, reverse=True)[:k]:
        veg = " (both vegetation)" if a in VEGETATION and b in VEGETATION else ""
        print("  {:>4}  {:<22} -> {}{}".format(count, a, b, veg))


def fusion_behaviour(d):
    """Does fusion resolve disagreements, or just follow one branch?"""
    y, pr, ps, pc = d["y"], d["pred_rgb"], d["pred_spectral"], d["pred_c1"]
    agree = pr == ps
    dis = ~agree
    print("")
    print("Fusion behaviour on {} pooled test samples".format(len(y)))
    print("  branches agree      : {:>6} ({:.2%}), accuracy {:.4f}".format(
        int(agree.sum()), agree.mean(), (pc[agree] == y[agree]).mean()))
    print("  branches disagree   : {:>6} ({:.2%})".format(int(dis.sum()), dis.mean()))
    if dis.sum():
        print("    rgb correct       : {:.4f}".format((pr[dis] == y[dis]).mean()))
        print("    spectral correct  : {:.4f}".format((ps[dis] == y[dis]).mean()))
        print("    fused correct     : {:.4f}".format((pc[dis] == y[dis]).mean()))
        follows_rgb = (pc[dis] == pr[dis]).mean()
        print("    fused followed rgb: {:.2%}  (0.5 = no systematic preference)".format(follows_rgb))
        # the ceiling any decision-level fusion could reach on these samples
        oracle = ((pr[dis] == y[dis]) | (ps[dis] == y[dis])).mean()
        print("    ORACLE (pick the right branch every time): {:.4f}".format(oracle))
        print("    -> fusion captures {:.1%} of the available headroom".format(
            (pc[dis] == y[dis]).mean() / oracle if oracle else 0.0))


def vegetation_check(rows_rgb, rows_spec):
    """C2's premise: NDVI should help precisely on the vegetation classes."""
    print("")
    print("C2 check -- does the spectral branch beat RGB on vegetation classes?")
    print("{:<24} {:>10} {:>10} {:>9}".format("class", "rgb F1", "spectral F1", "delta"))
    print("-" * 56)
    veg_deltas, other_deltas = [], []
    for name in CLASS_NAMES:
        d = rows_spec[name]["f1"] - rows_rgb[name]["f1"]
        (veg_deltas if name in VEGETATION else other_deltas).append(d)
        mark = " *" if name in VEGETATION else ""
        print("{:<24} {:>10.4f} {:>10.4f} {:>+9.4f}{}".format(
            name, rows_rgb[name]["f1"], rows_spec[name]["f1"], d, mark))
    print("")
    print("  mean delta, vegetation classes (*): {:+.4f}".format(np.mean(veg_deltas)))
    print("  mean delta, other classes         : {:+.4f}".format(np.mean(other_deltas)))


def main():
    args = parse_args()
    d = load_pooled(args.tag, args.folds)
    print("=" * 62)
    print("Per-class analysis: tag={}  ({} pooled test samples)".format(args.tag, len(d["y"])))
    print("=" * 62)

    rows_c1, cm_c1 = per_class_table(d["y"], d["pred_c1"], "Fused model (C1) per-class performance")
    top_confusions(cm_c1)
    fusion_behaviour(d)

    rows_rgb, _ = per_class_table(d["y"], d["pred_rgb"], "RGB branch per-class performance")
    rows_spec, _ = per_class_table(d["y"], d["pred_spectral"], "Spectral branch per-class performance")
    vegetation_check(rows_rgb, rows_spec)

    payload = {"tag": args.tag, "n_samples": int(len(d["y"])),
               "per_class_fused": rows_c1, "per_class_rgb": rows_rgb,
               "per_class_spectral": rows_spec,
               "confusion_matrix_fused": cm_c1.tolist(),
               "class_names": CLASS_NAMES}

    if args.compare:
        other = load_pooled(args.compare, args.folds)
        rows_o, _ = per_class_table(other["y"], other["pred_c1"],
                                     "Comparison tag '{}' per-class".format(args.compare))
        print("")
        print("Per-class F1 delta: {} minus {}".format(args.tag, args.compare))
        print("-" * 56)
        for name in CLASS_NAMES:
            delta = rows_c1[name]["f1"] - rows_o[name]["f1"]
            print("  {:<24} {:>+9.4f}".format(name, delta))
        payload["per_class_compare"] = {"tag": args.compare, "rows": rows_o}

    out = CV_DIR / args.tag / "error_analysis.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("")
    print("Saved " + str(out))


if __name__ == "__main__":
    main()
