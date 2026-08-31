"""
Paired statistical comparison between two model variants (docs/BLUEPRINT.md Section 11).

Consumes the per-fold predictions written by scripts/run_cross_validation.py and runs
the paired tests the blueprint commits to:

  * McNemar, on predictions pooled across folds (every sample is tested exactly once,
    so pooling gives one prediction per dataset sample per variant)
  * paired t-test and Wilcoxon, on the k per-fold accuracies
  * Cohen's d (paired) and a bootstrap CI on the per-fold accuracy difference

Specifiers are TAG:VARIANT, where TAG is a results/cv/<tag> directory and VARIANT is one
of: rgb_branch_only, spectral_branch_only, fixed_average_50_50, c1_confidence_fusion.

Usage, from inside lulc-project/:

    # is C1 better than the 50/50 average it reduces to?  (within one architecture)
    python scripts/compare_variants.py --a default:c1_confidence_fusion \
                                        --b default:fixed_average_50_50

    # does the branch-input ablation help C1?  (across architectures)
    python scripts/compare_variants.py --a nonrgb:c1_confidence_fusion \
                                        --b default:c1_confidence_fusion

Refuses to compare runs whose folds differ: paired tests on non-identical samples are
meaningless, and that failure is silent if not checked.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.statistical_tests import (
    mcnemar_test, paired_ttest, wilcoxon_test, cohens_d_paired, bootstrap_ci,
)

PROJECT_ROOT = Path(__file__).parent.parent
CV_ROOT = PROJECT_ROOT / "results" / "cv"

VARIANT_TO_KEY = {
    "rgb_branch_only": "pred_rgb",
    "spectral_branch_only": "pred_spectral",
    "fixed_average_50_50": "pred_avg",
    "c1_confidence_fusion": "pred_c1",
}


def parse_spec(spec):
    if ":" not in spec:
        raise ValueError(
            "specifier must be TAG:VARIANT, got {!r}. Variants: {}".format(
                spec, ", ".join(VARIANT_TO_KEY)))
    tag, variant = spec.split(":", 1)
    if variant not in VARIANT_TO_KEY:
        raise ValueError("unknown variant {!r}; choose from {}".format(
            variant, ", ".join(VARIANT_TO_KEY)))
    return tag, variant


def load_side(tag, variant, n_folds):
    """Returns (pooled_y, pooled_pred, pooled_index, per_fold_accuracy)."""
    key = VARIANT_TO_KEY[variant]
    ys, preds, idxs, accs = [], [], [], []
    for i in range(n_folds):
        path = CV_ROOT / tag / "fold_{}_preds.npz".format(i)
        if not path.exists():
            raise FileNotFoundError(
                "missing {}\nRun: python scripts/run_cross_validation.py --tag {}".format(path, tag))
        d = np.load(path)
        ys.append(d["y"])
        preds.append(d[key])
        idxs.append(d["test_index"])
        accs.append(float((d[key] == d["y"]).mean()))
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(idxs), np.array(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="TAG:VARIANT (the candidate)")
    p.add_argument("--b", required=True, help="TAG:VARIANT (the baseline)")
    p.add_argument("-k", "--folds", type=int, default=5)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    tag_a, var_a = parse_spec(args.a)
    tag_b, var_b = parse_spec(args.b)

    y_a, pred_a, idx_a, accs_a = load_side(tag_a, var_a, args.folds)
    y_b, pred_b, idx_b, accs_b = load_side(tag_b, var_b, args.folds)

    # Guard: paired tests require the same samples in the same order on both sides.
    if not np.array_equal(idx_a, idx_b):
        raise SystemExit(
            "ERROR: '{}' and '{}' were evaluated on different folds, so a paired test would be "
            "invalid. Both tags must be produced from the same data/eurosat_folds_k{}.json.".format(
                args.a, args.b, args.folds))
    if not np.array_equal(y_a, y_b):
        raise SystemExit("ERROR: label vectors differ between the two runs; refusing to compare.")

    n = len(y_a)
    print("comparing  A = {}   vs   B = {}".format(args.a, args.b))
    print("pooled over {} folds, {} samples".format(args.folds, n))
    print("")
    print("A accuracy: {:.4f} (per-fold {})".format(
        (pred_a == y_a).mean(), " ".join("{:.4f}".format(x) for x in accs_a)))
    print("B accuracy: {:.4f} (per-fold {})".format(
        (pred_b == y_b).mean(), " ".join("{:.4f}".format(x) for x in accs_b)))
    diff = accs_a - accs_b
    print("A - B     : {:+.4f} mean over folds".format(diff.mean()))

    print("")
    print("--- McNemar (pooled per-sample) ---")
    # mcnemar_test returns the DISCORDANT-PAIR COUNT and the exact-binomial p.
    # It is not a chi-square statistic -- this implementation deliberately avoids
    # the chi-square approximation (see src/evaluation/statistical_tests.py), so
    # labelling it chi2 in a paper table would be wrong.
    n_discordant, pval = mcnemar_test(y_a, pred_a, pred_b)
    a_only = int(((pred_a == y_a) & (pred_b != y_b)).sum())
    b_only = int(((pred_a != y_a) & (pred_b == y_b)).sum())
    print("exact binomial p={:.4g}   discordant pairs n={:.0f}".format(pval, n_discordant))
    print("A right where B wrong: {}   B right where A wrong: {}".format(a_only, b_only))

    print("")
    print("--- Paired tests on per-fold accuracy (n={}) ---".format(args.folds))
    t_stat, t_p = paired_ttest(accs_a, accs_b)
    print("paired t : t={:.4f}  p={:.4g}".format(t_stat, t_p))
    try:
        w_stat, w_p = wilcoxon_test(accs_a, accs_b)
        print("wilcoxon : W={:.4f}  p={:.4g}".format(w_stat, w_p))
    except ValueError as e:
        # scipy raises when every paired difference is zero
        print("wilcoxon : not computable ({})".format(e))
    print("cohen's d: {:.4f}".format(cohens_d_paired(accs_a, accs_b)))
    lo, hi = bootstrap_ci(diff, n_boot=args.bootstrap_samples)
    print("bootstrap 95% CI on mean fold difference: [{:+.4f}, {:+.4f}]".format(lo, hi))

    significant = pval < args.alpha
    print("")
    if significant and a_only > b_only:
        verdict = "A beats B (McNemar p<{})".format(args.alpha)
    elif significant and b_only > a_only:
        verdict = "B beats A (McNemar p<{})".format(args.alpha)
    else:
        verdict = "no significant difference (McNemar p>={})".format(args.alpha)
    print("VERDICT: " + verdict)
    if lo <= 0 <= hi:
        print("         bootstrap CI on the fold-level difference includes 0.")

    out = CV_ROOT / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    name = "{}_{}__vs__{}_{}.json".format(tag_a, var_a, tag_b, var_b)
    with open(out / name, "w") as f:
        json.dump({"a": args.a, "b": args.b, "n_samples": int(n), "n_folds": args.folds,
                   "accuracy_a": float((pred_a == y_a).mean()),
                   "accuracy_b": float((pred_b == y_b).mean()),
                   "per_fold_accuracy_a": [float(x) for x in accs_a],
                   "per_fold_accuracy_b": [float(x) for x in accs_b],
                   "mean_fold_difference": float(diff.mean()),
                   "mcnemar_n_discordant": float(n_discordant), "mcnemar_p": float(pval),
                   "mcnemar_test": "exact binomial (not chi-square)",
                   "a_right_b_wrong": a_only, "b_right_a_wrong": b_only,
                   "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
                   "cohens_d": float(cohens_d_paired(accs_a, accs_b)),
                   "bootstrap_ci": [float(lo), float(hi)],
                   "verdict": verdict}, f, indent=2)
    print("Saved " + str(out / name))


if __name__ == "__main__":
    main()
