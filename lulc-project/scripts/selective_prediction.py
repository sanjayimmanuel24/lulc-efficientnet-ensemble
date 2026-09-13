"""
Selective prediction (classification with a reject option) for the dual-branch model.

Motivation: an operational land-cover product does not need every tile classified
automatically. It needs to know WHICH tiles it can trust. A model that classifies
95% of tiles at very low error and flags the rest for an analyst is more useful
than one that classifies everything at 99.3%.

The question this asks that a single-backbone model cannot: a dual-branch model
emits TWO uncertainty signals -- the fused confidence, and whether the two
branches AGREE. Single models only have the first. Does the second add anything?

Metrics follow the standard selective-prediction literature (El-Yaniv & Wiener;
Geifman et al.): risk-coverage curves, AURC, and excess-AURC over the optimal
ordering. Lower is better for all three.

Runs on the saved per-sample probabilities -- no GPU, no retraining.

    python scripts/selective_prediction.py --tag default
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

from src.evaluation.statistical_tests import paired_ttest, wilcoxon_test, bootstrap_ci

PROJECT_ROOT = Path(__file__).parent.parent
OUT_FIG = PROJECT_ROOT / "results" / "figures"

N_FOLDS = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="default")
    p.add_argument("--coverages", type=float, nargs="+", default=[1.0, 0.99, 0.95, 0.90, 0.80])
    p.add_argument("--risk-targets", type=float, nargs="+", default=[0.01, 0.005, 0.002])
    p.add_argument("--compare", type=str, nargs="+", default=[],
                   help="Other tags whose pooled probabilities should appear in the deployment "
                        "table, e.g. baseline_resnet50. Must be built on the same folds.")
    return p.parse_args()


def risk_coverage(correct, score):
    """
    Sort by confidence descending; return (coverage, risk) at every prefix.

    risk[i] = error rate among the i most-confident samples. This is the curve an
    operator reads: "if I accept the top X% by confidence, what error do I get?"
    """
    order = np.argsort(-score, kind="mergesort")
    c = correct[order].astype(np.float64)
    n = len(c)
    cum_err = np.cumsum(1.0 - c)
    k = np.arange(1, n + 1)
    return k / n, cum_err / k


def aurc(correct, score):
    """Area under the risk-coverage curve (mean risk over all coverage levels)."""
    _, risk = risk_coverage(correct, score)
    return float(risk.mean())


def optimal_aurc(correct):
    """AURC of a perfect confidence ranking (all correct samples first) -- the floor."""
    return aurc(correct, correct.astype(np.float64))


def acc_at_coverage(correct, score, cov):
    n = len(correct)
    k = max(1, int(round(cov * n)))
    order = np.argsort(-score, kind="mergesort")[:k]
    return float(correct[order].mean())


def coverage_at_risk(correct, score, target_risk):
    """Largest coverage whose risk stays at or below the target. 0 if unreachable."""
    cov, risk = risk_coverage(correct, score)
    ok = np.where(risk <= target_risk)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def build_scores(d):
    """
    Confidence scores to compare. The first two are what a SINGLE model can offer;
    the rest use the second branch, which is the dual-branch-specific part.
    """
    rgb, spec, avg, c1 = d["rgb"], d["spectral"], d["avg"], d["c1"]
    conf_c1 = c1.max(-1)

    # continuous branch (dis)agreement: 1 - total variation distance between the
    # two branches' distributions. 1.0 = identical, 0.0 = disjoint.
    tv = 0.5 * np.abs(rgb - spec).sum(-1)
    agreement = 1.0 - tv

    return {
        "rgb_branch_confidence": rgb.max(-1),              # single-model baseline
        "spectral_branch_confidence": spec.max(-1),        # single-model baseline
        "fixed_avg_confidence": avg.max(-1),
        "c1_fusion_confidence": conf_c1,
        "branch_agreement_only": agreement,                # dual-branch signal alone
        "c1_conf_x_agreement": conf_c1 * agreement,        # both signals combined
    }


def per_fold(arr):
    """pooled_test_probs.npz concatenates folds in order; each test fold is n/5."""
    n = len(arr)
    assert n % N_FOLDS == 0, "cannot split {} samples into {} folds".format(n, N_FOLDS)
    return np.split(arr, N_FOLDS)


def main():
    args = parse_args()
    npz = PROJECT_ROOT / "results" / "cv" / args.tag / "pooled_test_probs.npz"
    if not npz.exists():
        raise FileNotFoundError(
            "{} not found -- run scripts/make_calibration_figures.py --tag {} first".format(npz, args.tag))
    d = np.load(npz)
    y = d["y"]
    scores = build_scores(d)
    pred_c1 = d["c1"].argmax(-1)
    correct_c1 = (pred_c1 == y)

    print("")
    print("Selective prediction, tag={}, {} pooled test samples".format(args.tag, len(y)))
    print("full-coverage accuracy (C1 fusion): {:.4f}".format(correct_c1.mean()))
    print("optimal AURC (perfect ranking): {:.5f}".format(optimal_aurc(correct_c1)))
    print("")

    # --- deployment comparison: each MODEL with its OWN predictions and OWN confidence ---
    # This is the operationally honest question: if you deployed a single-branch model,
    # what coverage/risk could you actually offer? The table after it holds predictions
    # fixed to isolate uncertainty quality from accuracy.
    print("A) DEPLOYMENT VIEW -- each model uses its own predictions and its own confidence")
    dep_hdr = "{:<28} {:>9} {:>9}".format("model", "acc", "AURC")
    for c in args.coverages[1:]:
        dep_hdr += " {:>9}".format("acc@{:.0%}".format(c))
    for r in args.risk_targets:
        dep_hdr += " {:>10}".format("cov@{:.1%}".format(r))
    print(dep_hdr)
    print("-" * len(dep_hdr))
    deployment = {}
    entries = [("rgb_branch_alone", d["rgb"], y), ("spectral_branch_alone", d["spectral"], y),
               ("fixed_avg", d["avg"], y), ("c1_fusion", d["c1"], y)]
    for other in args.compare:
        o_npz = PROJECT_ROOT / "results" / "cv" / other / "pooled_test_probs.npz"
        if not o_npz.exists():
            raise FileNotFoundError(
                "{} not found -- run make_calibration_figures.py --tag {} first".format(o_npz, other))
        od = np.load(o_npz)
        # Same folds means the same samples in the same order; verify rather than assume,
        # since a mismatched ordering would silently corrupt every comparison.
        if not np.array_equal(od["y"], y):
            raise SystemExit(
                "label vectors differ between '{}' and '{}' -- not the same folds/order".format(
                    args.tag, other))
        entries.append((other + " (single model)", od["c1"], y))

    for name, probs, yy in entries:
        corr = (probs.argmax(-1) == yy)
        conf = probs.max(-1)
        a = aurc(corr, conf)
        row = {"accuracy": float(corr.mean()), "aurc": a}
        line = "{:<28} {:>9.4f} {:>9.5f}".format(name, corr.mean(), a)
        for c in args.coverages[1:]:
            v = acc_at_coverage(corr, conf, c); row["acc_at_{:.2f}".format(c)] = v
            line += " {:>9.4f}".format(v)
        for r in args.risk_targets:
            v = coverage_at_risk(corr, conf, r); row["cov_at_risk_{:.3f}".format(r)] = v
            line += " {:>10.3f}".format(v)
        print(line)
        row["aurc_per_fold"] = [aurc(c, cf) for c, cf in zip(per_fold(corr), per_fold(conf))]
        deployment[name] = row
    print("")

    # Paired per-fold test of the proposed model against every comparison model.
    # Folds are the replication unit, matching how every other comparison in this
    # project is tested.
    ours = np.array(deployment["c1_fusion"]["aurc_per_fold"])
    for name in [n for n, _, _ in entries if n != "c1_fusion"]:
        theirs = np.array(deployment[name]["aurc_per_fold"])
        dd = ours - theirs                      # negative = our AURC is lower = better
        _, pv = paired_ttest(ours, theirs)
        _, pw = wilcoxon_test(ours, theirs)
        lo_, hi_ = bootstrap_ci(dd, n_boot=10000)
        ratio = theirs.mean() / ours.mean() if ours.mean() else float("nan")
        print("  c1_fusion vs {:<28} AURC {:.5f} vs {:.5f} ({:.2f}x better), "
              "better on {}/{} folds, paired-t p={:.4f}, wilcoxon p={:.4f}, CI [{:+.5f},{:+.5f}]".format(
                  name, ours.mean(), theirs.mean(), ratio,
                  int((dd < 0).sum()), N_FOLDS, pv, pw, lo_, hi_))
    print("")

    print("B) UNCERTAINTY-QUALITY VIEW -- all scores rank the SAME C1 predictions")
    # Every score ranks the SAME predictions (C1 fusion); only the ordering differs.
    # That isolates the quality of the uncertainty signal from the accuracy of the model.
    hdr = "{:<28} {:>9} {:>10}".format("selection score", "AURC", "excess")
    for c in args.coverages:
        hdr += " {:>9}".format("acc@{:.0%}".format(c))
    for r in args.risk_targets:
        hdr += " {:>10}".format("cov@{:.1%}".format(r))
    print(hdr)
    print("-" * len(hdr))

    opt = optimal_aurc(correct_c1)
    table, fold_aurcs = {}, {}
    for name, s in scores.items():
        a = aurc(correct_c1, s)
        row = {"aurc": a, "excess_aurc": a - opt}
        line = "{:<28} {:>9.5f} {:>10.5f}".format(name, a, a - opt)
        for c in args.coverages:
            v = acc_at_coverage(correct_c1, s, c)
            row["acc_at_{:.2f}".format(c)] = v
            line += " {:>9.4f}".format(v)
        for r in args.risk_targets:
            v = coverage_at_risk(correct_c1, s, r)
            row["cov_at_risk_{:.3f}".format(r)] = v
            line += " {:>10.3f}".format(v)
        print(line)
        table[name] = row
        fold_aurcs[name] = np.array([aurc(cc, ss) for cc, ss
                                     in zip(per_fold(correct_c1), per_fold(s))])

    # --- does the second branch's agreement signal beat confidence alone? ---
    for alt in ("branch_agreement_only", "c1_conf_x_agreement"):
        a0, a1 = fold_aurcs["c1_fusion_confidence"], fold_aurcs[alt]
        dd = a1 - a0
        _, pp = paired_ttest(a1, a0)
        print("")
        print("  {} vs confidence-only: mean {:+.5f}, better on {}/{} folds, paired-t p={:.4f}".format(
            alt, dd.mean(), int((dd < 0).sum()), N_FOLDS, pp))

    base, comb = "c1_fusion_confidence", "c1_conf_x_agreement"
    a_b, a_c = fold_aurcs[base], fold_aurcs[comb]
    diff = a_c - a_b                     # negative = combined is better
    _, p_t = paired_ttest(a_c, a_b)
    _, p_w = wilcoxon_test(a_c, a_b)
    lo, hi = bootstrap_ci(diff, n_boot=10000)
    print("")
    print("Does inter-branch agreement add to confidence alone?")
    print("  AURC per fold, confidence only : " + " ".join("{:.5f}".format(x) for x in a_b))
    print("  AURC per fold, conf x agreement: " + " ".join("{:.5f}".format(x) for x in a_c))
    print("  mean difference {:+.5f} (negative favours the combined signal)".format(diff.mean()))
    print("  better on {}/{} folds | paired-t p={:.4f} | wilcoxon p={:.4f}".format(
        int((diff < 0).sum()), N_FOLDS, p_t, p_w))
    print("  bootstrap 95% CI on the fold difference: [{:+.5f}, {:+.5f}]".format(lo, hi))
    relative = 100.0 * (a_b.mean() - a_c.mean()) / a_b.mean() if a_b.mean() else 0.0
    print("  relative AURC reduction: {:.1f}%".format(relative))

    # --- figure: risk-coverage curves ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    show = ["rgb_branch_confidence", "spectral_branch_confidence",
            "c1_fusion_confidence", "c1_conf_x_agreement"]
    for name in show:
        cov, risk = risk_coverage(correct_c1, scores[name])
        axes[0].plot(cov, 100 * risk, lw=1.6, label="{} (AURC {:.4f})".format(name, table[name]["aurc"]))
    axes[0].set_xlabel("coverage (fraction of tiles classified automatically)")
    axes[0].set_ylabel("risk = error rate on accepted tiles (%)")
    axes[0].set_title("Risk-coverage")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    for name in show:
        cov, risk = risk_coverage(correct_c1, scores[name])
        m = cov >= 0.70
        axes[1].plot(cov[m], 100 * risk[m], lw=1.6)
    axes[1].set_xlabel("coverage")
    axes[1].set_ylabel("risk (%)")
    axes[1].set_title("Operating region (coverage >= 70%)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Selective prediction on EuroSAT-MS (tag={}, {} samples)".format(args.tag, len(y)))
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "risk_coverage_{}.png".format(args.tag)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("")
    print("saved " + str(out))

    res = PROJECT_ROOT / "results" / "cv" / args.tag / "selective_prediction.json"
    with open(res, "w") as f:
        json.dump({"tag": args.tag, "n": int(len(y)), "deployment_view": deployment,
                   "full_coverage_accuracy": float(correct_c1.mean()),
                   "optimal_aurc": opt, "scores": table,
                   "agreement_vs_confidence": {
                       "aurc_confidence_per_fold": a_b.tolist(),
                       "aurc_combined_per_fold": a_c.tolist(),
                       "mean_difference": float(diff.mean()),
                       "folds_better": int((diff < 0).sum()),
                       "paired_t_p": float(p_t), "wilcoxon_p": float(p_w),
                       "bootstrap_ci": [float(lo), float(hi)],
                       "relative_aurc_reduction_pct": float(relative)}}, f, indent=2)
    print("saved " + str(res))


if __name__ == "__main__":
    main()
