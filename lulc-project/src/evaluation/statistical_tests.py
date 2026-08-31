"""
Statistical validation utilities.

Guidance on which test to use (see docs/BLUEPRINT.md section 11):
  - paired_ttest: comparing mean per-fold accuracy/F1 of two models across
    the SAME k folds, when differences are approximately normal.
  - wilcoxon_test: same setting, but non-parametric (no normality
    assumption) -- preferred with k <= 10 folds, which is typical here.
  - mcnemar_test: comparing two models on the SAME test set at the level
    of individual predictions (not fold-level means) -- tests whether
    their disagreements are symmetric.
  - bootstrap_ci: a distribution-free confidence interval for a single
    model's metric, useful when you only have one test set and want an
    interval rather than a single point estimate.
  - cohens_d: effect size to report alongside any p-value, since p-values
    alone don't convey practical magnitude.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Returns (t_statistic, p_value) for paired per-fold scores."""
    result = stats.ttest_rel(scores_a, scores_b)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Returns (statistic, p_value). Non-parametric paired test."""
    result = stats.wilcoxon(scores_a, scores_b)
    return float(result.statistic), float(result.pvalue)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    """
    McNemar's test on paired per-sample correctness from two models on the
    same test set. Uses the exact binomial test (recommended when the
    number of discordant pairs is small, which is common on modest
    remote-sensing test sets) rather than the chi-square approximation.

    Returns (n_discordant_pairs, p_value).
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    # n01: A wrong, B right ; n10: A right, B wrong
    n01 = int(np.sum((~correct_a) & correct_b))
    n10 = int(np.sum(correct_a & (~correct_b)))
    n_discordant = n01 + n10
    if n_discordant == 0:
        return 0.0, 1.0
    result = stats.binomtest(min(n01, n10), n_discordant, p=0.5, alternative="two-sided")
    return float(n_discordant), float(result.pvalue)


def cohens_d_paired(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Effect size for paired samples: mean difference / std of differences."""
    diff = scores_a - scores_b
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return float(diff.mean() / std)


def bootstrap_ci(scores: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `scores`."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_means = np.array([
        rng.choice(scores, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    return lower, upper


def holm_bonferroni(p_values: dict[str, float] | list[float], alpha: float = 0.05):
    """
    Holm-Bonferroni step-down correction for a family of comparisons.

    Needed because this project runs many paired tests across variants and metrics,
    and its one marginal result (nonrgb vs default, p=0.044) does not survive even
    mild correction. Reporting uncorrected p-values from a dozen-plus comparisons
    is the kind of thing a reviewer checks.

    Holm is uniformly more powerful than plain Bonferroni at the same family-wise
    error rate, so there is no reason to use Bonferroni instead.

    Returns a list of dicts (sorted by raw p) with the adjusted p-value and the
    reject/retain decision at `alpha`.
    """
    items = list(p_values.items()) if isinstance(p_values, dict) else             [(str(i), p) for i, p in enumerate(p_values)]
    m = len(items)
    if m == 0:
        return []

    ordered = sorted(items, key=lambda kv: kv[1])
    out, running_max = [], 0.0
    for rank, (name, p) in enumerate(ordered):
        adjusted = min((m - rank) * p, 1.0)
        # step-down monotonicity: adjusted p-values must not decrease down the list
        running_max = max(running_max, adjusted)
        out.append({"name": name, "p_raw": float(p), "p_adjusted": float(running_max),
                    "significant_at_alpha": bool(running_max <= alpha), "rank": rank + 1,
                    "family_size": m})
    return out
