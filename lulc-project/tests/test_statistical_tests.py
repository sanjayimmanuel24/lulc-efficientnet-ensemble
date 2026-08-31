import pytest
import numpy as np
from src.evaluation.statistical_tests import (
    paired_ttest, wilcoxon_test, mcnemar_test, cohens_d_paired, bootstrap_ci,
)


def test_identical_scores_give_pvalue_one_or_undefined_handled():
    a = np.array([0.9, 0.91, 0.89, 0.90, 0.92])
    b = a.copy()
    # identical paired samples: t-test statistic should be ~0 (or nan, since
    # variance of the difference is 0) -- check it doesn't crash and, when
    # finite, is close to 0 / p close to 1.
    t_stat, p_val = paired_ttest(a, b)
    assert np.isnan(t_stat) or abs(t_stat) < 1e-6


def test_clearly_better_model_gives_small_pvalue():
    rng = np.random.default_rng(0)
    # Model B consistently beats model A by ~5 points across 10 folds, with small noise.
    a = rng.normal(loc=0.80, scale=0.01, size=10)
    b = a + 0.05
    t_stat, p_val = paired_ttest(a, b)
    assert p_val < 0.01
    assert t_stat < 0  # a - mean is lower than b in ttest_rel(a, b) -> negative statistic


def test_wilcoxon_detects_consistent_improvement():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.80, scale=0.01, size=10)
    b = a + 0.05
    _, p_val = wilcoxon_test(a, b)
    assert p_val < 0.05


def test_mcnemar_no_discordant_pairs():
    y_true = np.array([0, 1, 1, 0])
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    n_disc, p_val = mcnemar_test(y_true, pred_a, pred_b)
    assert n_disc == 0
    assert p_val == 1.0


def test_mcnemar_detects_asymmetric_disagreement():
    rng = np.random.default_rng(2)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    # Make model B wrong on 60 samples where A is right, and A wrong on only
    # 5 samples where B is right -> strongly asymmetric disagreement.
    idx_b_wrong = rng.choice(n, size=60, replace=False)
    pred_b[idx_b_wrong] = 1 - y_true[idx_b_wrong]
    remaining = np.setdiff1d(np.arange(n), idx_b_wrong)
    idx_a_wrong = rng.choice(remaining, size=5, replace=False)
    pred_a[idx_a_wrong] = 1 - y_true[idx_a_wrong]

    n_disc, p_val = mcnemar_test(y_true, pred_a, pred_b)
    assert n_disc == 65
    assert p_val < 0.01


def test_cohens_d_zero_for_no_difference():
    a = np.array([0.8, 0.81, 0.79, 0.80])
    assert cohens_d_paired(a, a) == 0.0


def test_cohens_d_positive_when_a_greater():
    a = np.array([0.9, 0.9, 0.9, 0.9])
    b = np.array([0.8, 0.8, 0.8, 0.81])
    d = cohens_d_paired(a, b)
    assert d > 0


def test_bootstrap_ci_contains_true_mean_typically():
    rng = np.random.default_rng(3)
    true_mean = 0.85
    scores = rng.normal(loc=true_mean, scale=0.02, size=30)
    lower, upper = bootstrap_ci(scores, n_boot=2000, ci=0.95, seed=0)
    assert lower < upper
    assert lower < scores.mean() < upper


def test_bootstrap_ci_narrower_with_lower_variance():
    rng = np.random.default_rng(4)
    low_var = rng.normal(loc=0.85, scale=0.005, size=30)
    high_var = rng.normal(loc=0.85, scale=0.05, size=30)
    lo1, hi1 = bootstrap_ci(low_var, n_boot=2000, seed=1)
    lo2, hi2 = bootstrap_ci(high_var, n_boot=2000, seed=1)
    assert (hi1 - lo1) < (hi2 - lo2)


# --- Holm-Bonferroni: this project runs a dozen-plus paired tests ---

def test_holm_leaves_a_single_test_unchanged():
    from src.evaluation.statistical_tests import holm_bonferroni
    out = holm_bonferroni({"only": 0.04})
    assert out[0]["p_adjusted"] == pytest.approx(0.04)
    assert out[0]["significant_at_alpha"]


def test_holm_kills_a_marginal_result_in_a_large_family():
    """The concrete case: nonrgb vs default at p=0.044 among many comparisons."""
    from src.evaluation.statistical_tests import holm_bonferroni
    family = {"marginal": 0.044}
    family.update({f"other{i}": 0.5 for i in range(11)})
    out = {r["name"]: r for r in holm_bonferroni(family)}
    assert not out["marginal"]["significant_at_alpha"]
    assert out["marginal"]["p_adjusted"] == pytest.approx(0.044 * 12, abs=1e-9)


def test_holm_keeps_a_strong_result_significant():
    from src.evaluation.statistical_tests import holm_bonferroni
    family = {"strong": 1e-10}
    family.update({f"other{i}": 0.5 for i in range(11)})
    out = {r["name"]: r for r in holm_bonferroni(family)}
    assert out["strong"]["significant_at_alpha"]


def test_holm_adjusted_p_values_are_monotonic():
    from src.evaluation.statistical_tests import holm_bonferroni
    out = holm_bonferroni([0.001, 0.01, 0.02, 0.5])
    adj = [r["p_adjusted"] for r in out]
    assert adj == sorted(adj)
    assert all(a <= 1.0 for a in adj)


def test_holm_is_no_less_powerful_than_bonferroni():
    from src.evaluation.statistical_tests import holm_bonferroni
    ps = [0.001, 0.02, 0.03]
    out = holm_bonferroni(ps)
    for r in out:
        assert r["p_adjusted"] <= min(r["p_raw"] * len(ps), 1.0) + 1e-12


def test_holm_empty_family():
    from src.evaluation.statistical_tests import holm_bonferroni
    assert holm_bonferroni({}) == []
