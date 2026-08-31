import pytest
import numpy as np
from src.calibration.temperature_scaling import (
    softmax, fit_temperature, expected_calibration_error,
    maximum_calibration_error, brier_score,
)


def test_softmax_sums_to_one():
    logits = np.array([[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)


def test_softmax_temperature_smooths_distribution():
    logits = np.array([[5.0, 0.0, 0.0]])
    sharp = softmax(logits, temperature=1.0)
    smooth = softmax(logits, temperature=10.0)
    # Higher temperature -> closer to uniform -> lower max probability
    assert smooth.max() < sharp.max()


def test_ece_near_zero_for_perfectly_calibrated_predictions():
    # Construct predictions where confidence exactly matches empirical accuracy:
    # 100 samples at confidence 0.9, of which exactly 90 are correct.
    rng = np.random.default_rng(0)
    n = 100
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1
    labels = np.zeros(n, dtype=int)
    labels[90:] = 1  # 90 correct (class 0), 10 wrong (true class 1)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.02


def test_ece_high_for_overconfident_wrong_predictions():
    # All predictions claim 99% confidence but are only 50% accurate.
    n = 100
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.99
    probs[:, 1] = 0.01
    labels = np.zeros(n, dtype=int)
    labels[50:] = 1  # only half actually correct
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.4  # confidence ~0.99, accuracy ~0.5 -> gap ~0.49


def test_mce_at_least_as_large_as_ece():
    rng = np.random.default_rng(1)
    n = 200
    logits = rng.normal(size=(n, 4)) * 3
    labels = rng.integers(0, 4, size=n)
    probs = softmax(logits)
    ece = expected_calibration_error(probs, labels)
    mce = maximum_calibration_error(probs, labels)
    assert mce >= ece - 1e-9


def test_fit_temperature_recovers_known_scaling():
    # Generate labels from a "true" low-temperature (sharp) distribution,
    # then feed in logits that were artificially over-sharpened by 1/0.3,
    # i.e. as if divided by T_true = 0.3. Fitting should recover T close to
    # a value that raises (un-sharpens) them back towards good calibration,
    # i.e. a fitted T > 1.
    rng = np.random.default_rng(2)
    n = 2000
    k = 3
    true_logits = rng.normal(size=(n, k))
    over_sharpened_logits = true_logits / 0.3  # simulates an overconfident network
    labels = np.array([softmax(true_logits[i:i+1]).squeeze().argmax() for i in range(n)])
    # add label noise consistent with the true (well-calibrated) distribution
    probs_true = softmax(true_logits)
    labels = np.array([rng.choice(k, p=probs_true[i]) for i in range(n)])

    fitted_t = fit_temperature(over_sharpened_logits, labels)
    assert fitted_t > 1.0  # must un-sharpen the over-confident logits


def test_brier_score_zero_for_perfect_confident_correct_predictions():
    n = 10
    k = 3
    labels = np.array([0, 1, 2] * 3 + [0])
    probs = np.eye(k)[labels]
    assert np.isclose(brier_score(probs, labels), 0.0, atol=1e-9)


def test_brier_score_positive_for_uniform_predictions():
    n = 10
    k = 3
    labels = np.zeros(n, dtype=int)
    probs = np.full((n, k), 1.0 / k)
    assert brier_score(probs, labels) > 0.0


# --- adaptive (equal-mass) ECE: equal-width ECE is unreliable near the accuracy ceiling ---

def test_adaptive_ece_perfect_calibration_is_zero():
    from src.calibration.temperature_scaling import adaptive_expected_calibration_error
    rng = np.random.default_rng(0)
    # confidence 1.0 and always correct -> no gap under any binning
    probs = np.zeros((200, 4)); probs[:, 0] = 1.0
    labels = np.zeros(200, dtype=int)
    assert adaptive_expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_adaptive_ece_detects_overconfidence():
    from src.calibration.temperature_scaling import adaptive_expected_calibration_error
    # 90% confident, only 50% correct -> gap ~0.4
    probs = np.tile(np.array([0.9, 0.1]), (100, 1))
    labels = np.array([0] * 50 + [1] * 50)
    assert adaptive_expected_calibration_error(probs, labels) == pytest.approx(0.4, abs=0.02)


def test_adaptive_ece_keeps_bins_populated_at_the_ceiling():
    """
    The failure mode this function exists for: at ~99% accuracy nearly all mass
    sits in the top equal-width bin. Equal-mass binning must still spread samples,
    so it should not collapse to a single-bin estimate.
    """
    from src.calibration.temperature_scaling import (
        adaptive_expected_calibration_error, expected_calibration_error,
    )
    rng = np.random.default_rng(3)
    n = 2000
    conf = rng.uniform(0.95, 0.999, size=n)
    probs = np.zeros((n, 3))
    probs[:, 0] = conf
    probs[:, 1] = 1 - conf
    labels = (rng.uniform(size=n) > conf).astype(int)  # correct at rate ~conf
    a = adaptive_expected_calibration_error(probs, labels)
    e = expected_calibration_error(probs, labels)
    assert 0.0 <= a < 0.2 and 0.0 <= e < 0.2
    assert np.isfinite(a)


def test_adaptive_ece_handles_degenerate_single_confidence():
    from src.calibration.temperature_scaling import adaptive_expected_calibration_error
    probs = np.tile(np.array([0.7, 0.3]), (50, 1))   # every sample identical
    labels = np.zeros(50, dtype=int)
    val = adaptive_expected_calibration_error(probs, labels)
    assert val == pytest.approx(0.3, abs=1e-6)
