import numpy as np
import pytest

from common.evaluate import _rejection_area, aurc, auroc, bootstrap_auroc, evaluate, prr

CORRECT = np.array([True] * 12 + [False] * 38)


def test_perfect_score_has_auroc_one():
    assert auroc((~CORRECT).astype(float), CORRECT) == 1.0


def test_inverted_score_has_auroc_zero():
    assert auroc(CORRECT.astype(float), CORRECT) == 0.0


def test_random_score_has_auroc_near_half():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    assert abs(auroc(rng.random(2000), correct) - 0.5) < 0.05


def test_perfect_score_has_much_lower_aurc_than_random_score():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    perfect = aurc((~correct).astype(float), correct)
    random_score = aurc(rng.random(2000), correct)
    assert perfect < random_score - 0.1


def test_aurc_of_random_score_is_close_to_error_rate():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    error_rate = 1 - correct.mean()
    assert abs(aurc(rng.random(2000), correct) - error_rate) < 0.05


def test_ties_do_not_depend_on_label_order():
    wrong_first = np.array([False] * 38 + [True] * 12)
    wrong_last = wrong_first[::-1]
    a = aurc(np.zeros(50), wrong_first)
    b = aurc(np.zeros(50), wrong_last)
    assert abs(a - b) < 0.08
    assert abs(a - 0.76) < 0.08


def test_evaluate_returns_both_metrics():
    result = evaluate((~CORRECT).astype(float), CORRECT)
    assert set(result) == {"auroc", "aurc", "prr"}


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [True, False, True])


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    correct = rng.random(200) < 0.5
    u = (~correct).astype(float) + rng.normal(0, 0.5, 200)
    lo, hi = bootstrap_auroc(u, correct, n_boot=200)
    assert lo < auroc(u, correct) < hi


def test_bootstrap_skips_single_class_resamples():
    lo, hi = bootstrap_auroc([0.9, 0.1, 0.2, 0.3, 0.4], [False, True, True, True, True], n_boot=200)
    assert 0.0 <= lo <= hi <= 1.0


# ----- PRR -------------------------------------------------------------------


def lm_polygraph_rejection_area(estimator, target):
    """Copy of lm-polygraph's PredictionRejectionArea.__call__ (max_rejection=1, MIT licence).

    Source: lm_polygraph/ue_metrics/pred_rej_area.py. Kept here as the reference
    our implementation must match.
    """
    ue = np.array(estimator)
    num_obs = len(ue)
    num_rej = num_obs
    ue_argsort = np.argsort(ue)
    sorted_metrics = np.array(target)[ue_argsort]
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
    return np.sum(scores) / num_rej


def test_rejection_area_matches_lm_polygraph():
    rng = np.random.default_rng(1)
    correct = rng.random(200) < 0.3
    u = rng.random(200)  # continuous, so there are no ties
    quality = correct.astype(float)
    assert np.isclose(_rejection_area(u, quality), lm_polygraph_rejection_area(u, quality))


def test_prr_matches_lm_polygraph_normalized():
    rng = np.random.default_rng(2)
    correct = rng.random(200) < 0.3
    u = (~correct).astype(float) + rng.normal(0, 0.8, 200)  # some signal, no ties
    quality = correct.astype(float)
    method = lm_polygraph_rejection_area(u, quality)
    oracle = lm_polygraph_rejection_area(-quality, quality)
    shuffles = [lm_polygraph_rejection_area(rng.permutation(200), quality) for _ in range(1000)]
    expected = (method - np.mean(shuffles)) / (oracle - np.mean(shuffles))
    assert abs(prr(u, correct) - expected) < 0.02  # lm-polygraph's random area is sampled


def test_perfect_score_has_prr_one():
    assert np.isclose(prr((~CORRECT).astype(float), CORRECT), 1.0)


def test_random_score_has_prr_near_zero():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.24
    assert abs(prr(rng.random(2000), correct)) < 0.1


def test_inverted_score_has_negative_prr():
    assert prr(CORRECT.astype(float), CORRECT) < -0.5  # worst order: correct answers rejected first


def test_constant_score_has_prr_near_zero():
    assert abs(prr(np.zeros(50), CORRECT)) < 0.1


def test_prr_is_nan_when_all_answers_are_correct():
    assert np.isnan(prr([0.1, 0.5, 0.9], [True, True, True]))
