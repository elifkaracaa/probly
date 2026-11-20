"""Tests for selector.calibrate (D3: calibrated confidence)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from selector.calibrate import apply, calibrate_profile, choose_calibration_method, fit_calibration
from selector.profile import MethodChoice, Metrics, Profile, load_profile, save_profile

from probly.calibrator import calibrate
from probly.transformation.calibration import isotonic_regression, platt_scaling, sklearn_identity_logit_estimator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def data(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores where wrong answers are more uncertain on average."""
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.5
    uncertainty = rng.normal(0, 1, n) + 1.5 * (~correct)
    return uncertainty, correct


def make_profile(probability: bool) -> Profile:
    """Profile as the selector writes it, before calibration."""
    metrics = Metrics(auroc=0.8, auroc_low=0.7, auroc_high=0.9, prr=0.5, prr_low=0.3, prr_high=0.7)
    return Profile(
        model="m",
        model_id="org/m",
        dataset="triviaqa",
        n_questions=200,
        split_seed=0,
        calibration_size=100,
        method=MethodChoice(name="x", family="self_eval", access="logprobs", cost=2.0, probability=probability),
        reason="r",
        calibration_metrics=metrics,
    )


@pytest.mark.parametrize(("method", "wrapper"), [("platt", platt_scaling), ("isotonic", isotonic_regression)])
def test_stored_parameters_reproduce_problys_calibrator(method: str, wrapper: Callable[..., Any]) -> None:
    u, correct = data()
    calibration = fit_calibration(u, correct, method)
    fitted = calibrate(wrapper(sklearn_identity_logit_estimator()), correct.astype(int), (-u).reshape(-1, 1))
    assert fitted is not None
    grid = np.linspace(-4, 6, 101)
    expected = fitted.predict_proba((-grid).reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(apply(calibration, grid), expected, atol=1e-12)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_confidence_is_in_unit_interval_and_falls_with_uncertainty(method: str) -> None:
    u, correct = data()
    conf = apply(fit_calibration(u, correct, method), np.linspace(-4, 6, 101))
    assert conf.min() >= 0
    assert conf.max() <= 1
    assert np.all(np.diff(conf) <= 1e-12)


def test_bad_inputs_raise() -> None:
    u, correct = data()
    with pytest.raises(ValueError, match="method"):
        fit_calibration(u, correct, "temperature")
    with pytest.raises(ValueError, match="both correct and wrong"):
        fit_calibration(u, np.ones_like(correct), "platt")


def test_choice_reports_both_methods() -> None:
    u, correct = data()
    method, brier = choose_calibration_method(u, correct)
    assert method in {"platt", "isotonic"}
    assert set(brier) == {"platt", "isotonic"}
    assert brier[method] == min(brier.values())


def test_calibration_fixes_an_overconfident_probability(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # Says 90 % or 70 % confident, but is right only ~65 % / ~35 % of the time.
    u = np.where(rng.random(400) < 0.5, 0.1, 0.3)
    correct = np.where(u == 0.1, rng.random(400) < 0.65, rng.random(400) < 0.35)
    profile, _ = calibrate_profile(make_profile(probability=True), u[:200], correct[:200], u[200:], correct[200:])
    assert profile.calibration.type in {"platt", "isotonic"}
    assert profile.calibration.ece_before is not None
    assert profile.calibration.ece_after is not None
    assert profile.calibration.ece_after < profile.calibration.ece_before
    assert load_profile(save_profile(profile, tmp_path / "m.json")) == profile


def test_raw_score_has_no_ece_before() -> None:
    u, correct = data()
    profile, _ = calibrate_profile(make_profile(probability=False), u[:150], correct[:150], u[150:], correct[150:])
    assert profile.calibration.ece_before is None
    assert profile.calibration.ece_after is not None
    assert 0 <= profile.calibration.ece_after <= 1
