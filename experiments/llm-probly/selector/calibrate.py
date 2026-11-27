"""D3: turn the chosen method's raw score into a calibrated confidence in [0, 1].

The selector's chosen method gives an uncertainty on its own scale (msp: 0..inf,
p(True): 1 - probability, ...). For the API and the demo we need "how likely is
this answer correct", so we fit a calibration map on the calibration part:

- Platt scaling: a sigmoid, 2 parameters, smooth, works with few questions.
- Isotonic regression: a step function, more flexible, needs more data.

Both come from probly (``platt_scaling`` / ``isotonic_regression`` on an identity
estimator, which wrap scikit-learn's calibrators). Which one to use is decided by
5-fold cross-validation on the calibration part (Brier score), so the held-out
part stays untouched. ECE before and after is then measured on the held-out part.

The fitted map is stored in the profile in the uncertainty's own scale, so the API
only needs ``selector.profile.confidence(profile, uncertainty)``:

- platt: confidence = 1 / (1 + exp(a * u + b))
- isotonic: linear interpolation of ``(x, y)`` points, constant outside
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from common.evaluate import ece
import numpy as np
from sklearn.model_selection import StratifiedKFold

from probly.calibrator import calibrate
from probly.transformation.calibration import isotonic_regression, platt_scaling, sklearn_identity_logit_estimator
from selector.profile import Calibration, confidence

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from selector.profile import Profile

CALIBRATION_METHODS = ("platt", "isotonic")
DEFAULT_FOLDS = 5


def fit_calibration(uncertainty: ArrayLike, correct: ArrayLike, method: str) -> Calibration:
    """Fit one calibration map with probly and return it in the profile's format.

    Args:
        uncertainty: Raw uncertainties of the chosen method (higher = more uncertain).
        correct: Whether each answer was correct.
        method: ``"platt"`` or ``"isotonic"``.

    Returns:
        A ``Calibration`` whose parameters work on the raw uncertainty (no ECE yet).

    Raises:
        ValueError: If the method is unknown or the labels have only one class.
    """
    u = np.asarray(uncertainty, dtype=float)
    y = np.asarray(correct, dtype=bool).astype(int)
    if method not in CALIBRATION_METHODS:
        msg = f"method must be one of {CALIBRATION_METHODS}, got {method!r}"
        raise ValueError(msg)
    if y.min() == y.max():
        msg = "calibration needs both correct and wrong answers"
        raise ValueError(msg)

    # probly calibrates logits of "correct"; a higher uncertainty means a lower logit, so we feed -u.
    wrapper = platt_scaling if method == "platt" else isotonic_regression
    fitted = calibrate(wrapper(sklearn_identity_logit_estimator()), y, (-u).reshape(-1, 1))
    if fitted is None:
        msg = "probly's calibrate returned no fitted calibrator"
        raise RuntimeError(msg)
    calibrator = fitted.calibrated_classifiers_[0].calibrators[0]

    if method == "platt":
        # sklearn: p = 1 / (1 + exp(a_ * f + b_)) with f = -u  ->  a = -a_, b = b_ in terms of u.
        return Calibration(type="platt", parameters={"a": float(-calibrator.a_), "b": float(calibrator.b_)})
    # Thresholds are increasing in f = -u; reverse them so x is increasing in u.
    x = (-calibrator.X_thresholds_[::-1]).tolist()
    y_points = calibrator.y_thresholds_[::-1].tolist()
    return Calibration(type="isotonic", parameters={"x": x, "y": y_points})


def apply(calibration: Calibration, uncertainty: ArrayLike) -> np.ndarray:
    """Confidence in [0, 1] from a fitted calibration (same formulas as the API)."""
    u = np.asarray(uncertainty, dtype=float)
    if calibration.type == "platt":
        return 1.0 / (1.0 + np.exp(calibration.parameters["a"] * u + calibration.parameters["b"]))
    if calibration.type == "isotonic":
        return np.interp(u, calibration.parameters["x"], calibration.parameters["y"])
    msg = f"cannot apply calibration of type {calibration.type!r}"
    raise ValueError(msg)


def choose_calibration_method(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    folds: int = DEFAULT_FOLDS,
    seed: int = 0,
) -> tuple[str, dict[str, float]]:
    """Pick Platt or isotonic by cross-validated Brier score on the calibration part.

    Each fold fits the map on the other folds and scores the left-out questions with
    the Brier score (mean squared error between confidence and correctness; lower is
    better). It is used instead of ECE because ECE is too noisy on 20 questions.

    Args:
        uncertainty: Raw uncertainties of the calibration part.
        correct: Correctness labels of the calibration part.
        folds: Number of cross-validation folds.
        seed: Random seed of the fold assignment.

    Returns:
        The better method and the mean Brier score of each method.
    """
    u = np.asarray(uncertainty, dtype=float)
    y = np.asarray(correct, dtype=bool)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: dict[str, list[float]] = {method: [] for method in CALIBRATION_METHODS}
    for train, test in splitter.split(u, y):
        for method in CALIBRATION_METHODS:
            calibration = fit_calibration(u[train], y[train], method)
            scores[method].append(float(np.mean((apply(calibration, u[test]) - y[test]) ** 2)))
    mean = {method: float(np.mean(values)) for method, values in scores.items()}
    return min(mean, key=lambda method: mean[method]), mean


def calibrate_profile(
    profile: Profile,
    u_calibration: ArrayLike,
    correct_calibration: ArrayLike,
    u_held_out: ArrayLike,
    correct_held_out: ArrayLike,
) -> tuple[Profile, dict[str, float]]:
    """Fill a profile's calibration: choose the method, fit it, report ECE on the held-out part.

    Args:
        profile: Profile from the selector (calibration type ``"none"``).
        u_calibration: Chosen method's raw uncertainties on the calibration part.
        correct_calibration: Labels of the calibration part.
        u_held_out: Chosen method's raw uncertainties on the held-out part.
        correct_held_out: Labels of the held-out part.

    Returns:
        The calibrated profile and the cross-validated Brier score of each calibration method.
    """
    method, brier = choose_calibration_method(u_calibration, correct_calibration)
    calibration = fit_calibration(u_calibration, correct_calibration, method)

    u_out = np.asarray(u_held_out, dtype=float)
    y_out = np.asarray(correct_held_out, dtype=bool)
    # ECE before only makes sense if the raw score already is a probability (p(True), verbalized).
    calibration.ece_before = round(ece(u_out, y_out), 4) if profile.method.probability else None
    calibrated = replace(profile, calibration=calibration)
    calibration.ece_after = round(ece(1.0 - confidence(calibrated, u_out), y_out), 4)
    return calibrated, brier
