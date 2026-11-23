"""D4: the per-model profile, the selector's result as one JSON file.

The selector (D1-D3) runs once per model and writes ``profiles/<model>.json``.
The API (F1) and the demo (F3) only read this file; they never run the selector.
The full field list is documented in CONTRACT.md, section "Profile (D4)".

Run from ``experiments/llm-probly`` to check a saved profile::

    uv run python -m selector.profile profiles/qwen1.5b.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

SCHEMA_VERSION = 1
PROFILES_ROOT = Path(__file__).resolve().parent.parent / "profiles"

Family = Literal["agreement", "self_eval", "baseline", "hybrid"]
# Same values as E2's method registry (common/registry.py); tests/test_registry.py checks they agree.
Access = Literal["text_only", "logprobs", "white_box"]
CalibrationType = Literal["none", "platt", "isotonic"]

FAMILIES = ("agreement", "self_eval", "baseline", "hybrid")
ACCESS_LEVELS = ("text_only", "logprobs", "white_box")
CALIBRATION_TYPES = ("none", "platt", "isotonic")


@dataclass
class Metrics:
    """AUROC and PRR of one method on one part of the data, with 95 % bootstrap intervals."""

    auroc: float
    auroc_low: float
    auroc_high: float
    prr: float
    prr_low: float
    prr_high: float


@dataclass
class MethodChoice:
    """The method the selector chose for this model.

    Attributes:
        name: Method name, same as its plugin and result file (e.g. ``"p_true_zeroshot"``),
            or ``"stack"`` when D2's combination was chosen.
        family: Method family, as declared in the plugin (CONTRACT.md).
        access: What the method needs from the model (``common/registry.py``).
        cost: Generation passes per question, greedy answer included (``common/registry.py``).
        parameters: Settings the API needs to run the method (e.g. ``{"n_samples": 10}``).
            For ``"stack"``: ``methods``, ``weights``, ``intercept``, ``means``, ``stds``.
        probability: True if ``1 - uncertainty`` is already a probability of being correct.
    """

    name: str
    family: Family
    access: Access
    cost: float
    parameters: dict[str, Any] = field(default_factory=dict)
    probability: bool = False


@dataclass
class Calibration:
    """Maps the method's raw uncertainty to a confidence in [0, 1] (D3).

    Attributes:
        type: ``"none"`` (not calibrated yet), ``"platt"`` or ``"isotonic"``.
        parameters: ``platt``: ``a`` and ``b``, confidence = 1 / (1 + exp(a * u + b)).
            ``isotonic``: ``x`` (increasing uncertainties) and ``y`` (confidences),
            linear interpolation in between, constant outside.
        ece_before: ECE of the uncalibrated score on the held-out part (only for
            probability methods, otherwise None).
        ece_after: ECE after calibration on the held-out part.
    """

    type: CalibrationType = "none"
    parameters: dict[str, Any] = field(default_factory=dict)
    ece_before: float | None = None
    ece_after: float | None = None


@dataclass
class Profile:
    """Everything the API and the demo need to know about one model.

    Attributes:
        model: Short model name, as in ``data/`` and ``results/`` (e.g. ``"qwen1.5b"``).
        model_id: Hugging Face id, from the score files (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
        dataset: Dataset the selector was run on (e.g. ``"triviaqa"``).
        n_questions: Number of questions in the run.
        split_seed: Seed of the calibration / held-out split (selector.split).
        calibration_size: Number of questions the selector chose and fitted on.
        method: The chosen method.
        reason: One sentence on why, shown in the demo.
        calibration_metrics: Metrics of the chosen method on the calibration part.
        held_out_metrics: Metrics of the chosen method on the held-out part (filled by ``--save``).
        calibration: How to turn the raw score into a confidence.
        candidates: Calibration-part metrics of every method that was compared.
        schema_version: Version of this format. The loader refuses other versions.
        created_at: When the profile was written (UTC, ISO 8601).
    """

    model: str
    model_id: str
    dataset: str
    n_questions: int
    split_seed: int
    calibration_size: int
    method: MethodChoice
    reason: str
    calibration_metrics: Metrics
    held_out_metrics: Metrics | None = None
    calibration: Calibration = field(default_factory=Calibration)
    candidates: dict[str, Metrics] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def validate_profile(profile: Profile) -> list[str]:
    """Return a list of problems; an empty list means the profile is valid."""
    problems = []
    if profile.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version is {profile.schema_version}, expected {SCHEMA_VERSION}")
    if profile.method.family not in FAMILIES:
        problems.append(f"method.family must be one of {FAMILIES}, got {profile.method.family!r}")
    if profile.method.access not in ACCESS_LEVELS:
        problems.append(f"method.access must be one of {ACCESS_LEVELS}, got {profile.method.access!r}")
    if not (math.isfinite(profile.method.cost) and profile.method.cost >= 0):
        problems.append(f"method.cost must be a number >= 0, got {profile.method.cost}")
    if not 0 < profile.calibration_size < profile.n_questions:
        problems.append(f"calibration_size must be between 1 and {profile.n_questions - 1}")
    problems.extend(_calibration_problems(profile.calibration))
    return problems


def _calibration_problems(calibration: Calibration) -> list[str]:
    """Check that the calibration type and its parameters fit together."""
    params = calibration.parameters
    if calibration.type not in CALIBRATION_TYPES:
        return [f"calibration.type must be one of {CALIBRATION_TYPES}, got {calibration.type!r}"]
    if calibration.type == "platt" and not {"a", "b"} <= params.keys():
        return ["platt calibration needs parameters a and b"]
    if calibration.type == "isotonic":
        x, y = params.get("x", []), params.get("y", [])
        if not x or len(x) != len(y):
            return ["isotonic calibration needs non-empty x and y of the same length"]
        if np.any(np.diff(x) < 0):
            return ["isotonic calibration x must be increasing"]
        if min(y) < 0 or max(y) > 1:
            return ["isotonic calibration y must lie in [0, 1]"]
    return []


def save_profile(profile: Profile, path: str | Path | None = None) -> Path:
    """Check the profile and write it as JSON.

    Args:
        profile: The profile to save.
        path: Target file. Defaults to ``profiles/<model>.json``.

    Returns:
        The path that was written.

    Raises:
        ValueError: If the profile is invalid.
    """
    problems = validate_profile(profile)
    if problems:
        raise ValueError(f"{profile.model}: " + "; ".join(problems))
    path = Path(path) if path is not None else PROFILES_ROOT / f"{profile.model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2) + "\n")
    return path


def load_profile(path: str | Path) -> Profile:
    """Read and check a profile written by ``save_profile``.

    Raises:
        ValueError: If a field is missing, unknown or invalid, or the schema version differs.
    """
    data = json.loads(Path(path).read_text())
    try:
        fields: dict[str, Any] = {
            **data,
            "method": MethodChoice(**data["method"]),
            "calibration_metrics": Metrics(**data["calibration_metrics"]),
            "held_out_metrics": Metrics(**data["held_out_metrics"]) if data.get("held_out_metrics") else None,
            "calibration": Calibration(**data.get("calibration", {})),
            "candidates": {name: Metrics(**m) for name, m in data.get("candidates", {}).items()},
        }
        profile = Profile(**fields)
    except (KeyError, TypeError) as error:
        msg = f"{path}: not a valid profile ({error})"
        raise ValueError(msg) from error
    problems = validate_profile(profile)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return profile


def confidence(profile: Profile, uncertainty: ArrayLike) -> np.ndarray:
    """Turn the chosen method's raw uncertainty into a confidence in [0, 1].

    This is the one function the API calls after scoring an answer.

    Args:
        profile: Profile of the model that produced the answer.
        uncertainty: Raw uncertainty (higher = more uncertain), one value or an array.

    Returns:
        Confidence that the answer is correct, same shape as ``uncertainty``.

    Raises:
        ValueError: If the profile is not calibrated and the method's score is not a probability.
    """
    u = np.asarray(uncertainty, dtype=float)
    cal = profile.calibration
    if cal.type == "platt":
        return 1.0 / (1.0 + np.exp(cal.parameters["a"] * u + cal.parameters["b"]))
    if cal.type == "isotonic":
        return np.interp(u, cal.parameters["x"], cal.parameters["y"])
    if profile.method.probability:
        return np.clip(1.0 - u, 0.0, 1.0)
    msg = f"{profile.model}: {profile.method.name} is not calibrated yet and its score is not a probability"
    raise ValueError(msg)


def main() -> None:
    """Load a profile, check it and print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.path)
    m = profile.calibration_metrics
    print(f"{profile.model} ({profile.model_id}), schema v{profile.schema_version}")
    print(
        f"  method       {profile.method.name} ({profile.method.family}, {profile.method.access}, "
        f"cost {profile.method.cost:g})"
    )
    print(f"  reason       {profile.reason}")
    print(
        f"  AUROC        {m.auroc:.3f} [{m.auroc_low:.3f}, {m.auroc_high:.3f}] on {profile.calibration_size} "
        f"calibration questions"
    )
    print(f"  calibration  {profile.calibration.type}")


if __name__ == "__main__":
    main()
