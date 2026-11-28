"""Tests for selector.profile (D4: per-model profile schema)."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from selector import profile as profile_module
from selector.profile import (
    Calibration,
    MethodChoice,
    Metrics,
    Profile,
    confidence,
    load_profile,
    save_profile,
)

if TYPE_CHECKING:
    from pathlib import Path

METRICS = Metrics(auroc=0.8, auroc_low=0.7, auroc_high=0.9, prr=0.6, prr_low=0.4, prr_high=0.8)


def make_profile(**changes: object) -> Profile:
    """A valid profile; keyword arguments replace fields."""
    fields: dict[str, Any] = {
        "model": "m",
        "model_id": "org/m",
        "dataset": "triviaqa",
        "n_questions": 200,
        "split_seed": 0,
        "calibration_size": 100,
        "method": MethodChoice(
            name="p_true_zeroshot", family="self_eval", access="logprobs", cost=2.0, probability=True
        ),
        "reason": "best AUROC",
        "calibration_metrics": METRICS,
        "candidates": {"p_true_zeroshot": METRICS},
    }
    fields.update(changes)
    return Profile(**fields)


def test_save_and_load_give_the_same_profile(tmp_path: Path) -> None:
    profile = make_profile(
        held_out_metrics=METRICS,
        calibration=Calibration(type="platt", parameters={"a": 3.0, "b": -1.0}, ece_before=0.2, ece_after=0.05),
    )
    path = save_profile(profile, tmp_path / "m.json")
    assert load_profile(path) == profile


def test_profile_without_held_out_metrics_or_calibration_round_trips(tmp_path: Path) -> None:
    profile = make_profile()
    assert load_profile(save_profile(profile, tmp_path / "m.json")) == profile


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"method": MethodChoice(name="x", family="other", access="logprobs", cost=1.0)}, "family"),  # ty: ignore[invalid-argument-type]
        ({"method": MethodChoice(name="x", family="baseline", access="api", cost=1.0)}, "access"),  # ty: ignore[invalid-argument-type]
        ({"method": MethodChoice(name="x", family="baseline", access="logprobs", cost=-1.0)}, "cost"),
        ({"calibration_size": 200}, "calibration_size"),
        ({"calibration": Calibration(type="platt", parameters={"a": 1.0})}, "platt"),
        ({"calibration": Calibration(type="isotonic", parameters={"x": [0.5, 0.1], "y": [0.9, 0.2]})}, "increasing"),
        ({"schema_version": 99}, "schema_version"),
        ({"calibration": Calibration(type="other", parameters={})}, "calibration.type"),  # ty: ignore[invalid-argument-type]
        ({"calibration": Calibration(type="isotonic", parameters={})}, "non-empty"),
        ({"calibration": Calibration(type="isotonic", parameters={"x": [0.1, 0.5], "y": [0.9]})}, "same length"),
        ({"calibration": Calibration(type="isotonic", parameters={"x": [0.1, 0.5], "y": [0.9, 1.2]})}, "lie in"),
    ],
)
def test_invalid_profiles_are_not_saved(tmp_path: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        save_profile(make_profile(**changes), tmp_path / "m.json")


def test_load_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    path = save_profile(make_profile(), tmp_path / "m.json")
    data = json.loads(path.read_text())
    del data["reason"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a valid profile"):
        load_profile(path)
    data["reason"], data["extra"] = "r", 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a valid profile"):
        load_profile(path)


def test_confidence_platt_is_high_for_low_uncertainty() -> None:
    profile = make_profile(calibration=Calibration(type="platt", parameters={"a": 4.0, "b": -2.0}))
    low, high = confidence(profile, [0.0, 1.0])
    assert 0.5 < low < 1.0
    assert 0.0 < high < 0.5


def test_confidence_isotonic_interpolates_and_clamps() -> None:
    profile = make_profile(calibration=Calibration(type="isotonic", parameters={"x": [0.0, 1.0], "y": [0.9, 0.1]}))
    np.testing.assert_allclose(confidence(profile, [-1.0, 0.5, 2.0]), [0.9, 0.5, 0.1])


def test_uncalibrated_probability_method_uses_one_minus_uncertainty() -> None:
    np.testing.assert_allclose(confidence(make_profile(), [0.2, 0.7]), [0.8, 0.3])


def test_uncalibrated_raw_score_raises() -> None:
    profile = make_profile(method=MethodChoice(name="msp", family="baseline", access="logprobs", cost=1.0))
    with pytest.raises(ValueError, match="not calibrated"):
        confidence(profile, 3.2)


def test_load_rejects_a_file_that_was_edited_into_an_invalid_state(tmp_path: Path) -> None:
    path = save_profile(make_profile(), tmp_path / "m.json")
    data = json.loads(path.read_text())
    data["calibration_size"] = 200  # as large as the whole run
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="calibration_size"):
        load_profile(path)


def test_main_prints_a_summary_of_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = save_profile(make_profile(), tmp_path / "m.json")
    monkeypatch.setattr(sys, "argv", ["profile", str(path)])
    profile_module.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "m (org/m), schema v1"
    assert "p_true_zeroshot (self_eval, logprobs, cost 2)" in lines[1]
    assert "best AUROC" in lines[2]
    assert "0.800 [0.700, 0.900] on 100 calibration questions" in lines[3]
    assert lines[4].endswith("none")
