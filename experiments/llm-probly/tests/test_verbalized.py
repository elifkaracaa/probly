"""Tests for the reply parsing of verbalized confidence (common/verbalized.py)."""

from __future__ import annotations

from common.verbalized import ASK, ASK_ANCHORED, parse_confidence
import pytest


@pytest.mark.parametrize(
    ("reply", "scale", "expected"),
    [
        ("85", 100, 0.85),
        ("I am 90% sure", 100, 0.9),
        ("0.99", 100, 0.99),
        ("7", 10, 0.7),
        ("1.0", 10, 0.1),  # on 0-10 a decimal is not a fraction (Eylul's review of #61)
        ("0.5", 10, 0.05),
        ("7.5", 10, 0.75),
        ("12", 10, 1.0),
        ("250", 100, 1.0),
        ("no idea", 100, None),
        ("", 100, None),
    ],
)
def test_parse_confidence(reply, scale, expected):
    result = parse_confidence(reply, scale)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_anchored_prompt_explains_both_ends_of_the_scale():
    assert "0 to 10" in ASK_ANCHORED
    assert "certainly wrong" in ASK_ANCHORED
    assert "certainly correct" in ASK_ANCHORED
    assert ASK_ANCHORED not in ASK.values()
