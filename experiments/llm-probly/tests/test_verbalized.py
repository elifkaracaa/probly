"""Tests for the reply parsing of verbalized confidence (common/verbalized.py)."""

from __future__ import annotations

from common.verbalized import parse_confidence
import pytest


@pytest.mark.parametrize(
    ("reply", "scale", "expected"),
    [
        ("85", 100, 0.85),
        ("I am 90% sure", 100, 0.9),
        ("0.99", 100, 0.99),
        ("7", 10, 0.7),
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
