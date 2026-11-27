"""Tests for the week-1 reference code in scripts/r3_methods.py (reply rule, one verbalized implementation)."""

from __future__ import annotations

from common import verbalized
import pytest
from scripts import r3_methods


@pytest.mark.parametrize(
    ("reply", "true", "false"),
    [
        ("True", True, False),
        (" (A) True", True, False),
        ("A.", True, False),
        ("false", False, True),
        ("(B)", False, True),
        ("Answer", False, False),  # starts with "a" but is a word, not the letter A
        ("maybe", False, False),
    ],
)
def test_week1_reply_rule(reply, true, false):
    assert r3_methods.is_true(reply) is true
    assert r3_methods.is_false(reply) is false


def test_verbalized_is_the_shared_implementation():
    assert r3_methods.verbalized_2s is verbalized.verbalized_2s
    assert r3_methods.parse_confidence is verbalized.parse_confidence
    assert r3_methods.ASK is verbalized.ASK


def test_prompt_asks_for_a_or_b():
    text = r3_methods.PT_PROMPT.format(q="Q?", a="Paris")
    assert text.startswith("Question: Q?\nPossible answer: Paris\n")
    assert text.endswith("The possible answer is:")
