"""Tests for the TriviaQA loader and the dataset registry in common/data.py."""

from __future__ import annotations

import common.data
import pytest

# 20 fake TriviaQA items; each lists its canonical answer once more among the aliases.
FAKE_TRIVIA = [
    {"question": f"Question {i}?", "answer": {"value": f"Answer {i}", "aliases": [f"alias {i}", f"Answer {i}"]}}
    for i in range(20)
]


@pytest.fixture
def fake_trivia(monkeypatch: pytest.MonkeyPatch) -> None:
    # No download: load_dataset returns the 20 fake items above.
    monkeypatch.setattr(common.data, "load_dataset", lambda *_args, **_kwargs: FAKE_TRIVIA)


@pytest.mark.usefixtures("fake_trivia")
def test_trivia_loader_puts_canonical_answer_first_without_duplicates() -> None:
    rows = common.data.load_questions("triviaqa", 3, seed=0)

    assert [r["id"] for r in rows] == [0, 1, 2]
    for r in rows:
        number = r["question"].removeprefix("Question ").removesuffix("?")
        assert r["aliases"] == [f"Answer {number}", f"alias {number}"]


@pytest.mark.usefixtures("fake_trivia")
def test_trivia_loader_order_depends_only_on_the_seed() -> None:
    first = common.data.load_questions("triviaqa", 5, seed=1)
    again = common.data.load_questions("triviaqa", 5, seed=1)
    other = common.data.load_questions("triviaqa", 5, seed=2)

    assert first == again
    assert [r["question"] for r in first] != [r["question"] for r in other]


def test_unknown_dataset_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match=r"Unknown dataset 'squad'; known: .*triviaqa"):
        common.data.load_questions("squad", 3)


@pytest.mark.usefixtures("fake_trivia")
def test_smaller_set_is_a_prefix_of_the_larger_one() -> None:
    # The 200- and 1000-question caches rely on this (scripts/check_prefix.py).
    small = common.data.load_questions("triviaqa", 5, seed=0)
    large = common.data.load_questions("triviaqa", 12, seed=0)
    assert large[:5] == small
