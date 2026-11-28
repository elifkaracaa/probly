"""Tests for common/reflexive.py with a fake tokenizer and model (no download needed)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from common import reflexive
import pytest
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class FakeTokenizer:
    """Knows only the strings in ``vocab``; every other string is split into 2 tokens."""

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [98, 99]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return FakeEncoding(input_ids=torch.zeros(1, 3, dtype=torch.long))


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeModel:
    """Returns the same next-token logits for every prompt."""

    device = "cpu"

    def __init__(self, logits):
        self.logits = torch.tensor(logits).view(1, 1, -1)

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


LOGITS = [2.0, 0.0, -1.0, -1.0]  # id 0 = "A", id 1 = "B"


def fakes(vocab: dict[str, int]) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Fake model and tokenizer, typed as the real ones for the type checker."""
    return cast("PreTrainedModel", FakeModel(LOGITS)), cast("PreTrainedTokenizerBase", FakeTokenizer(vocab))


def test_p_true_plain_ab():
    model, tok = fakes({"A": 0, "B": 1})
    d = reflexive.p_true_details(model, tok, "q", "a", mode="plain_ab")
    assert d["p_true"] == pytest.approx(1 / (1 + math.exp(-2.0)))  # sigmoid(2 - 0)
    assert 0.0 < d["coverage"] <= 1.0


@pytest.mark.parametrize("vocab", [{"B": 1}, {"A": 0}, {}])
def test_empty_group_raises(vocab):
    model, tok = fakes(vocab)
    with pytest.raises(ValueError, match="empty token group"):
        reflexive.p_true(model, tok, "q", "a", mode="plain_ab")


class FakeChatTokenizer(FakeTokenizer):
    """Fake tokenizer that also has a chat template (wraps the text in markers)."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "<chat>" + messages[0]["content"] + "</chat>"


def test_build_prompt_plain_ab_ends_with_an_open_bracket():
    _, tok = fakes({})
    prompt = reflexive.build_prompt(tok, "Q?", "Paris", mode="plain_ab")
    assert prompt == reflexive.build_text("Q?", "Paris") + " ("


def test_build_prompt_chat_groups_uses_the_chat_template():
    tok = cast("PreTrainedTokenizerBase", FakeChatTokenizer({}))
    prompt = reflexive.build_prompt(tok, "Q?", "Paris", mode="chat_groups")
    assert prompt == "<chat>" + reflexive.build_text("Q?", "Paris") + "</chat>"


def test_build_prompt_unknown_mode_raises():
    _, tok = fakes({})
    with pytest.raises(ValueError, match="unknown mode: nope"):
        reflexive.build_prompt(tok, "Q?", "Paris", mode="nope")


def test_build_text_adds_the_brainstorm_line_only_when_given():
    assert "brainstormed" not in reflexive.build_text("Q?", "Paris")
    assert "brainstormed" not in reflexive.build_text("Q?", "Paris", [])
    with_ideas = reflexive.build_text("Q?", "Paris", ["Lyon", "Nice"])
    assert "Here are some ideas that were brainstormed: Lyon, Nice\n" in with_ideas


def test_true_logit_is_the_logit_of_the_single_true_token():
    model = cast("PreTrainedModel", FakeModel(LOGITS))
    tok = cast("PreTrainedTokenizerBase", FakeChatTokenizer({"True": 1}))
    assert reflexive.true_logit(model, tok, "q", "a") == pytest.approx(0.0)  # LOGITS[1]
    tok = cast("PreTrainedTokenizerBase", FakeChatTokenizer({"True": 0}))
    assert reflexive.true_logit(model, tok, "q", "a") == pytest.approx(2.0)  # LOGITS[0]
