"""Tests for the verbalized-confidence and empirical p(True) plugins, with stubs instead of a model.

No model is loaded: the sampler and the evaluator are replaced by stubs that return fixed replies, so the tests
check what the code adds on top of the library (prompt, reply parsing, score contract, failed rows, result fields).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from common import verbalized
from methods import p_true_empirical, verbalized_2s
import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

MODEL = cast("PreTrainedModel", object())
TOKENIZER = cast("PreTrainedTokenizerBase", object())
FAKE_LM = cast(
    "PreTrainedModel", SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
)  # the empirical plugin stores both


def make_rows(n: int = 3) -> list[dict[str, Any]]:
    """Return ``n`` cached rows."""
    return [{"id": i, "model": "fake-model", "question": f"Q{i}?", "greedy_answer": f"A{i}"} for i in range(n)]


class StubSampler:
    """Stand-in for HFTextGenerationSampler: always replies with ``reply`` and records how it was used."""

    reply = ""
    instances: list[StubSampler] = []  # noqa: RUF012

    def __init__(self, model: object, tok: object, **kwargs: Any) -> None:  # noqa: ANN401
        self.model = model
        self.tok = tok
        self.kwargs = kwargs
        self.chats: list = []
        StubSampler.instances.append(self)

    def __call__(self, chats: list) -> SimpleNamespace:
        self.chats = chats
        return SimpleNamespace(tensor=SimpleNamespace(text=np.array([[StubSampler.reply]], dtype=object)))


@pytest.fixture
def stub_sampler(monkeypatch):
    StubSampler.instances = []
    monkeypatch.setattr(verbalized, "HFTextGenerationSampler", StubSampler)
    return StubSampler


def test_verbalized_2s_reads_the_number_in_the_reply(stub_sampler):
    stub_sampler.reply = "85"
    confidence, failed, reply = verbalized.verbalized_2s(MODEL, TOKENIZER, "Q?", "A")
    assert confidence == pytest.approx(0.85)
    assert failed is False
    assert reply == "85"


def test_verbalized_2s_unreadable_reply_is_failed_with_confidence_zero(stub_sampler):
    stub_sampler.reply = "I am not sure"
    confidence, failed, reply = verbalized.verbalized_2s(MODEL, TOKENIZER, "Q?", "A")
    assert confidence == 0.0
    assert failed is True
    assert reply == "I am not sure"


def test_verbalized_2s_asks_in_a_chat_and_decodes_greedily(stub_sampler):
    stub_sampler.reply = "7"
    verbalized.verbalized_2s(MODEL, TOKENIZER, "Q?", "A", scale=10)
    sampler = stub_sampler.instances[0]
    assert sampler.chats == [
        [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": verbalized.ASK[10]},
        ]
    ]
    assert sampler.kwargs["do_sample"] is False
    assert sampler.kwargs["num_samples"] == 1


def test_verbalized_2s_ask_replaces_the_default_question(stub_sampler):
    stub_sampler.reply = "3"
    confidence, _, _ = verbalized.verbalized_2s(MODEL, TOKENIZER, "Q?", "A", scale=10, ask=verbalized.ASK_ANCHORED)
    assert confidence == pytest.approx(0.3)
    assert stub_sampler.instances[0].chats[0][-1]["content"] == verbalized.ASK_ANCHORED


def test_verbalized_plugin_needs_the_model():
    with pytest.raises(ValueError, match="needs the model"):
        verbalized_2s.score(make_rows(), model=MODEL)


def test_verbalized_plugin_uncertainty_is_one_minus_confidence(monkeypatch):
    replies = {"Q0?": (0.9, False, "90"), "Q1?": (0.0, True, "no idea"), "Q2?": (0.0, False, "0")}
    calls = []

    def fake_verbalized_2s(model, tok, question, answer, scale):
        calls.append((model, tok, question, answer, scale))
        return replies[question]

    monkeypatch.setattr(verbalized_2s, "verbalized_2s", fake_verbalized_2s)
    result = verbalized_2s.score(make_rows(), model=MODEL, tok=TOKENIZER)
    assert result["uncertainty"] == pytest.approx([0.1, 1.0, 1.0])
    assert result["failed"] == [False, True, False]  # a stated 0 is a valid answer, not a failure
    assert result["replies"] == ["90", "no idea", "0"]
    assert result["ids"] == [0, 1, 2]
    assert result["method"] == "verbalized_2s"
    assert result["model"] == "fake-model"
    assert result["scale"] == 100
    assert result["probability"] is True
    assert result["model_calls_per_question"] == 1
    assert result["seconds_per_question"] >= 0.0
    assert calls[0] == (MODEL, TOKENIZER, "Q0?", "A0", 100)


class StubJudge:
    """Stand-in for HFSelfEvaluator that returns the given replies from ``sample_judgments``."""

    def __init__(self, replies: list[list[str]]) -> None:
        self.replies = replies
        self.init_kwargs: dict[str, Any] = {}
        self.call: tuple = ()

    def factory(self, model: object, tok: object, **kwargs: Any) -> StubJudge:  # noqa: ANN401
        self.init_kwargs = {"model": model, "tok": tok, **kwargs}
        return self

    def sample_judgments(self, questions: list[str], answers: list[str], num_samples: int) -> list[list[str]]:
        self.call = (questions, answers, num_samples)
        return self.replies


def test_empirical_plugin_needs_the_model():
    with pytest.raises(ValueError, match="needs the model"):
        p_true_empirical.score(make_rows(), tok=TOKENIZER)


def test_empirical_p_true_is_the_share_of_true_judgments(monkeypatch):
    replies = [
        ["True"] * 8 + ["False"] * 2,  # p = 0.8, uncertainty 0.2
        ["(A)", "(B)", "B", "maybe"],  # p = 0.25 of 4, one reply unreadable
        ["False"] * 10,  # p = 0, a valid answer
    ]
    stub = StubJudge(replies)
    monkeypatch.setattr(p_true_empirical, "HFSelfEvaluator", stub.factory)
    result = p_true_empirical.score(make_rows(), model=FAKE_LM, tok=TOKENIZER)
    assert result["uncertainty"] == pytest.approx([0.2, 0.75, 1.0])
    assert result["failed"] == [False, False, False]
    assert result["parsed_fraction"] == pytest.approx([1.0, 0.75, 1.0])
    assert stub.call == (["Q0?", "Q1?", "Q2?"], ["A0", "A1", "A2"], 10)
    assert stub.init_kwargs == {"model": FAKE_LM, "tok": TOKENIZER, "batch_size": 1}
    assert result["method"] == "p_true_empirical"
    assert result["model"] == "fake-model"
    assert result["probability"] is True
    assert result["model_calls_per_question"] == 10
    assert result["implementation"] == "HFSelfEvaluator.sample_judgments"
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["ids"] == [0, 1, 2]
    assert result["seconds_per_question"] >= 0.0


def test_empirical_no_readable_reply_is_failed_with_uncertainty_one(monkeypatch):
    stub = StubJudge([["maybe", "", "I think so"], ["True", "True"]])
    monkeypatch.setattr(p_true_empirical, "HFSelfEvaluator", stub.factory)
    result = p_true_empirical.score(make_rows(2), model=FAKE_LM, tok=TOKENIZER)
    assert result["uncertainty"] == pytest.approx([1.0, 0.0])
    assert result["failed"] == [True, False]
    assert result["parsed_fraction"] == pytest.approx([0.0, 1.0])
