"""Tests for Hugging Face self-evaluation representers."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")
import torch

from probly.representer.self_evaluation.huggingface import (
    HFSelfEvaluator,
    build_self_evaluation_text,
    parse_judgment,
)

VOCAB = 101


def char_id(char: str) -> int:
    """Map a character to a token id in 1..100 (0 is the pad id)."""
    return ord(char) % 100 + 1


class FakeTokenizer:
    """Character-level tokenizer that pads on the side given by ``padding_side``."""

    pad_token_id = 0
    pad_token = "<pad>"  # noqa: S105
    eos_token = "<eos>"  # noqa: S105

    def __init__(self, padding_side: str = "right", multi_token_options: bool = False) -> None:
        """Initialize the fake tokenizer."""
        self.padding_side = padding_side
        self.multi_token_options = multi_token_options
        self.template_calls = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002
        if self.multi_token_options:
            return [char_id(c) for c in text + "!!"]
        return [char_id(c) for c in text]

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.template_calls += 1
        return "<chat>" + messages[0]["content"]

    def __call__(
        self,
        texts: list[str],
        return_tensors: str,
        padding: bool,
        add_special_tokens: bool,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is True
        ids = [[char_id(c) for c in text] for text in texts]
        width = max(len(row) for row in ids)
        input_ids, masks = [], []
        for row in ids:
            pad = [0] * (width - len(row))
            mask_pad = [0] * (width - len(row))
            if self.padding_side == "left":
                input_ids.append(pad + row)
                masks.append(mask_pad + [1] * len(row))
            else:
                input_ids.append(row + pad)
                masks.append([1] * len(row) + mask_pad)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(masks)}


class FakeModel:
    """Model whose last-token logits depend on the tokens, on position ids and on the attention mask."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        """Initialize random but fixed embeddings."""
        generator = torch.Generator().manual_seed(0)
        self.token_table = torch.randn(VOCAB, VOCAB, generator=generator)
        self.position_table = torch.randn(512, VOCAB, generator=generator)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> SimpleNamespace:
        mask = attention_mask.unsqueeze(-1).float()
        token = self.token_table[input_ids]
        context = (token * mask).cumsum(dim=1) / mask.cumsum(dim=1).clamp(min=1)
        logits = token + self.position_table[position_ids] + context
        return SimpleNamespace(logits=logits)


QUESTIONS = [
    "Who wrote Hamlet?",
    "In which year did the Berlin Wall fall, and what happened next?",
    "Capital of France?",
]
ANSWERS = ["Shakespeare", "1989", "Paris is the capital of France"]


def test_prompt_text_without_and_with_brainstorm() -> None:
    plain = build_self_evaluation_text("Q?", "A1")
    assert plain == (
        "Question: Q?\nPossible answer: A1\nIs the possible answer:\n (A) True\n (B) False\nThe possible answer is:"
    )
    ideas = build_self_evaluation_text("Q?", "A1", ["x", "y"])
    assert "Here are some ideas that were brainstormed: x, y\n" in ideas
    assert ideas.index("brainstormed") < ideas.index("Possible answer")


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("mode", ["plain_ab", "chat_groups"])
def test_batched_scores_equal_single_scores(padding_side: str, mode: str) -> None:
    tokenizer = FakeTokenizer(padding_side=padding_side)
    model = FakeModel()
    brainstorm = [["Marlowe", "Bacon"], None, ["Lyon"]]

    batched = HFSelfEvaluator(model, tokenizer, mode=mode, batch_size=3).evaluate(QUESTIONS, ANSWERS, brainstorm)
    single = HFSelfEvaluator(model, tokenizer, mode=mode, batch_size=1).evaluate(QUESTIONS, ANSWERS, brainstorm)

    torch.testing.assert_close(torch.tensor(batched.p_true), torch.tensor(single.p_true), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.tensor(batched.coverage), torch.tensor(single.coverage), atol=1e-6, rtol=1e-6)


def test_batch_size_does_not_change_scores() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    full = HFSelfEvaluator(model, tokenizer, batch_size=8).evaluate(QUESTIONS, ANSWERS)
    chunked = HFSelfEvaluator(model, tokenizer, batch_size=2).evaluate(QUESTIONS, ANSWERS)
    torch.testing.assert_close(torch.tensor(full.p_true), torch.tensor(chunked.p_true), atol=1e-6, rtol=1e-6)


def test_p_true_matches_manual_computation() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    evaluator = HFSelfEvaluator(model, tokenizer, mode="plain_ab", batch_size=1)
    result = evaluator.evaluate(QUESTIONS[:1], ANSWERS[:1])

    text = build_self_evaluation_text(QUESTIONS[0], ANSWERS[0]) + " ("
    ids = torch.tensor([[char_id(c) for c in text]])
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    logits = model(ids, torch.ones_like(ids), positions).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)
    log_true, log_false = log_probs[char_id("A")], log_probs[char_id("B")]
    expected = torch.sigmoid(log_true - log_false).item()

    assert result.p_true[0] == pytest.approx(expected, abs=1e-6)
    assert result.coverage[0] == pytest.approx((log_true.exp() + log_false.exp()).item(), abs=1e-6)
    assert 0.0 <= result.p_true[0] <= 1.0
    assert 0.0 <= result.coverage[0] <= 1.0
    assert result.mode == "plain_ab"


def test_chat_groups_uses_the_chat_template() -> None:
    evaluator = HFSelfEvaluator(FakeModel(), FakeTokenizer(), mode="chat_groups")
    evaluator.evaluate(QUESTIONS, ANSWERS)
    # The evaluator works on its own copy of the tokenizer.
    assert evaluator.tokenizer.template_calls == len(QUESTIONS)


def test_original_tokenizer_is_not_modified() -> None:
    tokenizer = FakeTokenizer(padding_side="right")
    evaluator = HFSelfEvaluator(FakeModel(), tokenizer)
    assert tokenizer.padding_side == "right"
    assert evaluator.tokenizer.padding_side == "left"


def test_empty_inputs_give_empty_result() -> None:
    result = HFSelfEvaluator(FakeModel(), FakeTokenizer()).evaluate([], [])
    assert result.p_true.shape == (0,)
    assert result.coverage.shape == (0,)


def test_invalid_arguments_raise() -> None:
    model, tokenizer = FakeModel(), FakeTokenizer()
    with pytest.raises(ValueError, match="Unknown mode"):
        HFSelfEvaluator(model, tokenizer, mode="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch_size"):
        HFSelfEvaluator(model, tokenizer, batch_size=0)
    with pytest.raises(ValueError, match="Empty token group"):
        HFSelfEvaluator(model, FakeTokenizer(multi_token_options=True))
    evaluator = HFSelfEvaluator(model, tokenizer)
    with pytest.raises(ValueError, match="answers"):
        evaluator.evaluate(["q1", "q2"], ["a1"])
    with pytest.raises(ValueError, match="brainstorm"):
        evaluator.evaluate(["q1"], ["a1"], [["x"], ["y"]])


def _old_is_true(text: str) -> bool:
    text = text.strip().lower()
    return text.startswith("true") or re.match(r"^\(?a\)?(\W|$)", text) is not None


def _old_is_false(text: str) -> bool:
    text = text.strip().lower()
    return text.startswith("false") or re.match(r"^\(?b\)?(\W|$)", text) is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("True", True),
        (" true.", True),
        ("A", True),
        ("(A)", True),
        ("A) True", True),
        ("(True)", True),
        ("**True**", True),
        ('"A"', True),
        ("False", False),
        ("B.", False),
        ("(B)", False),
        ("**False**", False),
        ("Apple", None),
        ("Banana", None),
        ("", None),
        ("   ", None),
        ("maybe", None),
        ("1", None),
    ],
)
def test_parse_judgment(text: str, expected: bool | None) -> None:
    assert parse_judgment(text) is expected


@pytest.mark.parametrize(
    "text",
    ["True", "true!", "A", "A.", "(A)", "A)", "False", "B", "(B)", "B:", "Apple", "Bob", "xyz", ""],
)
def test_parse_judgment_agrees_with_week_1_rule_where_it_read_a_reply(text: str) -> None:
    # Replies that the old rule could not read may be readable now, so only the readable ones are compared.
    if _old_is_true(text):
        assert parse_judgment(text) is True
    elif _old_is_false(text):
        assert parse_judgment(text) is False


class RecordingSampler:
    """Stands in for HFTextGenerationSampler and records how it is used."""

    instances: list[RecordingSampler] = []  # noqa: RUF012

    def __init__(self, model: object, tokenizer: object, **kwargs: object) -> None:
        """Record the constructor arguments."""
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.calls: list[list[str]] = []
        RecordingSampler.instances.append(self)

    def __call__(self, prompts: list[str]) -> SimpleNamespace:
        self.calls.append(list(prompts))
        num_samples = self.kwargs["num_samples"]
        assert isinstance(num_samples, int)
        # The replies carry the length of their prompt, so a test can check that they are assigned correctly.
        text = np.array([[f"{len(prompt)}-{i}" for i in range(num_samples)] for prompt in prompts], dtype=object)
        return SimpleNamespace(tensor=SimpleNamespace(text=text))


@pytest.fixture
def recording_sampler(monkeypatch: pytest.MonkeyPatch) -> type[RecordingSampler]:
    RecordingSampler.instances = []
    monkeypatch.setattr("probly.representer.self_evaluation.huggingface.HFTextGenerationSampler", RecordingSampler)
    return RecordingSampler


def test_sample_judgments_batches_questions_in_one_sampler(recording_sampler: type[RecordingSampler]) -> None:
    evaluator = HFSelfEvaluator(FakeModel(), FakeTokenizer(), batch_size=8)
    brainstorm = [["Marlowe"], None, None]
    replies = evaluator.sample_judgments(
        QUESTIONS, ANSWERS, brainstorm, num_samples=4, temperature=0.7, max_new_tokens=5, question_batch_size=2
    )

    prompts = [
        build_self_evaluation_text(question, answer, idea)
        for question, answer, idea in zip(QUESTIONS, ANSWERS, brainstorm, strict=True)
    ]
    # Replies come back in the order of the questions, although the batches are sorted by prompt length.
    assert replies == [[f"{len(prompt)}-{i}" for i in range(4)] for prompt in prompts]
    (sampler,) = recording_sampler.instances
    assert sampler.kwargs == {
        "num_samples": 4,
        "temperature": 0.7,
        "max_new_tokens": 5,
        "with_log_likelihood": False,
    }
    assert [len(call) for call in sampler.calls] == [2, 1]
    # The batches are sorted by prompt length: shortest two first, the longest prompt (with brainstorm) last.
    assert sorted(prompts, key=len) == [*sampler.calls[0], *sampler.calls[1]]
    assert len(prompts[0]) == max(len(prompt) for prompt in prompts)


def test_sample_judgments_defaults_to_the_evaluator_batch_size(recording_sampler: type[RecordingSampler]) -> None:
    evaluator = HFSelfEvaluator(FakeModel(), FakeTokenizer(), batch_size=2)
    evaluator.sample_judgments(QUESTIONS, ANSWERS, num_samples=3)
    assert [len(call) for call in recording_sampler.instances[0].calls] == [2, 1]


def test_sample_judgments_invalid_arguments_raise() -> None:
    evaluator = HFSelfEvaluator(FakeModel(), FakeTokenizer())
    with pytest.raises(ValueError, match="answers"):
        evaluator.sample_judgments(["q1", "q2"], ["a1"])
    with pytest.raises(ValueError, match="brainstorm"):
        evaluator.sample_judgments(["q1"], ["a1"], [["x"], ["y"]])
    with pytest.raises(ValueError, match="question_batch_size"):
        evaluator.sample_judgments(["q1"], ["a1"], question_batch_size=0)
