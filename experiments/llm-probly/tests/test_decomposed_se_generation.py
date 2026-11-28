"""Tests for the generation step of decomposed semantic entropy (clarify_and_answer), without a model.

probly's clarifier and sampler are replaced by fakes that return fixed texts, so the test checks
what clarify_and_answer does around them: chunking, the prompts it builds, cleaning and ordering.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from common.decomposed_se import clarify_and_answer


def test_clarify_and_answer_with_fake_generators(monkeypatch):
    clarified: list[list[str]] = []
    prompts_seen: list[list[str]] = []
    settings: dict[str, dict[str, Any]] = {}

    class FakeClarifier:
        def __init__(self, _model, _tok, num_samples, **kwargs):
            self.n = num_samples
            settings["clarifier"] = {"num_samples": num_samples, **kwargs}

        def represent(self, questions):
            clarified.append(list(questions))
            # the first reading comes back empty, the others in quotes
            text = [[f'"{q} (reading {c})"' if c else "" for c in range(self.n)] for q in questions]
            return SimpleNamespace(tensor=SimpleNamespace(text=text))

    class FakeSampler:
        def __init__(self, _model, _tok, num_samples, **kwargs):
            self.n = num_samples
            settings["sampler"] = {"num_samples": num_samples, **kwargs}

        def represent(self, prompts):
            prompts_seen.append(list(prompts))
            # the first answer comes back empty, the others keep talking on a second line
            text = [[f"answer {k}\nmore text" if k else "" for k in range(self.n)] for _ in prompts]
            return SimpleNamespace(tensor=SimpleNamespace(text=text))

    monkeypatch.setattr("probly.representer.clarifier.huggingface.HFQuestionClarifier", FakeClarifier)
    monkeypatch.setattr("probly.representer.sampler.huggingface.HFTextGenerationSampler", FakeSampler)
    monkeypatch.setattr("common.generation.stop_at_newline_config", lambda _model, _tok: "stop-at-newline")
    monkeypatch.setattr("common.decomposed_se.QUESTIONS_PER_CALL", 2)
    model, tok = cast("Any", object()), cast("Any", object())  # only passed on to the fakes

    results = clarify_and_answer(model, tok, ["Q1?", "Q2?", "Q3?"], n_clarifications=3, n_answers=2)

    assert clarified == [["Q1?", "Q2?"], ["Q3?"]]  # at most QUESTIONS_PER_CALL questions per call
    assert settings["clarifier"] == {"num_samples": 3, "temperature": 1.0, "max_new_tokens": 48}
    assert settings["sampler"] == {
        "num_samples": 2,
        "temperature": 0.7,
        "max_new_tokens": 32,
        "generation_config": "stop-at-newline",
    }
    # one prompt per clarification: the benchmark's instruction line, then the clarified question
    first_call = prompts_seen[0]
    assert len(first_call) == 2 * 3
    assert len({prompt.splitlines()[0] for prompt in first_call}) == 1
    assert [prompt.splitlines()[-1] for prompt in first_call[:3]] == [
        "Question: Q1?",
        "Question: Q1? (reading 1)",
        "Question: Q1? (reading 2)",
    ]
    assert len(results) == 3
    assert results[2]["clarifications"] == ["Q3?", "Q3? (reading 1)", "Q3? (reading 2)"]  # empty -> the question
    assert results[2]["answers"] == ["(no answer)", "answer 1"] * 3  # first line only; empty -> placeholder
