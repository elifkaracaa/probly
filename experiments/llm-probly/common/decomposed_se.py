"""Decomposed semantic entropy through input clarification (B4).

Following Hou et al. (2024, "Decomposing Uncertainty for Large Language Models
through Input Clarification Ensembling", https://arxiv.org/abs/2311.08718):

1. The model rephrases the question ``C`` times (probly's ``HFQuestionClarifier``).
2. For every clarification the model answers ``K`` times.
3. All ``C * K`` answers are clustered by meaning together (one shared NLI matrix,
   A1), so every clarification gets a distribution over the same clusters.

With ``p_c`` the cluster distribution of clarification ``c`` and ``p`` their mean:

- **total** = ``H(p)``: how much all answers disagree.
- **aleatoric** = ``H(p) - mean_c H(p_c)`` (mutual information between answer and
  clarification): the answers change with the reading of the question, so the
  *question* is ambiguous.
- **epistemic** = ``mean_c H(p_c)``: the answers disagree even for one fixed
  reading, so the *model* does not know.

total = aleatoric + epistemic. :func:`decompose` computes them with probly's
``SecondOrderEntropyDecomposition`` (what ``quantify`` returns) on a sample of
categorical distributions, one per clarification. probly names the parts for a
*model* ensemble (conditional entropy = aleatoric, mutual information =
epistemic); here the ensemble members are *readings of the question*, so, as in
Hou et al., the mutual information is the ambiguity of the input (aleatoric)
and the conditional entropy is the model's own uncertainty (epistemic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from probly.quantification import SecondOrderEntropyDecomposition
from probly.representation.distribution.numpy_categorical import (
    NumpyCategoricalDistributionSample,
    NumpyProbabilityCategoricalDistribution,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

N_CLARIFICATIONS = 3
N_ANSWERS = 5
QUESTIONS_PER_CALL = 8
"""Questions per generation call (memory on a laptop)."""


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats of a probability vector (zeros ignored); used to check :func:`decompose`."""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def decompose(cluster_ids: Sequence[int], n_clarifications: int, n_answers: int) -> dict[str, float]:
    """Total, aleatoric and epistemic uncertainty from the clusters of all answers.

    Args:
        cluster_ids: Cluster of every answer, clarification by clarification:
            answers ``c * n_answers`` to ``(c + 1) * n_answers - 1`` belong to clarification ``c``.
        n_clarifications: Number of clarifications ``C``.
        n_answers: Answers per clarification ``K``.

    Returns:
        ``{"total", "aleatoric", "epistemic"}`` in nats; total = aleatoric + epistemic.

    Raises:
        ValueError: If the number of cluster ids is not ``C * K``.
    """
    ids = np.asarray(cluster_ids, dtype=int)
    if ids.shape != (n_clarifications * n_answers,):
        msg = f"expected {n_clarifications} x {n_answers} cluster ids, got {ids.size}."
        raise ValueError(msg)
    n_clusters = int(ids.max()) + 1
    per_clarification = np.stack(
        [np.bincount(row, minlength=n_clusters) / n_answers for row in ids.reshape(n_clarifications, n_answers)]
    )
    sample = NumpyCategoricalDistributionSample(
        array=NumpyProbabilityCategoricalDistribution(per_clarification), sample_axis=0
    )
    parts = SecondOrderEntropyDecomposition(sample)
    total, conditional, mutual = np.asarray([parts.total, parts.aleatoric, parts.epistemic], dtype=float)
    return {
        "total": float(total),
        "aleatoric": max(float(mutual), 0.0),  # mutual information: the question is ambiguous
        "epistemic": float(conditional),  # conditional entropy: the model does not know
    }


def clean_clarification(text: str, question: str) -> str:
    """First line of a generated clarification without quotes; the question itself if empty."""
    lines = [line.strip().strip("\"'") for line in str(text).strip().splitlines() if line.strip()]
    return lines[0] if lines and lines[0] else question


def clean_answer(text: str) -> str:
    """First line of a generated answer; a placeholder for an empty one, so NLI still sees a text."""
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    return lines[0] if lines else "(no answer)"


def clarify_and_answer(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    questions: Sequence[str],
    *,
    n_clarifications: int = N_CLARIFICATIONS,
    n_answers: int = N_ANSWERS,
    temperature: float = 0.7,
    max_new_tokens: int = 32,
) -> list[dict[str, Any]]:
    """Clarify every question and answer every clarification.

    Uses probly's ``HFQuestionClarifier`` and ``HFTextGenerationSampler`` with the
    benchmark's answer prompt (scripts/generate.py) and its stop-at-newline rule.

    Args:
        model: The LLM (on its device).
        tok: Its tokenizer.
        questions: Question texts.
        n_clarifications: Clarifications per question.
        n_answers: Answers per clarification.
        temperature: Sampling temperature of the answers (the clarifier always samples at 1.0).
        max_new_tokens: Token limit of an answer (clarifications get 48).

    Returns:
        One dict per question with ``clarifications`` (``C`` texts) and ``answers``
        (``C * K`` texts, clarification by clarification).
    """
    from scripts.generate import INSTRUCTION  # noqa: PLC0415

    from common.generation import stop_at_newline_config  # noqa: PLC0415  (loads transformers)
    from probly.representer.clarifier.huggingface import HFQuestionClarifier  # noqa: PLC0415
    from probly.representer.sampler.huggingface import HFTextGenerationSampler  # noqa: PLC0415

    generative = cast("Any", model)  # probly's samplers are typed for GenerativePreTrainedModel
    stop_config = stop_at_newline_config(model, tok)
    clarifier = HFQuestionClarifier(generative, tok, num_samples=n_clarifications, temperature=1.0, max_new_tokens=48)
    answerer = HFTextGenerationSampler(
        generative,
        tok,
        num_samples=n_answers,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_config=stop_config,
    )
    results = []
    for first in range(0, len(questions), QUESTIONS_PER_CALL):
        chunk = list(questions[first : first + QUESTIONS_PER_CALL])
        raw = clarifier.represent(chunk).tensor.text
        clarifications = [[clean_clarification(t, q) for t in raw[i]] for i, q in enumerate(chunk)]
        prompts = [f"{INSTRUCTION}\nQuestion: {c}" for cs in clarifications for c in cs]
        answers = answerer.represent(prompts).tensor.text
        for i, cs in enumerate(clarifications):
            rows = answers[i * n_clarifications : (i + 1) * n_clarifications]
            results.append(
                {
                    "clarifications": cs,
                    "answers": [clean_answer(a) for row in rows for a in row],
                }
            )
    return results
