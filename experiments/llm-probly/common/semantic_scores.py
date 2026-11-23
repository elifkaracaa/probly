"""One call from sampled answers to every semantic uncertainty score (B5).

The benchmark plugins read a cache of 200 or 1000 questions. This module is for
a user who has their own question and a handful of sampled answers and wants
the scores now::

    from common.semantic_scores import semantic_scores

    scores = semantic_scores(
        ["Paris", "Paris", "It is Paris.", "Lyon"],
        sample_log_likelihoods=[-0.1, -0.2, -0.4, -2.3],  # optional, for semantic entropy
    )
    scores["semantic_entropy"], scores["num_semantic_sets"]

Every score is an uncertainty: higher means less consistent samples. One NLI
matrix is computed per question (A1) and shared by all scores, so the extra
scores cost no extra NLI call. See docs/semantic_methods.md for what each
score means and a runnable example.
"""

from __future__ import annotations

from itertools import combinations
import logging
import math
import time
from typing import TYPE_CHECKING

from methods.lexical_similarity import rouge_l

from common.graph_measures import graph_scores
from common.nli_matrix import QUESTIONS_PER_BATCH, compute_matrices, sample_block, se_and_dse
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import torch

    from probly.representer.semantic_clustering.huggingface import HFSemanticClusterer

log = logging.getLogger(__name__)

SCORES = (
    "semantic_entropy",
    "discrete_semantic_entropy",
    "lexical_similarity",
    "num_semantic_sets",
    "eig_val_laplacian",
    "deg_mat",
    "eccentricity",
)
"""Keys of the dict returned by :func:`semantic_scores`, all uncertainties (higher = less consistent)."""

MIN_SAMPLES = 2


class SemanticInputError(ValueError):
    """The samples or log-likelihoods given to :func:`semantic_scores` cannot be scored."""


def check_inputs(
    samples: Sequence[str],
    sample_log_likelihoods: Sequence[float] | None = None,
    where: str = "",
) -> None:
    """Raise a :class:`SemanticInputError` that says what is wrong and how to fix it.

    Args:
        samples: Sampled answers to one question.
        sample_log_likelihoods: Optional log-likelihood of every sample.
        where: Prefix for the message, e.g. ``"question 3: "``.

    Raises:
        SemanticInputError: If the inputs cannot be scored.
    """
    if isinstance(samples, str):
        msg = f"{where}samples must be a list of answers, got one string. Wrap it: [{samples!r}, ...]."
        raise SemanticInputError(msg)
    samples = list(samples)
    if len(samples) < MIN_SAMPLES:
        msg = (
            f"{where}need at least {MIN_SAMPLES} sampled answers to measure agreement, got {len(samples)}. "
            "Sample the model several times (do_sample=True, e.g. 10 samples)."
        )
        raise SemanticInputError(msg)
    for i, sample in enumerate(samples):
        if not isinstance(sample, str):
            msg = f"{where}sample {i} is {type(sample).__name__}, not text. Decode the generated tokens first."
            raise SemanticInputError(msg)
        if not sample.strip():
            msg = (
                f"{where}sample {i} is empty. Small models sometimes stop at once; "
                "drop empty samples or allow more new tokens."
            )
            raise SemanticInputError(msg)
    if sample_log_likelihoods is None:
        return
    lls = list(sample_log_likelihoods)
    if len(lls) != len(samples):
        msg = (
            f"{where}got {len(samples)} samples but {len(lls)} log-likelihoods; "
            "give one log-likelihood per sample, in the same order."
        )
        raise SemanticInputError(msg)
    for i, value in enumerate(lls):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            msg = f"{where}log-likelihood {i} is {value!r}; it must be a finite number (e.g. the mean token log-prob)."
            raise SemanticInputError(msg)


def lexical_uncertainty(samples: Sequence[str]) -> float:
    """Minus the mean pairwise ROUGE-L of the samples (same as the lexical_similarity plugin)."""
    pairs = list(combinations(samples, 2))
    return -sum(rouge_l(a, b) for a, b in pairs) / len(pairs)


def scores_from_matrix(
    matrix: torch.Tensor,
    samples: Sequence[str],
    sample_log_likelihoods: Sequence[float] | None = None,
) -> dict[str, float]:
    """All semantic scores of one question from its NLI matrix.

    Args:
        matrix: Question matrix ``(K+1, K+1, 3)`` (index 0 = the answer being judged).
        samples: The ``K`` sampled answers, same order as the matrix.
        sample_log_likelihoods: Optional; without them semantic entropy is NaN.

    Returns:
        One float per name in :data:`SCORES`.
    """
    k = len(samples)
    if sample_log_likelihoods is None:
        se, dse = math.nan, se_and_dse(matrix, [0.0] * k)[1]
    else:
        se, dse = se_and_dse(matrix, list(sample_log_likelihoods))
    return {
        "semantic_entropy": float(se),
        "discrete_semantic_entropy": float(dse),
        "lexical_similarity": lexical_uncertainty(samples),
        **graph_scores(matrix),
    }


def cluster_ids(matrix: torch.Tensor) -> list[int]:
    """Semantic cluster of every sample (the clusters semantic entropy uses)."""
    return HFGreedySemanticClusterer.cluster_nli_matrix(sample_block(matrix)).tolist()


_clusterer: HFSemanticClusterer | None = None


def default_clusterer() -> HFSemanticClusterer:
    """The benchmark's NLI clusterer, loaded once (first call downloads ~0.5 GB)."""
    global _clusterer  # noqa: PLW0603
    if _clusterer is None:
        from methods._se_common import NLI_MODEL, get_clusterer  # noqa: PLC0415  (loads torch models)

        log.info("loading the NLI model %s (first call only)", NLI_MODEL)
        start = time.perf_counter()
        _clusterer = get_clusterer()
        log.info("NLI model ready in %.1f s", time.perf_counter() - start)
    return _clusterer


def semantic_scores_many(
    items: Iterable[dict],
    *,
    clusterer: HFSemanticClusterer | None = None,
    log_every: int = QUESTIONS_PER_BATCH,
) -> list[dict]:
    """Score many questions; the NLI pairs of several questions share model batches.

    Args:
        items: One dict per question with ``samples`` (list of answers) and
            optionally ``sample_log_likelihoods``, ``answer`` (the answer being
            judged; default: the first sample) and ``question``.
        clusterer: NLI clusterer; default: the benchmark's ``microsoft/deberta-base-mnli``.
        log_every: Log progress after this many questions.

    Returns:
        One dict per question: the scores of :data:`SCORES`, ``clusters`` (cluster
        id per sample) and ``seconds`` (NLI + scoring time of its batch, per question).

    Raises:
        SemanticInputError: If a question's inputs cannot be scored; the message names the question.
    """
    items = list(items)
    rows = []
    for i, item in enumerate(items):
        if "samples" not in item:
            msg = f"question {i}: missing 'samples' (the list of sampled answers)."
            raise SemanticInputError(msg)
        check_inputs(item["samples"], item.get("sample_log_likelihoods"), where=f"question {i}: ")
        samples = list(item["samples"])
        rows.append({"greedy_answer": item.get("answer") or samples[0], "samples": samples})

    nli = clusterer or default_clusterer()
    results: list[dict] = []
    start = time.perf_counter()
    for first in range(0, len(rows), log_every):
        batch_start = time.perf_counter()
        chunk = rows[first : first + log_every]
        matrices = compute_matrices(nli, chunk)
        per_question = (time.perf_counter() - batch_start) / len(chunk)
        for offset, (row, matrix) in enumerate(zip(chunk, matrices, strict=True)):
            item = items[first + offset]
            scores = scores_from_matrix(matrix, row["samples"], item.get("sample_log_likelihoods"))
            results.append({**scores, "clusters": cluster_ids(matrix), "seconds": per_question})
        done = len(results)
        elapsed = time.perf_counter() - start
        log.info(
            "scored %d/%d questions (%.2f s per question, about %.0f s left)",
            done,
            len(rows),
            elapsed / done,
            elapsed / done * (len(rows) - done),
        )
    if any(item.get("sample_log_likelihoods") is None for item in items):
        log.info("semantic_entropy is NaN where no sample_log_likelihoods were given; the other scores need text only")
    return results


def semantic_scores(
    samples: Sequence[str],
    *,
    sample_log_likelihoods: Sequence[float] | None = None,
    answer: str | None = None,
    clusterer: HFSemanticClusterer | None = None,
) -> dict:
    """Every semantic uncertainty score of one question.

    Args:
        samples: Sampled answers to the question (at least 2; 10 in the benchmark).
        sample_log_likelihoods: Optional mean token log-likelihood of every
            sample; needed only for ``semantic_entropy`` (NaN without them).
        answer: The answer being judged (e.g. the greedy answer). Default: the
            first sample. The scores themselves compare the samples only.
        clusterer: NLI clusterer; default: ``microsoft/deberta-base-mnli``, loaded once.

    Returns:
        Dict with the scores of :data:`SCORES` (higher = more uncertain),
        ``clusters`` (cluster id of every sample) and ``seconds``.

    Raises:
        SemanticInputError: If the inputs cannot be scored; the message says how to fix them.
    """
    item = {"samples": samples, "sample_log_likelihoods": sample_log_likelihoods, "answer": answer}
    try:
        return semantic_scores_many([item], clusterer=clusterer, log_every=1)[0]
    except SemanticInputError as error:
        raise SemanticInputError(str(error).removeprefix("question 0: ")) from None
