"""Hybrid scores: a confidence signal times the consistency of the samples (C5, CoCoA).

Confidence methods (MSP, p(True)) look at one answer only: how sure the model is about
*this* answer. Agreement methods look only at whether the samples agree with each other.
CoCoA (Vashurin et al. 2025, "Uncertainty Quantification for LLMs through Minimum Bayes
Risk: Bridging Confidence and Consistency", https://arxiv.org/abs/2502.04964) multiplies
the two::

    uncertainty = u(greedy) * (1 - mean_i s(greedy, sample_i))

``u`` is the confidence-based uncertainty of the greedy answer (``-log P(answer)`` for
MSP, ``1 - p(True)`` for p(True)) and ``s`` is the similarity of the greedy answer to each
of the K samples. A confident answer that the samples contradict, and an answer that the
samples repeat but the model finds unlikely, both get a high score.

The similarity is read from the shared per-question NLI matrix (A1/A5), so a hybrid score
needs no extra NLI call: ``s(a, b)`` is the mean of the entailment probabilities ``a -> b``
and ``b -> a``. The paper uses a cross-encoder similarity instead; NLI entailment keeps the
score on the matrix every semantic method already reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.nli_matrix import GREEDY_INDEX
from probly.representer.semantic_clustering.huggingface import NLI_LABELS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

ENTAILMENT = NLI_LABELS.index("entailment")


def greedy_similarities(matrix: torch.Tensor) -> list[float]:
    """Similarity of the greedy answer to each sample, from one question matrix.

    Args:
        matrix: Question matrix of shape ``(K+1, K+1, 3)`` with NLI probabilities in probly's
            label order, greedy answer first.

    Returns:
        ``K`` values in ``[0, 1]``: the mean of the entailment probabilities greedy -> sample
        and sample -> greedy.

    Raises:
        ValueError: If the matrix has no samples or the wrong shape.
    """
    if matrix.ndim != 3 or matrix.shape[0] != matrix.shape[1] or matrix.shape[2] != len(NLI_LABELS):
        msg = f"expected a (K+1, K+1, {len(NLI_LABELS)}) matrix, got shape {tuple(matrix.shape)}."
        raise ValueError(msg)
    if matrix.shape[0] < 2:
        msg = "the matrix has no samples (only the greedy answer)."
        raise ValueError(msg)
    forward = matrix[GREEDY_INDEX, GREEDY_INDEX + 1 :, ENTAILMENT]
    backward = matrix[GREEDY_INDEX + 1 :, GREEDY_INDEX, ENTAILMENT]
    return ((forward + backward) / 2).tolist()


def inconsistency(matrix: torch.Tensor) -> float:
    """One minus the mean similarity of the greedy answer to the samples (0 = all samples agree).

    Args:
        matrix: Question matrix as in :func:`greedy_similarities`.

    Returns:
        A value in ``[0, 1]``.
    """
    similarities = greedy_similarities(matrix)
    return 1.0 - sum(similarities) / len(similarities)


def cocoa(confidence_uncertainty: float, matrix: torch.Tensor) -> float:
    """CoCoA uncertainty of one question: confidence uncertainty times inconsistency.

    Args:
        confidence_uncertainty: Uncertainty of the greedy answer from a confidence method,
            non-negative (``-log P(answer)`` or ``1 - p(True)``).
        matrix: Question matrix as in :func:`greedy_similarities`.

    Returns:
        The hybrid uncertainty (higher = more uncertain).

    Raises:
        ValueError: If ``confidence_uncertainty`` is negative or not finite.
    """
    if not confidence_uncertainty >= 0.0 or confidence_uncertainty == float("inf"):
        msg = f"confidence_uncertainty must be finite and >= 0, got {confidence_uncertainty}."
        raise ValueError(msg)
    return confidence_uncertainty * inconsistency(matrix)


def cocoa_record(
    name: str,
    rows: Sequence[dict[str, Any]],
    confidence: dict[str, Any],
    get_matrix: Callable[[dict[str, Any]], torch.Tensor],
) -> dict[str, Any]:
    """Score record of a hybrid method from a confidence method's record (see CONTRACT.md).

    A question fails if the confidence method failed on it or its matrix cannot be used;
    it then gets uncertainty 1.0 (contract rule 3).

    Args:
        name: Method name of the hybrid record.
        rows: Rows of the data cache.
        confidence: Score record of the confidence ingredient, with the same ids as ``rows``.
        get_matrix: Returns the NLI matrix of one row.

    Returns:
        The score record. ``seconds_per_question`` is the confidence method's cost; the
        matrices come from the shared cache.

    Raises:
        ValueError: If the ids of ``confidence`` do not match ``rows``.
    """
    ids = [row["id"] for row in rows]
    if list(confidence["ids"]) != ids:
        msg = f"the {confidence['method']} record has other question ids than the data cache."
        raise ValueError(msg)
    uncertainty, failed = [], []
    for row, value, bad in zip(rows, confidence["uncertainty"], confidence["failed"], strict=True):
        try:
            if bad:
                msg = "confidence ingredient failed"
                raise ValueError(msg)  # noqa: TRY301 - one failure path for all reasons
            uncertainty.append(cocoa(float(value), get_matrix(row)))
            failed.append(False)
        except ValueError:
            uncertainty.append(1.0)
            failed.append(True)
    return {
        "method": name,
        "model": rows[0]["model"],
        "answer": "greedy",
        "confidence": confidence["method"],
        "ids": ids,
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": float(confidence.get("seconds_per_question", 0.0)),
    }
