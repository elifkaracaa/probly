"""Degree matrix (B1): how weakly the samples agree with each other on average.

uncertainty = 1 - mean degree / K, with the degree of a sample the sum of its
NLI similarities to all samples. Follows LM-Polygraph's DegMat (NLI score,
entail affinity; Lin et al., 2024). Higher = more uncertain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from methods._graph_common import score_rows

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "deg_mat"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score every row from the shared NLI matrix (see methods/_graph_common.py)."""
    return score_rows(rows, NAME)
