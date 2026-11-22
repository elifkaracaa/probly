"""Eccentricity (B1): spread of the samples in the spectral embedding of their NLI graph.

Frobenius norm of the centred eigenvector coordinates of the normalized
Laplacian (eigenvalues below 0.9). Follows LM-Polygraph's Eccentricity (NLI
score, entail affinity; Lin et al., 2024). Higher = more uncertain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from methods._graph_common import score_rows

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "eccentricity"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score every row from the shared NLI matrix (see methods/_graph_common.py)."""
    return score_rows(rows, NAME)
