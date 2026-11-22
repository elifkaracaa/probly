"""Sum of eigenvalues of the graph Laplacian (B1).

A continuous version of the number of semantic sets: sum of max(0, 1 - eigenvalue)
over the normalized Laplacian of the samples' NLI similarity graph. Follows
LM-Polygraph's EigValLaplacian (NLI score, entail affinity; Lin et al., 2024).
Higher = more uncertain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from methods._graph_common import score_rows

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "eig_val_laplacian"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score every row from the shared NLI matrix (see methods/_graph_common.py)."""
    return score_rows(rows, NAME)
