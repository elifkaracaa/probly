"""Eccentricity (B1): spread of the samples in the spectral embedding of their NLI graph.

Frobenius norm of the centred eigenvector coordinates of the normalized
Laplacian (eigenvalues below 0.9). Follows LM-Polygraph's Eccentricity (NLI
score, entail affinity; Lin et al., 2024). Higher = more uncertain.
"""

from __future__ import annotations

from methods._graph_common import score_rows

NAME = "eccentricity"
FAMILY = "agreement"
NEEDS_MODEL = False


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:  # noqa: ARG001
    """Score every row from the shared NLI matrix (see methods/_graph_common.py)."""
    return score_rows(rows, NAME)
