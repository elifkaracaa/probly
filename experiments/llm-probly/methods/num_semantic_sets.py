"""Number of semantic sets (B1): how many meaning clusters the samples form.

Counts the clusters of probly's greedy NLI clusterer (the clusters semantic
entropy uses), read from the shared NLI matrix. Follows LM-Polygraph's
NumSemSets (Lin et al., 2024). Higher = more uncertain.
"""

from __future__ import annotations

from methods._graph_common import score_rows

NAME = "num_semantic_sets"
FAMILY = "agreement"
NEEDS_MODEL = False


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:  # noqa: ARG001
    """Score every row from the shared NLI matrix (see methods/_graph_common.py)."""
    return score_rows(rows, NAME)
