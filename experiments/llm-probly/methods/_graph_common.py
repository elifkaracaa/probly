"""Shared code for the graph-measure plugins (B1).

The four plugins read the same per-question NLI matrices as SE and discrete SE, through
``_se_common.prepare_matrices``: from memory, then from the persistent NLI cache the runner set
(A5, ``data/<model>/<dataset>_<n>.nli.<tag>.npz``), else computed with the NLI setup of
``common.nli_matrix.nli_settings()``. So a run of SE and the graph measures on the same data runs
the NLI model at most once per question, and a second run does no NLI work at all.

Graph scores are kept per matrix fingerprint (``_se_common.matrix_key``), which includes the
NLI setup, so changing the setup in the same process never reuses old scores.
"""

from __future__ import annotations

import time

from common.graph_measures import graph_scores
from common.nli_matrix import nli_settings

from methods import _se_common

_scores: dict[str, dict[str, float]] = {}


def score_rows(rows: list[dict], name: str) -> dict:
    """Score record for one graph measure (``name`` is a key of ``graph_scores``), see CONTRACT.md."""
    start = time.perf_counter()
    pairs_before = _se_common.nli_pairs_scored
    _se_common.prepare_matrices(rows)
    nli_seconds = _se_common._nli_seconds  # noqa: SLF001 - what building each matrix took (A5), as in score_rows of SE
    ids, uncertainty, failed = [], [], []
    for row in rows:
        ids.append(row["id"])
        key = _se_common.matrix_key(row)
        try:
            if key not in _scores:
                _scores[key] = graph_scores(_se_common.get_matrix(row))
            uncertainty.append(_scores[key][name])
            failed.append(False)
        except Exception:  # noqa: BLE001
            uncertainty.append(1.0)
            failed.append(True)
    return {
        "method": name,
        "model": rows[0].get("model"),
        "answer": "greedy",
        "ids": ids,
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": (time.perf_counter() - start) / len(rows),
        "nli_pairs_per_question": (_se_common.nli_pairs_scored - pairs_before) / len(rows),
        "nli_seconds_per_question": sum(nli_seconds.get(_se_common.matrix_key(r), 0.0) for r in rows) / len(rows),
        "nli": nli_settings().tag,
    }
