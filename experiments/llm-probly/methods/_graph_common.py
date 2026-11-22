"""Shared code for the graph-measure plugins (B1).

The four plugins read the same per-question NLI matrices. If the week-2 NLI
cache (``data/<model>/<dataset>_<n>.nli.npz``) exists for these rows and was
built with the same NLI model and setting, it is used; otherwise the matrices
come from ``_se_common.get_matrix``, which computes them once per process and
shares them with the SE plugins. This file lookup is a bridge until the
persistent NLI cache (A5) is in the runner.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from common.data import DATA_DIR, load_jsonl
from common.graph_measures import graph_scores
from common.nli_matrix import load_nli_cache
from methods import _se_common

if TYPE_CHECKING:
    from pathlib import Path

    import torch

_scores: dict[tuple[str, ...], dict[str, float]] = {}


def _cache_file_for(rows: list[dict]) -> Path | None:
    """Return the ``.nli.npz`` file that holds exactly these rows, or None."""
    dataset, n, model = rows[0].get("dataset"), len(rows), rows[0].get("model")
    for jsonl in DATA_DIR.glob(f"*/{dataset}_{n}.jsonl"):
        npz = jsonl.with_suffix(".nli.npz")
        if not npz.exists():
            continue
        cached_rows = load_jsonl(jsonl)
        if cached_rows[0].get("model") != model:
            continue
        if [(r["id"], r["greedy_answer"], r["samples"]) for r in cached_rows] == [
            (r["id"], r["greedy_answer"], r["samples"]) for r in rows
        ]:
            return npz
    return None


def get_matrices(rows: list[dict]) -> list[torch.Tensor]:
    """NLI matrices of these rows: from the cache file if it matches, else computed."""
    npz = _cache_file_for(rows)
    if npz is not None:
        cache = load_nli_cache(npz)
        if cache["nli_model"] == _se_common.NLI_MODEL and not cache["use_question"]:
            return cache["matrices"]
    return [_se_common.get_matrix(row) for row in rows]


def score_rows(rows: list[dict], name: str) -> dict:
    """Score record for one graph measure (``name`` is a key of ``graph_scores``), see CONTRACT.md."""
    start = time.perf_counter()
    matrices = get_matrices(rows)
    ids, uncertainty, failed = [], [], []
    for row, matrix in zip(rows, matrices, strict=True):
        ids.append(row["id"])
        key = (row["greedy_answer"], *row["samples"])
        try:
            if key not in _scores:
                _scores[key] = graph_scores(matrix)
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
    }
