"""Shared code for the semantic-entropy plugins (SE and discrete SE).

Both plugins read the same per-question NLI matrix (A1). The matrix is computed
once per question and kept in memory, so when the runner calls
``semantic_entropy`` and then ``discrete_semantic_entropy`` in the same process,
the NLI model runs only for the first one.

Which NLI model is used, and whether the question goes in front of each answer, comes from
``common.nli_matrix.nli_settings()`` (default deberta-large, answers only; the runner sets it
from the config keys ``nli_model`` and ``nli_use_question``). Each record says which setup it used
in its ``nli`` field.

The matrices are also kept on disk (A5): when the runner has set a cache file
(``common.nli_matrix.set_nli_cache_file``), matrices whose fingerprint matches are read from it,
only the missing or changed questions go to the NLI model, and the file is updated. A second run
on the same data therefore does no NLI work (``nli_pairs_per_question`` is 0).

``seconds_per_question`` is the time of this run (CONTRACT.md: about 0 when the matrices come from
the cache). So that the cost of the NLI step is not lost, every record also has
``nli_seconds_per_question``: what building the matrices it used took, measured now or stored in the cache.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from common.nli_matrix import (
    QUESTIONS_PER_BATCH,
    cached_nli_seconds,
    compute_matrices,
    matrix_fingerprint,
    nli_cache_file,
    nli_pair_count,
    nli_settings,
    pick_device,
    read_cached_matrices,
    save_nli_cache,
    se_and_dse,
)

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    from pathlib import Path

    import torch

_clusterers: dict[str, HFGreedySemanticClusterer] = {}
_matrices: dict[str, torch.Tensor] = {}
"""In-memory matrices of this process, keyed by :func:`matrix_key`."""
_disk: dict[Path, dict[int, tuple[str, torch.Tensor]]] = {}
"""Entries of each persistent cache file read in this process: ``{question id: (fingerprint, matrix)}``."""
_nli_seconds: dict[str, float] = {}
"""NLI seconds it took to build each matrix in ``_matrices`` (measured, or the cache file's value)."""
nli_pairs_scored = 0
"""Number of NLI pairs sent to the model in this process (for before / after reports)."""


def get_clusterer() -> HFGreedySemanticClusterer:
    """Load the NLI clusterer of the current settings once per process."""
    name = nli_settings().hf_name
    if name not in _clusterers:
        clusterer = HFGreedySemanticClusterer.from_model_name(name)
        clusterer.model.to(pick_device())
        _clusterers[name] = clusterer
    return _clusterers[name]


def matrix_key(row: dict) -> str:
    """Key of a question matrix: its fingerprint under the current NLI settings (model, question if used, answers)."""
    return matrix_fingerprint(row, nli_settings())


def _disk_entries(path: Path) -> dict[int, tuple[str, torch.Tensor]]:
    """Reusable entries of a cache file, read once per process."""
    if path not in _disk:
        _disk[path] = read_cached_matrices(path, nli_settings())
    return _disk[path]


def prepare_matrices(rows: list[dict]) -> None:
    """Make the NLI matrices of ``rows`` available: from memory, then from the cache file, else computed.

    Missing matrices are computed ``QUESTIONS_PER_BATCH`` questions per model batch. If a cache
    file is set and anything was computed, the file is rewritten with the old and new entries.
    """
    global nli_pairs_scored  # noqa: PLW0603
    path = nli_cache_file()
    entries = _disk_entries(path) if path is not None else {}
    for row in rows:
        key = matrix_key(row)
        cached = entries.get(row["id"]) if path is not None else None
        if key not in _matrices and cached is not None and cached[0] == key:
            _matrices[key] = cached[1]
            _nli_seconds[key] = cached_nli_seconds(path) if path is not None else 0.0

    missing = {matrix_key(row): row for row in rows if matrix_key(row) not in _matrices}
    todo = list(missing.items())
    for start in range(0, len(todo), QUESTIONS_PER_BATCH):
        chunk = todo[start : start + QUESTIONS_PER_BATCH]
        t0 = time.time()
        matrices = compute_matrices(
            get_clusterer(), [row for _, row in chunk], use_question=nli_settings().use_question
        )
        seconds = (time.time() - t0) / len(chunk)
        for (key, row), matrix in zip(chunk, matrices, strict=True):
            _matrices[key] = matrix
            _nli_seconds[key] = seconds
            nli_pairs_scored += nli_pair_count(row)

    if path is not None and todo:
        old_seconds = cached_nli_seconds(path)
        for row in rows:
            entries[row["id"]] = (matrix_key(row), _matrices[matrix_key(row)])
        per_entry = [_nli_seconds.get(fp, old_seconds) for fp, _ in entries.values()]
        _write_cache(path, entries, seconds_per_question=sum(per_entry) / len(per_entry))


def _write_cache(path: Path, entries: dict[int, tuple[str, torch.Tensor]], *, seconds_per_question: float) -> None:
    """Write all entries of a cache file, sorted by question id (the order of the data cache)."""
    ids = sorted(entries)
    settings = nli_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")  # written fully first, so a stopped run never leaves half a file
    save_nli_cache(
        tmp,
        [entries[i][1] for i in ids],
        ids,
        nli_model=settings.hf_name,
        use_question=settings.use_question,
        seconds_per_question=seconds_per_question,
        fingerprints=[entries[i][0] for i in ids],
    )
    tmp.replace(path)


def get_matrix(row: dict) -> torch.Tensor:
    """Return the NLI matrix of one question, computing it only the first time."""
    prepare_matrices([row])
    return _matrices[matrix_key(row)]


def compute_se_and_dse(row: dict) -> tuple[float, float]:
    """Semantic entropy and discrete semantic entropy of one question."""
    return se_and_dse(get_matrix(row), row["samples_ll_mean"])


def score_rows(rows: list[dict], name: str, which: int) -> dict:
    """Score record for SE (``which=0``) or discrete SE (``which=1``), see CONTRACT.md."""
    ids, unc, failed = [], [], []
    pairs_before = nli_pairs_scored
    t0 = time.time()
    prepare_matrices(rows)
    for i, r in enumerate(rows):
        ids.append(r.get("id", i))
        try:
            unc.append(compute_se_and_dse(r)[which])
            failed.append(False)
        except Exception:  # noqa: BLE001
            unc.append(1.0)
            failed.append(True)
    return {
        "method": name,
        "model": rows[0].get("model"),
        "answer": "greedy",
        "ids": ids,
        "uncertainty": unc,
        "failed": failed,
        "seconds_per_question": (time.time() - t0) / len(rows),
        "nli_pairs_per_question": (nli_pairs_scored - pairs_before) / len(rows),
        "nli_seconds_per_question": sum(_nli_seconds[matrix_key(r)] for r in rows) / len(rows),
        "nli": nli_settings().tag,
    }
