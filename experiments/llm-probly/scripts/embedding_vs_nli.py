"""Speed and AUROC of the embedding backend vs NLI clustering (B3).

Reads the cached samples and the benchmark's NLI matrices (no LLM): the NLI setup of
``common.nli_matrix.nli_settings()`` (deberta-large by default) from its A5 cache, computed and
added to the cache for any question that is missing. Run from
``experiments/llm-probly``::

    uv run python -m scripts.embedding_vs_nli --models qwen0.5b qwen1.5b

For every model it scores SE, discrete SE and the graph measures with the NLI
matrix and with the embedding matrices (the runner's ``--semantic-backend
embedding`` path) at every cosine threshold of ``--thresholds``. The threshold is **chosen on the first
model** (highest SE AUROC) and then only reported on the others, so they are a
held-out check. Speed is measured on this machine for both backends on the
same ``--timing-questions`` questions.

Writes ``results/<model>/<dataset>_<n>/analysis/embedding_backend.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING

from common.data import cache_path, load_jsonl
from common.evaluate import auroc
from common.io import RESULTS_ROOT
from common.nli_matrix import QUESTIONS_PER_BATCH, compute_matrices, nli_cache_path, nli_settings, set_nli_cache_file
from common.semantic_scores import cluster_ids, scores_from_matrix
import numpy as np

from probly.representer.embedder.huggingface import DEFAULT_EMBEDDING_MODEL
from probly.representer.semantic_clustering.huggingface import HFEmbeddingSemanticClusterer

if TYPE_CHECKING:
    from pathlib import Path

    import torch


def tied(values: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Make scores closer than ``tolerance`` exact ties before AUROC.

    Same as ``common.sample_count.tied`` (B2, in review); switch to that import once B2 is merged.
    """
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    start = np.concatenate(([True], np.diff(ranked) >= tolerance))
    out = np.empty_like(ranked)
    out[order] = ranked[np.flatnonzero(start)[np.cumsum(start) - 1]]
    return out


SCORED = ("semantic_entropy", "discrete_semantic_entropy", "eig_val_laplacian", "deg_mat", "num_semantic_sets")


def embedding_matrices(
    clusterer: HFEmbeddingSemanticClusterer, rows: list[dict], threshold: float
) -> tuple[list[torch.Tensor], float]:
    """Matrices of every question through the library path (the runner's path) and seconds per question.

    Same call as ``--semantic-backend embedding`` in the runner, so identical answers are
    deduplicated and get entailment 1, as in the NLI matrices.
    """
    clusterer.threshold = threshold
    start = time.perf_counter()
    matrices = []
    for first in range(0, len(rows), QUESTIONS_PER_BATCH):
        matrices += compute_matrices(clusterer, rows[first : first + QUESTIONS_PER_BATCH])
    return matrices, (time.perf_counter() - start) / len(rows)


def nli_seconds(rows: list[dict]) -> float:
    """Seconds per question of the benchmark's NLI path (current NLI settings, A2 + A3) on this machine."""
    from methods._se_common import get_clusterer  # noqa: PLC0415  (loads the NLI model)

    clusterer = get_clusterer()
    compute_matrices(clusterer, rows[:2])  # warm-up
    start = time.perf_counter()
    compute_matrices(clusterer, rows)
    return (time.perf_counter() - start) / len(rows)


def benchmark_nli_matrices(rows: list[dict], data_file: Path) -> list[torch.Tensor]:
    """NLI matrices of the benchmark's setup: from its A5 cache, computed (and cached) where missing."""
    from methods import _se_common  # noqa: PLC0415  (loads the NLI model only if something is missing)

    set_nli_cache_file(nli_cache_path(data_file, nli_settings()))
    try:
        _se_common.prepare_matrices(rows)
    finally:
        set_nli_cache_file(None)
    return [_se_common.get_matrix(row) for row in rows]


def evaluate(matrices: list[torch.Tensor], rows: list[dict], correct: np.ndarray) -> dict:
    """AUROC of every scored method, mean cluster count and the cluster ids."""
    scores = [scores_from_matrix(m, r["samples"], r["samples_ll_mean"]) for m, r in zip(matrices, rows, strict=True)]
    result = {name: auroc(tied(np.array([s[name] for s in scores])), correct) for name in SCORED}
    clusters = [cluster_ids(m) for m in matrices]
    result["mean_clusters"] = float(np.mean([len(set(c)) for c in clusters]))
    return {"auroc": result, "clusters": clusters}


def same_partition(a: list[int], b: list[int]) -> bool:
    """True if two cluster assignments group the samples identically (ids may differ)."""
    return len({*zip(a, b, strict=True)}) == len(set(a)) == len(set(b))


def to_markdown(model: str, run: str, record: dict) -> str:
    """The comparison as Markdown."""
    speedup = record["seconds_nli"] / record["seconds_embedding"]
    head = "| backend | " + " | ".join(SCORED) + " | mean clusters | same clusters as NLI |"
    lines = [
        f"# Embedding backend vs NLI: `{model}/{run}`",
        "",
        (
            f"Embedding model `{record['embedding_model']}`; threshold chosen on `{record['selected_on']}`: "
            f"**{record['threshold']}**{' (held-out model)' if record['selected_on'] != model else ''}."
        ),
        "",
        "## Speed (this machine, same questions)",
        "",
        "| backend | seconds per question | speed-up |",
        "|---|---|---|",
        f"| NLI ({record['nli_setup']}, A2 + A3) | {record['seconds_nli']:.3f} | 1x |",
        f"| embedding | {record['seconds_embedding']:.3f} | {speedup:.0f}x |",
        "",
        "## AUROC",
        "",
        head,
        "|---|" + "---|" * (len(SCORED) + 2),
    ]

    def row(label: str, res: dict) -> str:
        cells = [f"{res['auroc'][n]:.3f}" for n in SCORED]
        same = "" if "same_as_nli" not in res else f"{res['same_as_nli']:.0%}"
        return f"| {label} | " + " | ".join(cells) + f" | {res['auroc']['mean_clusters']:.2f} | {same} |"

    lines.append(row(f"NLI ({record['nli_setup']})", record["nli"]))
    lines += [row(f"embedding, t = {t}", res) for t, res in record["embedding"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Compare the two backends on every model."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["qwen0.5b", "qwen1.5b"])
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ap.add_argument("--timing-questions", type=int, default=20)
    a = ap.parse_args()

    clusterer = HFEmbeddingSemanticClusterer.from_model_name()
    run, threshold, selected_on = f"{a.dataset}_{a.n}", None, a.models[0]
    for model in a.models:
        rows = load_jsonl(cache_path(model, a.n, a.dataset))
        nli_matrices = benchmark_nli_matrices(rows, cache_path(model, a.n, a.dataset))
        correct = np.array([bool(r["correct"]) for r in rows])

        _, seconds_embedding = embedding_matrices(clusterer, rows[: a.timing_questions], a.thresholds[0])
        seconds_nli = nli_seconds(rows[: a.timing_questions])

        nli = evaluate(nli_matrices, rows, correct)
        embedding = {}
        for t in a.thresholds:
            res = evaluate(embedding_matrices(clusterer, rows, t)[0], rows, correct)
            res["same_as_nli"] = float(
                np.mean([same_partition(x, y) for x, y in zip(res["clusters"], nli["clusters"], strict=True)])
            )
            embedding[t] = res
        if threshold is None:  # choose on the first model only
            threshold = max(a.thresholds, key=lambda t: embedding[t]["auroc"]["semantic_entropy"])

        record = {
            "model": model,
            "run": run,
            "nli_setup": nli_settings().tag,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "selected_on": selected_on,
            "threshold": threshold,
            "seconds_nli": seconds_nli,
            "seconds_embedding": seconds_embedding,
            "nli": {"auroc": nli["auroc"]},
            "embedding": {t: {"auroc": r["auroc"], "same_as_nli": r["same_as_nli"]} for t, r in embedding.items()},
        }
        out = RESULTS_ROOT / model / run / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / "embedding_backend.json").write_text(json.dumps(record, indent=2) + "\n")
        (out / "embedding_backend.md").write_text(to_markdown(model, run, record))
        print(f"{model}: NLI {seconds_nli:.3f} s/q, embedding {seconds_embedding:.3f} s/q")
        print(f"   wrote {out / 'embedding_backend.md'}")


if __name__ == "__main__":
    main()
