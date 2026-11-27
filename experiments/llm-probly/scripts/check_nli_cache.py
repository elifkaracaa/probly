"""Check a saved NLI cache against probly's clusterer.

For every question, cluster the samples twice:
1. with probly's HFGreedySemanticClusterer (runs the NLI model), and
2. from the saved NLI cache (no model), with each rule of ``common.nli_matrix.RULES``:
   - probly:  no contradiction in either direction and not neutral-neutral (probly's rule)
   - strict:  entailment in both directions
   - lenient: no contradiction in either direction
Then report for how many questions each rule gives exactly the same clusters as probly.

Run from experiments/llm-probly:
    uv run python -m scripts.check_nli_cache                                    # 5-question file
    uv run python -m scripts.check_nli_cache --n 200 --limit 30                 # first 30 of the 200 file
    uv run python -m scripts.check_nli_cache --n 200 --nli-model deberta-large  # the benchmark's cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.nli_matrix import (
    NLI_MODELS,
    RULES,
    NliSettings,
    as_partition,
    greedy_clusters,
    load_nli_cache,
    nli_cache_path,
)
import numpy as np

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Cluster every question with probly and from the cache, and print how often they agree."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--n", type=int, default=5, help="which cache file: triviaqa_<n>")
    ap.add_argument("--nli-model", default="deberta-base", help=f"NLI setup of the cache: {sorted(NLI_MODELS)}")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    args = ap.parse_args()

    data = ROOT / "data" / args.model / f"triviaqa_{args.n}.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    cache = load_nli_cache(nli_cache_path(data, NliSettings(args.nli_model)))
    if args.limit:
        rows = rows[: args.limit]

    # 1. Is the cache itself consistent with the data file?
    ids_ok = [str(i) for i in cache["ids"][: len(rows)]] == [str(r["id"]) for r in rows]
    texts_ok = all(cache["matrices"][i].shape[0] == 1 + len(r["samples"]) for i, r in enumerate(rows))
    print(f"cache {cache['nli_model']}, use_question={cache['use_question']}")
    print(f"ids match: {ids_ok}   texts per question match: {texts_ok}")

    # 2. probly's clusterer (same NLI model as the cache)
    clusterer = HFGreedySemanticClusterer.from_model_name(cache["nli_model"])
    matches = dict.fromkeys(RULES, 0)
    for i, r in enumerate(rows):
        samples = np.asarray(r["samples"], dtype=object)
        ids = clusterer._cluster_row(samples).tolist()  # noqa: SLF001 - one semantic id per sample, same as SE uses
        probly_clusters = as_partition(
            [[k + 1 for k, s in enumerate(ids) if s == c] for c in sorted(set(ids))]
        )  # +1: in the cache, index 0 is the greedy answer and the samples are 1..K

        indices = range(1, len(r["samples"]) + 1)
        line = f"id {r['id']:>4}: probly {len(probly_clusters)} clusters"
        for rule in RULES:
            cached = as_partition(greedy_clusters(cache["matrices"][i], indices, rule))
            same = cached == probly_clusters
            matches[rule] += same
            line += f" | {rule}: {len(cached)}{'' if same else ' DIFF'}"
        print(line)

    print(f"\nsame clusters as probly ({len(rows)} questions):")
    for rule in RULES:
        print(f"  {rule:<8} {matches[rule]}/{len(rows)}")


if __name__ == "__main__":
    main()
