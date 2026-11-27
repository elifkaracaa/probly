"""Build the cocoa_p_true record from a saved p_true_zeroshot record, without the LLM (C5).

The plugin ``methods/cocoa_p_true.py`` runs zero-shot p(True) itself, which needs the model.
For runs where ``results/<model>/<dataset>_<n>/p_true_zeroshot.json`` already exists, this
script reads it and the run's NLI cache (A5) and writes the same record with
``common.hybrid.cocoa_record`` (the function the plugin uses). Run from ``experiments/llm-probly``::

    uv run python -m scripts.hybrid_from_results --runs qwen0.5b/triviaqa_200 qwen7b/triviaqa_200

Questions whose NLI matrix is missing from the cache are computed with the current NLI settings
(``common.nli_matrix.nli_settings()``, deberta-large by default) and added to the cache.
"""

from __future__ import annotations

import argparse

from common.data import cache_path, load_jsonl
from common.hybrid import cocoa_record
from common.io import RESULTS_ROOT, load_scores, save_scores
from common.nli_matrix import nli_cache_path, nli_settings, set_nli_cache_file

from methods import _se_common, cocoa_p_true


def parse_run(run: str) -> tuple[str, str, int]:
    """Split ``<model>/<dataset>_<n>`` into model, dataset and number of questions.

    Args:
        run: Run name as in ``results/``, e.g. ``qwen1.5b/triviaqa_200``.

    Returns:
        ``(model, dataset, n)``.

    Raises:
        ValueError: If the name does not have that form.
    """
    model, _, rest = run.partition("/")
    dataset, _, n = rest.rpartition("_")
    if not model or not dataset or not n.isdigit():
        msg = f"run must look like <model>/<dataset>_<n>, got {run!r}."
        raise ValueError(msg)
    return model, dataset, int(n)


def build(run: str) -> dict:
    """Write ``cocoa_p_true.json`` for one run and return the record."""
    model, dataset, n = parse_run(run)
    data_file = cache_path(model, n, dataset)
    rows = load_jsonl(data_file)
    ids = [row["id"] for row in rows]
    confidence = load_scores(RESULTS_ROOT / run / f"{cocoa_p_true.CONFIDENCE}.json", expected_ids=ids)
    set_nli_cache_file(nli_cache_path(data_file, nli_settings()))
    try:
        _se_common.prepare_matrices(rows)
        record = cocoa_record(cocoa_p_true.NAME, rows, confidence, _se_common.get_matrix)
    finally:
        set_nli_cache_file(None)
    record["nli"] = nli_settings().tag
    path = save_scores(record, run=run, expected_ids=ids)
    print(f"{run}: saved {path.relative_to(RESULTS_ROOT.parent)} ({sum(record['failed'])} failed)")
    return record


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="runs like qwen1.5b/triviaqa_200")
    args = parser.parse_args()
    for run in args.runs:
        build(run)


if __name__ == "__main__":
    main()
