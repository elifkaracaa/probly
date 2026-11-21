"""Score a text-only API cache with the four text-only methods (F2).

Runs ``lexical_similarity``, ``discrete_semantic_entropy``,
``verbalized_2s_api`` and ``p_true_empirical_api`` on a cache made by
``scripts/generate_api.py``, saves one result file per method to
``results/<model-key>/<cache name>/<method>.json`` (same record format as the
other plugins) and prints AUROC next to the local model's result, if given.

Example (run from experiments/llm-probly, Ollama running)::

    python -m scripts.score_api_cache --cache data/ollama-qwen0.5b/triviaqa_200.jsonl \
        --compare data/qwen0.5b/triviaqa_200.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (
    str(ROOT) not in sys.path
):  # allow `python scripts/score_api_cache.py` as well as `-m`
    sys.path.insert(0, str(ROOT))

from common.evaluate import auroc  # noqa: E402

METHODS = (
    "lexical_similarity",
    "discrete_semantic_entropy",
    "verbalized_2s_api",
    "p_true_empirical_api",
)


def load_rows(path: Path) -> list[dict]:
    """Read a JSONL cache.

    Args:
        path: Cache path.

    Returns:
        The rows.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def local_auroc(compare: Path, results_root: Path, method: str) -> float | None:
    """AUROC of the local model's saved result for the matching method.

    Args:
        compare: Local cache (for the correctness labels), e.g. ``data/qwen0.5b/triviaqa_200.jsonl``.
        results_root: Root of the results folder.
        method: Method name; a trailing ``_api`` is dropped to find the local result.

    Returns:
        The AUROC, or ``None`` if the local result file does not exist.
    """
    name = method.removesuffix("_api")
    path = results_root / compare.parent.name / compare.stem / f"{name}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    correct = {row["id"]: row["correct"] for row in load_rows(compare)}
    return float(auroc(record["uncertainty"], [correct[i] for i in record["ids"]]))


def main(argv: list[str] | None = None) -> dict[str, dict]:
    """Command-line entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv``).

    Returns:
        Per-method summary (AUROC, local AUROC, failures, cost).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="cache from scripts/generate_api.py"
    )
    parser.add_argument(
        "--compare", type=Path, default=None, help="local cache of the same questions"
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    rows = load_rows(args.cache)
    correct = [row["correct"] for row in rows]
    out_dir = args.results_root / args.cache.parent.name / args.cache.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    accuracy = sum(correct) / len(correct)
    print(f"{args.cache}: {len(rows)} questions, accuracy {accuracy:.3f}")
    if args.compare:
        local_rows = load_rows(args.compare)
        print(
            f"{args.compare}: accuracy {sum(r['correct'] for r in local_rows) / len(local_rows):.3f}"
        )

    summary: dict[str, dict] = {}
    for method in args.methods:
        record = importlib.import_module(f"methods.{method}").score(rows)
        (out_dir / f"{method}.json").write_text(json.dumps(record))
        summary[method] = {
            "auroc": float(auroc(record["uncertainty"], correct)),
            "local_auroc": local_auroc(args.compare, args.results_root, method)
            if args.compare
            else None,
            "failed": sum(record["failed"]),
            "seconds_per_question": record["seconds_per_question"],
            "api_requests_per_question": record.get("api_requests_per_question", 0),
        }
        s = summary[method]
        local = "n/a" if s["local_auroc"] is None else f"{s['local_auroc']:.3f}"
        print(
            f"{method:28s} AUROC={s['auroc']:.3f}  local={local}  failed={s['failed']}  "
            f"s/q={s['seconds_per_question']:.2f}  api_requests/q={s['api_requests_per_question']:g}",
            flush=True,
        )
    (out_dir / "summary_api.json").write_text(
        json.dumps({"accuracy": accuracy, "methods": summary}, indent=2)
    )
    return summary


if __name__ == "__main__":
    main()
