"""Step 1 of the selector: load every method's scores for one model into one table.

The selector never runs a model. It only reads what the runner already saved:

- ``data/<model>/<dataset>_<n>.jsonl``: one row per question, with ``correct``.
- ``results/<model>/<dataset>_<n>/<method>.json``: one score file per method.

Run from ``experiments/llm-probly``::

    uv run python -m selector.table --model qwen1.5b
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from common.data import cache_path, load_jsonl
from common.io import RESULTS_ROOT, load_scores
import pandas as pd

# Score files that are checks, not real methods. The selector never picks them.
NOT_CANDIDATES = frozenset({"dummy_random"})


@dataclass
class ScoreTable:
    """All method scores of one model, aligned by question.

    Attributes:
        model: Short model name, e.g. ``"qwen1.5b"``.
        run: Run folder name, e.g. ``"triviaqa_200"``.
        scores: One row per question (index = question id), one column per method.
            Values are uncertainties (higher = more uncertain), as in CONTRACT.md.
        correct: Whether the model's greedy answer was right, same index as ``scores``.
        info: Per method: ``seconds_per_question``, ``failed_fraction`` and
            ``probability`` (True if uncertainty = 1 - probability of being correct).
        skipped: Score files that were not loaded, with the reason.
    """

    model: str
    run: str
    scores: pd.DataFrame
    correct: pd.Series
    info: pd.DataFrame
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def methods(self) -> list[str]:
        """Names of the candidate methods in this table."""
        return list(self.scores.columns)


def load_score_table(
    model: str,
    dataset: str = "triviaqa",
    n: int = 200,
    results_root: Path = RESULTS_ROOT,
    rows: list[dict] | None = None,
) -> ScoreTable:
    """Load the labels and every method's scores for one model.

    Args:
        model: Short model name (sub-folder of ``data/`` and ``results/``).
        dataset: Dataset name, e.g. ``"triviaqa"``.
        n: Number of questions in the run.
        results_root: Folder that holds ``<model>/<dataset>_<n>/``. Tests pass a temporary folder.
        rows: Labelled cache rows. Read from ``data/`` when ``None``.

    Returns:
        A ``ScoreTable`` with one column per method that has a valid score file.

    Raises:
        FileNotFoundError: If the model has no results folder for this run.
        ValueError: If the cache rows have no ``correct`` labels.
    """
    run = f"{dataset}_{n}"
    folder = Path(results_root) / model / run
    if not folder.is_dir():
        msg = f"no results folder {folder}; run the benchmark for {model} first"
        raise FileNotFoundError(msg)

    if rows is None:
        rows = load_jsonl(cache_path(model, n, dataset))
    if any("correct" not in row for row in rows):
        msg = f"{model}/{run}: cache rows have no 'correct' label; run scripts.label first"
        raise ValueError(msg)
    ids = [row["id"] for row in rows]
    correct = pd.Series([bool(row["correct"]) for row in rows], index=ids, name="correct")

    columns: dict[str, list[float]] = {}
    info: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for path in sorted(folder.glob("*.json")):
        name = path.stem
        if name in NOT_CANDIDATES:
            skipped[name] = "check method, not a candidate"
            continue
        try:
            record = load_scores(path, expected_ids=ids)
        except ValueError as error:
            skipped[name] = f"invalid file: {error}"
            continue
        columns[name] = record["uncertainty"]
        failed = record["failed"]
        info[name] = {
            "seconds_per_question": float(record["seconds_per_question"]),
            "failed_fraction": sum(failed) / len(failed),
            "probability": bool(record.get("probability", False)),
        }

    scores = pd.DataFrame(columns, index=ids)
    scores.index.name = "id"
    return ScoreTable(
        model=model,
        run=run,
        scores=scores,
        correct=correct,
        info=pd.DataFrame.from_dict(info, orient="index"),
        skipped=skipped,
    )


def main() -> None:
    """Print what was loaded for one model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen1.5b")
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    table = load_score_table(args.model, args.dataset, args.n)
    n_correct = int(table.correct.sum())
    print(
        f"{table.model} / {table.run}: {len(table.correct)} questions, {n_correct} correct "
        f"({n_correct / len(table.correct):.1%})"
    )
    print(f"{len(table.methods)} candidate methods\n")
    print(table.info.to_string(float_format=lambda x: f"{x:.3f}"))
    for name, reason in table.skipped.items():
        print(f"skipped {name}: {reason}")


if __name__ == "__main__":
    main()
