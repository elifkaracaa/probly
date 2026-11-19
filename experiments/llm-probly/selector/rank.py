"""Step 3 of the selector: rank every method on the calibration part, with intervals.

For each candidate method this computes, on the calibration questions only:

- AUROC: how well the uncertainty separates wrong from correct answers (0.5 = random).
- PRR: how much of the oracle's gain the method gets when the most uncertain
  answers are rejected first (0 = random, 1 = oracle).

Both use the group's functions from ``common/evaluate.py``, so the numbers match
the results table. Each value gets a 95 % bootstrap interval: the questions are
resampled with replacement many times and the metric is recomputed. All methods
use the same resamples, so their intervals are directly comparable. Step 4 uses
these intervals to decide which methods are tied with the best one.

Run from ``experiments/llm-probly``::

    uv run python -m selector.rank --model qwen1.5b
"""

from __future__ import annotations

import argparse
import math

from common.evaluate import auroc, prr
import numpy as np
import pandas as pd

from selector.split import DEFAULT_CALIBRATION_SIZE, DEFAULT_SEED, make_split, subset
from selector.table import ScoreTable, load_score_table

DEFAULT_N_BOOT = 1000
DEFAULT_ALPHA = 0.05
# PRR breaks ties randomly and averages over shuffles. The point estimate uses the
# full 20 shuffles of common/evaluate.py; inside the bootstrap fewer are enough,
# because the bootstrap itself already averages over many random draws.
BOOT_TIE_SEEDS = 3


def bootstrap_indices(correct: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Draw ``n_boot`` resamples of the question positions.

    Resamples that contain only correct or only wrong answers are dropped,
    because AUROC and PRR are undefined there.

    Args:
        correct: Correctness labels of the questions, as booleans.
        n_boot: Number of resamples to draw.
        seed: Random seed.

    Returns:
        One array of positions per kept resample.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    kept = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if correct[idx].all() or not correct[idx].any():
            continue
        kept.append(idx)
    return kept


def rank_methods(
    table: ScoreTable,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compute AUROC and PRR with bootstrap intervals for every method, best AUROC first.

    Args:
        table: Scores and labels to evaluate on (the calibration part, step 2).
        n_boot: Number of bootstrap resamples.
        alpha: 1 - confidence level of the intervals (0.05 gives 95 % intervals).
        seed: Random seed for the resamples.

    Returns:
        One row per method (index = method name) with the columns ``auroc``,
        ``auroc_low``, ``auroc_high``, ``prr``, ``prr_low`` and ``prr_high``.

    Raises:
        ValueError: If the table has no methods, or no resample has both classes.
    """
    if not table.methods:
        msg = f"{table.model}: no candidate methods to rank"
        raise ValueError(msg)
    correct = table.correct.to_numpy(dtype=bool)
    resamples = bootstrap_indices(correct, n_boot, seed)
    if not resamples:
        msg = f"{table.model}: every bootstrap resample had only one class"
        raise ValueError(msg)

    low_q, high_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = {}
    for method in table.methods:
        u = table.scores[method].to_numpy(dtype=float)
        boot_auroc = [auroc(u[idx], correct[idx]) for idx in resamples]
        boot_prr = [prr(u[idx], correct[idx], n_tie_seeds=BOOT_TIE_SEEDS, seed=seed) for idx in resamples]
        boot_prr = [value for value in boot_prr if not math.isnan(value)]
        rows[method] = {
            "auroc": auroc(u, correct),
            "auroc_low": float(np.percentile(boot_auroc, low_q)),
            "auroc_high": float(np.percentile(boot_auroc, high_q)),
            "prr": prr(u, correct),
            "prr_low": float(np.percentile(boot_prr, low_q)),
            "prr_high": float(np.percentile(boot_prr, high_q)),
        }
    ranking = pd.DataFrame.from_dict(rows, orient="index")
    ranking.index.name = "method"
    return ranking.sort_values("auroc", ascending=False)


def main() -> None:
    """Print the ranking of one model's methods on its calibration part."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen1.5b")
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    table = load_score_table(args.model, args.dataset, args.n)
    split = make_split(table, args.calibration_size, args.seed)
    calibration = subset(table, split.calibration)
    ranking = rank_methods(calibration, n_boot=args.n_boot, seed=args.seed)

    print(
        f"{table.model} / {table.run}: calibration part, {len(split.calibration)} questions, "
        f"{int(calibration.correct.sum())} correct, {args.n_boot} bootstrap resamples"
    )
    print(f"{'method':<27} {'AUROC':>6}  {'95% interval':<15} {'PRR':>6}  {'95% interval':<15}")
    for method, row in ranking.iterrows():
        print(
            f"{method:<27} {row.auroc:>6.3f}  [{row.auroc_low:.3f}, {row.auroc_high:.3f}]  "
            f"{row.prr:>6.3f}  [{row.prr_low:+.3f}, {row.prr_high:+.3f}]"
        )


if __name__ == "__main__":
    main()
