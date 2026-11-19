"""Step 2 of the selector: split one model's questions into a calibration and a held-out part.

The selector chooses the method (and later fits the combiner and the calibration)
on the calibration part only. The held-out part is never looked at until the
final check, so the headline number "the selector is as good as the best fixed
method" is measured on questions the selector has not seen.

The split is stratified: both parts keep the model's share of correct answers.
Without that, a small model like qwen0.5b (19 % correct) could end up with almost
no correct answers in one part, and AUROC would be meaningless there.

Run from ``experiments/llm-probly``::

    uv run python -m selector.split --model qwen0.5b
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from sklearn.model_selection import train_test_split

from selector.table import ScoreTable, load_score_table

DEFAULT_CALIBRATION_SIZE = 100
DEFAULT_SEED = 0
# AUROC needs both classes; fewer than this many of one class per part is too few to rank methods.
MIN_PER_CLASS = 5


@dataclass(frozen=True)
class Split:
    """Question ids of the two parts, each sorted.

    Attributes:
        calibration: Ids the selector may use to choose and fit.
        held_out: Ids kept back for the final evaluation.
        seed: Random seed that produced this split (stored in the profile, D4).
    """

    calibration: list[int]
    held_out: list[int]
    seed: int


def make_split(table: ScoreTable, calibration_size: int = DEFAULT_CALIBRATION_SIZE, seed: int = DEFAULT_SEED) -> Split:
    """Draw a stratified calibration / held-out split of the table's questions.

    Args:
        table: Scores and labels of one model (step 1).
        calibration_size: Number of questions in the calibration part.
        seed: Random seed. The same seed and table always give the same split.

    Returns:
        The two id lists.

    Raises:
        ValueError: If the sizes do not fit, or a part would get fewer than
            ``MIN_PER_CLASS`` correct or wrong answers.
    """
    ids = list(table.correct.index)
    n = len(ids)
    if not 0 < calibration_size < n:
        msg = f"calibration_size must be between 1 and {n - 1} for {n} questions, got {calibration_size}"
        raise ValueError(msg)

    calibration, held_out = train_test_split(
        ids,
        train_size=calibration_size,
        stratify=table.correct.to_numpy(),
        random_state=seed,
    )
    split = Split(calibration=sorted(calibration), held_out=sorted(held_out), seed=seed)

    for part_name, part in (("calibration", split.calibration), ("held-out", split.held_out)):
        n_correct = int(table.correct.loc[part].sum())
        n_wrong = len(part) - n_correct
        if min(n_correct, n_wrong) < MIN_PER_CLASS:
            msg = (
                f"{table.model}: {part_name} part has {n_correct} correct and {n_wrong} wrong answers; "
                f"need at least {MIN_PER_CLASS} of each"
            )
            raise ValueError(msg)
    return split


def subset(table: ScoreTable, ids: list[int]) -> ScoreTable:
    """Return a copy of the table that only contains the given question ids."""
    return replace(table, scores=table.scores.loc[ids], correct=table.correct.loc[ids])


def main() -> None:
    """Print the split sizes and the share of correct answers in each part."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen1.5b")
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    table = load_score_table(args.model, args.dataset, args.n)
    split = make_split(table, args.calibration_size, args.seed)
    print(f"{table.model} / {table.run}, seed {split.seed}")
    for part_name, part in (("calibration", split.calibration), ("held-out", split.held_out)):
        part_table = subset(table, part)
        n_correct = int(part_table.correct.sum())
        print(f"  {part_name:<12} {len(part):>4} questions, {n_correct:>3} correct ({n_correct / len(part):.1%})")


if __name__ == "__main__":
    main()
