"""D2: combine several methods' scores into one with logistic-regression stacking.

Each method looks at a different signal (token probabilities, self-evaluation,
agreement between samples), so a combination might beat the best single method.
The stack is a logistic regression that predicts "the answer is correct" from the
standardized scores of all methods; its uncertainty is ``1 - P(correct)``.

It is always reported against the best single method, in two ways:

- Calibration part: 5-fold cross-validation where, in every fold, both the stack and
  the choice of the best single method are made on the training folds only.
- Held-out part: the stack is fitted on the whole calibration part and scored on
  the held-out questions, with a paired bootstrap interval of
  "AUROC(stack) - AUROC(best single)".

Run from ``experiments/llm-probly``::

    uv run python -m selector.combine --models qwen0.5b,qwen1.5b
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from common.evaluate import auroc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from selector.rank import DEFAULT_N_BOOT
from selector.select import paired_differences
from selector.split import DEFAULT_CALIBRATION_SIZE, DEFAULT_SEED, make_split, subset
from selector.table import load_score_table

if TYPE_CHECKING:
    from selector.table import ScoreTable

DEFAULT_FOLDS = 5
STACK = "stack"
# Regularization strengths tried by the inner cross-validation (small C = strong regularization).
C_GRID = np.logspace(-3, 3, 13)


def fit_stack(table: ScoreTable, methods: list[str] | None = None, seed: int = DEFAULT_SEED) -> GridSearchCV:
    """Fit the stacking model on a table's scores and labels.

    The regularization strength is chosen by an inner 5-fold cross-validation over
    ``C_GRID``, which matters with only ~100 questions and ~10 correlated methods.

    Args:
        table: Scores and labels to fit on (the calibration part, or a fold of it).
        methods: Methods to use as features. All methods of the table when ``None``.
        seed: Random seed of the inner cross-validation folds.

    Returns:
        The fitted search; ``best_estimator_`` is the standardization + logistic regression pipeline.
    """
    methods = table.methods if methods is None else methods
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": C_GRID},
        cv=StratifiedKFold(n_splits=DEFAULT_FOLDS, shuffle=True, random_state=seed),
        scoring="roc_auc",
    )
    search.fit(table.scores[methods].to_numpy(dtype=float), table.correct.to_numpy(dtype=bool))
    return search


def stack_uncertainty(model: GridSearchCV, table: ScoreTable, methods: list[str]) -> np.ndarray:
    """Uncertainty of the stack for every question of the table: ``1 - P(correct)``."""
    return 1.0 - model.predict_proba(table.scores[methods].to_numpy(dtype=float))[:, 1]


def stack_weights(model: GridSearchCV, methods: list[str]) -> dict[str, float]:
    """Weight of each standardized method score; negative = higher uncertainty means "more likely wrong"."""
    coefficients = model.best_estimator_[-1].coef_[0]
    return {method: float(w) for method, w in zip(methods, coefficients, strict=True)}


def cross_validated_comparison(
    table: ScoreTable, methods: list[str], folds: int = DEFAULT_FOLDS, seed: int = DEFAULT_SEED
) -> tuple[float, float]:
    """Fair cross-validated AUROC of the stack and of "pick the best single method".

    In every fold, both are decided on the training folds only and scored on the
    left-out fold: the stack is fitted there, and the best single method is the one
    with the highest training AUROC. The fold AUROCs are averaged (not pooled), because
    the stacks of different folds give scores on slightly different scales.

    Args:
        table: Scores and labels (the calibration part).
        methods: Candidate methods / stack features.
        folds: Number of outer folds.
        seed: Random seed of the folds.

    Returns:
        Mean fold AUROC of the stack and of the best-single-method rule.
    """
    y = table.correct.to_numpy(dtype=bool)
    ids = table.correct.index
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    stack_scores, single_scores = [], []
    for train, test in splitter.split(np.zeros(len(y)), y):
        train_table, test_table = subset(table, list(ids[train])), subset(table, list(ids[test]))
        y_train, y_test = y[train], y[test]
        best = max(methods, key=lambda m: auroc(train_table.scores[m].to_numpy(dtype=float), y_train))
        single_scores.append(auroc(test_table.scores[best].to_numpy(dtype=float), y_test))
        model = fit_stack(train_table, methods, seed=seed)
        stack_scores.append(auroc(stack_uncertainty(model, test_table, methods), y_test))
    return float(np.mean(stack_scores)), float(np.mean(single_scores))


@dataclass
class StackReport:
    """Stack vs best single method for one model.

    Attributes:
        model: Short model name.
        methods: Methods used as features.
        best: Best single method by AUROC on the calibration part.
        cv_best_auroc: Cross-validated AUROC of "pick the best single method" on the calibration part.
        cv_stack_auroc: Cross-validated AUROC of the stack on the calibration part.
        held_out_best_auroc: AUROC of ``best`` on the held-out part.
        held_out_stack_auroc: AUROC of the stack (fitted on the calibration part) on the held-out part.
        held_out_difference: ``diff``, ``low`` and ``high`` of AUROC(stack) - AUROC(best) on the
            held-out part, with a paired 95 % bootstrap interval.
        weights: Weight of each method in the stack fitted on the whole calibration part.
    """

    model: str
    methods: list[str]
    best: str
    cv_best_auroc: float
    cv_stack_auroc: float
    held_out_best_auroc: float
    held_out_stack_auroc: float
    held_out_difference: dict[str, float]
    weights: dict[str, float]


def compare_stack(
    calibration: ScoreTable,
    held_out: ScoreTable,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> StackReport:
    """Fit the stack on the calibration part and compare it with the best single method.

    Args:
        calibration: Scores and labels the stack and the best method are chosen on.
        held_out: Scores and labels used only for the final comparison.
        n_boot: Number of bootstrap resamples for the held-out interval.
        seed: Random seed.

    Returns:
        The comparison on both parts.
    """
    methods = calibration.methods
    y_cal = calibration.correct.to_numpy(dtype=bool)
    single = {m: auroc(calibration.scores[m].to_numpy(dtype=float), y_cal) for m in methods}
    best = max(single, key=lambda m: single[m])

    cv_stack, cv_best = cross_validated_comparison(calibration, methods, seed=seed)
    model = fit_stack(calibration, methods, seed=seed)
    with_stack = replace(
        held_out, scores=held_out.scores.assign(**{STACK: stack_uncertainty(model, held_out, methods)})
    )
    difference = paired_differences(with_stack, best, [STACK], n_boot=n_boot, seed=seed)[STACK]
    y_out = held_out.correct.to_numpy(dtype=bool)

    return StackReport(
        model=calibration.model,
        methods=methods,
        best=best,
        cv_best_auroc=cv_best,
        cv_stack_auroc=cv_stack,
        held_out_best_auroc=auroc(held_out.scores[best].to_numpy(dtype=float), y_out),
        held_out_stack_auroc=auroc(with_stack.scores[STACK].to_numpy(dtype=float), y_out),
        held_out_difference=difference,
        weights=stack_weights(model, methods),
    )


def main() -> None:
    """Print stack vs best single method for each model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen0.5b,qwen1.5b,qwen3b,qwen7b")
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"{'model':<9} {'best single':<26} {'5-fold CV AUROC':>20} {'held-out AUROC':>17}   held-out stack - best")
    print(f"{'':<9} {'':<26} {'best':>9} {'stack':>10} {'best':>8} {'stack':>8}")
    reports = []
    for model_name in args.models.split(","):
        table = load_score_table(model_name, args.dataset, args.n)
        split = make_split(table, args.calibration_size, args.seed)
        report = compare_stack(
            subset(table, split.calibration), subset(table, split.held_out), n_boot=args.n_boot, seed=args.seed
        )
        reports.append(report)
        d = report.held_out_difference
        print(
            f"{report.model:<9} {report.best:<26} {report.cv_best_auroc:>9.3f} "
            f"{report.cv_stack_auroc:>10.3f} {report.held_out_best_auroc:>8.3f} "
            f"{report.held_out_stack_auroc:>8.3f}   {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"
        )
    print("\nStack weights (standardized scores; the larger |w|, the more the stack relies on the method):")
    for report in reports:
        top = sorted(report.weights.items(), key=lambda item: -abs(item[1]))
        print(f"  {report.model:<9} " + ", ".join(f"{m} {w:+.2f}" for m, w in top if abs(w) > 1e-3))


if __name__ == "__main__":
    main()
