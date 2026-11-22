"""Held-out evaluation: does "pick a method per model" beat "always use the same method"?

This is the project's headline check. For every model, the selector chooses on the
calibration part (steps 1-4). Then, on the held-out part, which the selector never
saw, it is compared with every fixed method, i.e. using the same method for every model.

Per model the table shows the selector's held-out AUROC and cost, the held-out AUROC
of every fixed method, and the paired bootstrap difference "selector - best fixed
method on that model". The last row averages over the models: if the selector's mean
AUROC matches the best fixed method's at a lower mean cost, picking per model is worth it.

Note: "best fixed method on that model" is chosen on the held-out part itself, so it is
an optimistic (oracle) baseline; the selector has to come close to it without seeing those questions.

Run from ``experiments/llm-probly``::

    uv run python -m selector.evaluate
    uv run python -m selector.evaluate --out results/selector_held_out.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from common.evaluate import auroc
import numpy as np

from selector.rank import DEFAULT_N_BOOT
from selector.select import DEFAULT_REGISTRY, paired_differences, select_method
from selector.split import DEFAULT_CALIBRATION_SIZE, DEFAULT_SEED, make_split, subset
from selector.table import load_score_table

DEFAULT_MODELS = "qwen0.5b,qwen1.5b,qwen3b,qwen7b"


@dataclass
class ModelResult:
    """Held-out result of one model.

    Attributes:
        model: Short model name.
        chosen: Method the selector chose on the calibration part.
        chosen_auroc: Held-out AUROC of the chosen method.
        chosen_cost: Cost of the chosen method (registry).
        fixed_auroc: Held-out AUROC of every method of this model.
        oracle: Best method on the held-out part (optimistic baseline).
        difference: ``diff``, ``low``, ``high`` of AUROC(chosen) - AUROC(oracle), paired bootstrap.
    """

    model: str
    chosen: str
    chosen_auroc: float
    chosen_cost: float
    fixed_auroc: dict[str, float]
    oracle: str
    difference: dict[str, float]


def evaluate_model(
    model: str,
    dataset: str = "triviaqa",
    n: int = 200,
    calibration_size: int = DEFAULT_CALIBRATION_SIZE,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> ModelResult:
    """Select on the calibration part and evaluate on the held-out part of one model.

    Args:
        model: Short model name.
        dataset: Dataset name.
        n: Number of questions in the run.
        calibration_size: Questions the selector may use.
        n_boot: Bootstrap resamples (selection and held-out interval).
        seed: Random seed (split and bootstrap).

    Returns:
        The model's held-out result.
    """
    table = load_score_table(model, dataset, n)
    split = make_split(table, calibration_size, seed)
    selection = select_method(subset(table, split.calibration), n_boot=n_boot, seed=seed)
    held_out = subset(table, split.held_out)
    y = held_out.correct.to_numpy(dtype=bool)
    fixed = {m: auroc(held_out.scores[m].to_numpy(dtype=float), y) for m in held_out.methods if m in DEFAULT_REGISTRY}
    oracle = max(fixed, key=lambda m: fixed[m])
    if selection.chosen == oracle:
        difference = {"diff": 0.0, "low": 0.0, "high": 0.0}
    else:
        difference = paired_differences(held_out, oracle, [selection.chosen], n_boot=n_boot, seed=seed)[
            selection.chosen
        ]
    return ModelResult(
        model=model,
        chosen=selection.chosen,
        chosen_auroc=fixed[selection.chosen],
        chosen_cost=float(DEFAULT_REGISTRY[selection.chosen]["cost"]),
        fixed_auroc=fixed,
        oracle=oracle,
        difference=difference,
    )


def common_methods(results: list[ModelResult]) -> list[str]:
    """Methods that exist for every model, ordered by mean held-out AUROC (best first)."""
    shared = set.intersection(*(set(r.fixed_auroc) for r in results))
    return sorted(shared, key=lambda m: -float(np.mean([r.fixed_auroc[m] for r in results])))


def to_markdown(results: list[ModelResult], dataset: str, n: int, calibration_size: int) -> str:
    """Render the held-out comparison as a Markdown report."""
    methods = common_methods(results)
    lines = [
        f"# Selector vs fixed methods (held-out, {dataset} {n}, {n - calibration_size} held-out questions per model)",
        "",
        "The selector chooses on the calibration part only; every number below is on the held-out part.",
        "",
        (
            "| Model | Selector's choice | Cost | Held-out AUROC | Best fixed on this model (oracle) "
            "| Selector - oracle [95 % CI] |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        d = r.difference
        lines.append(
            f"| {r.model} | {r.chosen} | {r.chosen_cost:g} | {r.chosen_auroc:.3f} | "
            f"{r.oracle} {r.fixed_auroc[r.oracle]:.3f} | {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] |"
        )
    lines += [
        "",
        "Mean over the models (selector vs always using one method):",
        "",
        "| Strategy | Mean held-out AUROC | Mean cost |",
        "|---|---|---|",
        (
            f"| **Selector (per model)** | **{np.mean([r.chosen_auroc for r in results]):.3f}** "
            f"| **{np.mean([r.chosen_cost for r in results]):.1f}** |"
        ),
    ]
    for m in methods:
        mean_auroc = np.mean([r.fixed_auroc[m] for r in results])
        lines.append(f"| always {m} | {mean_auroc:.3f} | {DEFAULT_REGISTRY[m]['cost']:g} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Evaluate the selector on every model and print (or save) the Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this Markdown file")
    args = parser.parse_args()

    results = [
        evaluate_model(m, args.dataset, args.n, args.calibration_size, args.n_boot, args.seed)
        for m in args.models.split(",")
    ]
    report = to_markdown(results, args.dataset, args.n, args.calibration_size)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
