"""Step 4 of the selector: choose one method per model and explain why.

The rule, applied on the calibration part only:

1. Keep the candidates the user can run: access level and cost budget.
2. Find the method with the highest AUROC (the "best").
3. For every other method, compute the AUROC difference to the best on each
   bootstrap resample (paired: both methods are scored on the same questions).
   A method is "tied with the best" if the 95 % interval of that difference
   contains 0, i.e. the data cannot tell it apart from the best.
4. Among the best and the tied methods, choose the cheapest. Equal cost: higher AUROC.

The paired test replaces "the two AUROC intervals overlap": separate intervals
overlap even when one method is clearly better (on qwen3b, msp's interval touches
p_true_zeroshot's despite a 0.145 AUROC gap), so overlap would pick msp there.

Run from ``experiments/llm-probly``::

    uv run python -m selector.select                 # all models, print the choice
    uv run python -m selector.select --save          # also calibrate (D3) and write profiles/<model>.json
    uv run python -m selector.select --access text_only --max-cost 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from typing import TYPE_CHECKING

from common.evaluate import auroc
from common.io import RESULTS_ROOT
from common.registry import ACCESS_LEVELS, REGISTRY
import numpy as np

from selector.calibrate import calibrate_profile
from selector.profile import MethodChoice, Metrics, Profile, save_profile
from selector.rank import DEFAULT_ALPHA, DEFAULT_N_BOOT, bootstrap_indices, rank_methods
from selector.split import DEFAULT_CALIBRATION_SIZE, DEFAULT_SEED, Split, make_split, subset
from selector.table import load_score_table

if TYPE_CHECKING:
    import pandas as pd

    from selector.table import ScoreTable

# Family, access level and cost of every method come from E2's registry (common/registry.py).
DEFAULT_REGISTRY: dict[str, dict] = {name: spec.to_dict() for name, spec in REGISTRY.items()}

# What each access level allows, from the registry's order (text_only < logprobs < white_box):
# a white-box model can run everything, an API without logprobs only text methods.
# Methods that failed on more than this share of the calibration questions are not candidates:
# a failed question gets a placeholder score (uncertainty 1.0, see CONTRACT.md), not a real one.
DEFAULT_MAX_FAILED = 0.05

ALLOWED_ACCESS = {level: set(ACCESS_LEVELS[: i + 1]) for i, level in enumerate(ACCESS_LEVELS)}


@dataclass
class Selection:
    """The selector's choice for one model, with everything needed to explain it.

    Attributes:
        model: Short model name.
        chosen: The selected method.
        best: The method with the highest calibration AUROC among the candidates.
        tied: Methods whose AUROC difference to the best has a 95 % interval containing 0.
        differences: Per compared method: AUROC difference to the best and its interval
            (``diff``, ``low``, ``high``; negative = worse than the best).
        ranking: Step 3's table for the candidates (AUROC, PRR and intervals).
        excluded: Methods that were not candidates, with the reason.
        reason: One sentence for the demo and the profile.
    """

    model: str
    chosen: str
    best: str
    tied: list[str]
    differences: dict[str, dict[str, float]]
    ranking: pd.DataFrame
    excluded: dict[str, str] = field(default_factory=dict)
    reason: str = ""


def filter_candidates(
    methods: list[str],
    registry: dict[str, dict],
    access: str = "white_box",
    max_cost: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Split the methods into candidates and excluded ones (with the reason).

    Args:
        methods: Methods that have scores for this model.
        registry: Family, access level and cost per method.
        access: What the user has: ``white_box``, ``logprobs`` or ``text_only``.
        max_cost: Highest cost per question the user accepts; ``None`` = no limit.

    Returns:
        The candidate names and a dict of excluded names with their reason.
    """
    if access not in ALLOWED_ACCESS:
        msg = f"access must be one of {list(ALLOWED_ACCESS)}, got {access!r}"
        raise ValueError(msg)
    candidates, excluded = [], {}
    for method in methods:
        entry = registry.get(method)
        if entry is None:
            excluded[method] = "not in the registry (no access level / cost)"
        elif entry["access"] not in ALLOWED_ACCESS[access]:
            excluded[method] = f"needs {entry['access']} access"
        elif max_cost is not None and entry["cost"] > max_cost:
            excluded[method] = f"cost {entry['cost']:g} > budget {max_cost:g}"
        else:
            candidates.append(method)
    return candidates, excluded


def paired_differences(
    table: ScoreTable,
    best: str,
    others: list[str],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """AUROC of each other method minus the best's, with a paired bootstrap interval.

    Args:
        table: Scores and labels (the calibration part).
        best: The method to compare against.
        others: Methods to compare.
        n_boot: Number of bootstrap resamples (the same ones as in step 3).
        alpha: 1 - confidence level of the intervals.
        seed: Random seed for the resamples.

    Returns:
        Per method: ``diff`` (point estimate), ``low`` and ``high`` (interval bounds).
    """
    correct = table.correct.to_numpy(dtype=bool)
    resamples = bootstrap_indices(correct, n_boot, seed)
    u_best = table.scores[best].to_numpy(dtype=float)
    boot_best = np.array([auroc(u_best[idx], correct[idx]) for idx in resamples])
    result = {}
    for method in others:
        u = table.scores[method].to_numpy(dtype=float)
        boot = np.array([auroc(u[idx], correct[idx]) for idx in resamples]) - boot_best
        low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        result[method] = {
            "diff": auroc(u, correct) - auroc(u_best, correct),
            "low": float(low),
            "high": float(high),
        }
    return result


def select_method(
    table: ScoreTable,
    registry: dict[str, dict] | None = None,
    access: str = "white_box",
    max_cost: float | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_failed: float = DEFAULT_MAX_FAILED,
) -> Selection:
    """Choose the cheapest method that is statistically tied with the best one.

    Args:
        table: Scores and labels of the calibration part (steps 1 and 2).
        registry: Family, access level and cost per method. Defaults to ``DEFAULT_REGISTRY``
            (from ``common/registry.py``).
        access: What the user has: ``white_box``, ``logprobs`` or ``text_only``.
        max_cost: Highest accepted cost per question; ``None`` = no limit.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for the resamples.
        max_failed: Methods with a larger share of failed questions are excluded.

    Returns:
        The selection with its reason.

    Raises:
        ValueError: If no method is left after filtering.
    """
    registry = DEFAULT_REGISTRY if registry is None else registry
    candidates, excluded = filter_candidates(table.methods, registry, access, max_cost)
    if "failed_fraction" in table.info.columns:
        for method in list(candidates):
            failed = float(table.info.loc[method, "failed_fraction"]) if method in table.info.index else 0.0
            if failed > max_failed:
                candidates.remove(method)
                excluded[method] = f"failed on {failed:.0%} of the questions (placeholder scores)"
    if not candidates:
        msg = f"{table.model}: no candidate method left (access {access}, budget {max_cost}); excluded: {excluded}"
        raise ValueError(msg)

    ranking = rank_methods(replace(table, scores=table.scores[candidates]), n_boot=n_boot, seed=seed)
    best = str(ranking.index[0])
    others = [m for m in ranking.index if m != best]
    differences = paired_differences(table, best, others, n_boot=n_boot, seed=seed)
    tied = [m for m in others if differences[m]["high"] >= 0]

    # Cheapest among the best and its ties; equal cost -> higher AUROC; equal AUROC -> name (stable).
    pool = [best, *tied]
    chosen = min(pool, key=lambda m: (registry[m]["cost"], -ranking.loc[m, "auroc"], m))

    selection = Selection(
        model=table.model,
        chosen=chosen,
        best=best,
        tied=tied,
        differences=differences,
        ranking=ranking,
        excluded=excluded,
    )
    selection.reason = explain(selection, registry)
    return selection


def explain(selection: Selection, registry: dict[str, dict]) -> str:
    """One sentence on why the chosen method was selected."""
    chosen, best = selection.chosen, selection.best
    a = selection.ranking.loc[chosen, "auroc"]
    if chosen == best:
        if not selection.tied:
            return f"{chosen} has the highest AUROC ({a:.3f}) and every other method is significantly worse."
        cheaper = [m for m in selection.tied if registry[m]["cost"] < registry[best]["cost"]]
        if not cheaper:
            return f"{chosen} has the highest AUROC ({a:.3f}); the tied methods are not cheaper."
        return f"{chosen} has the highest AUROC ({a:.3f})."
    d = selection.differences[chosen]
    ratio = registry[best]["cost"] / registry[chosen]["cost"]
    return (
        f"{chosen} (AUROC {a:.3f}) is not significantly worse than the best, {best} "
        f"(difference {d['diff']:+.3f}, 95 % interval [{d['low']:+.3f}, {d['high']:+.3f}]), "
        f"and costs {ratio:g}x less."
    )


def to_metrics(row: pd.Series) -> Metrics:
    """Turn one row of step 3's ranking into the profile's Metrics."""
    return Metrics(**{key: round(float(row[key]), 4) for key in Metrics.__dataclass_fields__})


def build_profile(
    selection: Selection,
    table: ScoreTable,
    split: Split,
    registry: dict[str, dict] | None = None,
    dataset: str = "triviaqa",
    model_id: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> Profile:
    """Turn a selection into the D4 profile, with the chosen method's metrics on both parts.

    The calibration map is filled afterwards by ``selector.calibrate.calibrate_profile``.

    Args:
        selection: Result of ``select_method``.
        table: The full table of the model (both parts), for the run size.
        split: The split the selection was made on.
        registry: The registry the selection used.
        dataset: Dataset name.
        model_id: Hugging Face id of the model. Read from the chosen method's result file when ``None``.
        n_boot: Bootstrap resamples for the held-out intervals.

    Returns:
        A profile ready for ``save_profile``.
    """
    registry = DEFAULT_REGISTRY if registry is None else registry
    entry = registry[selection.chosen]
    held_out = subset(table, split.held_out)
    held_out_ranking = rank_methods(
        replace(held_out, scores=held_out.scores[[selection.chosen]]), n_boot=n_boot, seed=split.seed
    )
    if model_id is None:
        result_file = RESULTS_ROOT / table.model / table.run / f"{selection.chosen}.json"
        model_id = json.loads(result_file.read_text())["model"]
    return Profile(
        model=table.model,
        model_id=model_id,
        dataset=dataset,
        n_questions=len(table.correct),
        split_seed=split.seed,
        calibration_size=len(split.calibration),
        method=MethodChoice(
            name=selection.chosen,
            family=entry["family"],
            access=entry["access"],
            cost=float(entry["cost"]),
            probability=bool(table.info.loc[selection.chosen, "probability"]),
        ),
        reason=selection.reason,
        calibration_metrics=to_metrics(selection.ranking.loc[selection.chosen]),
        held_out_metrics=to_metrics(held_out_ranking.loc[selection.chosen]),
        candidates={str(m): to_metrics(row) for m, row in selection.ranking.iterrows()},
    )


def print_selection(selection: Selection, registry: dict[str, dict]) -> None:
    """Print the choice, the comparison with the best, and the exclusions."""
    print(f"\n{selection.model}: chosen {selection.chosen}")
    print(f"  {selection.reason}")
    print(f"  {'method':<27} {'cost':>5} {'AUROC':>6}  {'diff to best':>12}  {'95% interval':<17} tied")
    for method, row in selection.ranking.iterrows():
        cost = registry[str(method)]["cost"]
        if method == selection.best:
            print(f"  {method:<27} {cost:>5g} {row.auroc:>6.3f}  {'(best)':>12}")
            continue
        d = selection.differences[str(method)]
        tied = "yes" if method in selection.tied else "no"
        print(
            f"  {method:<27} {cost:>5g} {row.auroc:>6.3f}  {d['diff']:>+12.3f}  "
            f"[{d['low']:+.3f}, {d['high']:+.3f}]  {tied}"
        )
    for method, why in selection.excluded.items():
        print(f"  excluded {method}: {why}")


def main() -> None:
    """Run the selector for one or all models."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen0.5b,qwen1.5b,qwen3b,qwen7b")
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    parser.add_argument("--access", default="white_box", choices=list(ALLOWED_ACCESS))
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save", action="store_true", help="write profiles/<model>.json")
    args = parser.parse_args()

    for model in args.models.split(","):
        table = load_score_table(model, args.dataset, args.n)
        split = make_split(table, args.calibration_size, args.seed)
        selection = select_method(
            subset(table, split.calibration),
            access=args.access,
            max_cost=args.max_cost,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        print_selection(selection, DEFAULT_REGISTRY)
        if args.save:
            profile = build_profile(selection, table, split, dataset=args.dataset)
            u = table.scores[selection.chosen]
            profile, brier = calibrate_profile(
                profile,
                u.loc[split.calibration],
                table.correct.loc[split.calibration],
                u.loc[split.held_out],
                table.correct.loc[split.held_out],
            )
            cal = profile.calibration
            before = "n/a (raw score)" if cal.ece_before is None else f"{cal.ece_before:.3f}"
            print(
                f"  calibration: {cal.type} (CV Brier platt {brier['platt']:.4f}, isotonic {brier['isotonic']:.4f}); "
                f"held-out ECE before {before}, after {cal.ece_after:.3f}"
            )
            print(f"  saved {save_profile(profile)}")


if __name__ == "__main__":
    main()
