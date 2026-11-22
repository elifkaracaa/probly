"""Sample-count curve and early-stop rule for the semantic methods (B2).

Reads only the cache (``data/<model>/<dataset>_<n>.jsonl`` and its ``.nli.npz``),
so it needs no LLM and no NLI model. Run from ``experiments/llm-probly``::

    uv run python -m scripts.sample_count_curve --models qwen0.5b qwen1.5b

For each model it writes:

- ``results/<model>/<dataset>_<n>/analysis/sample_count.json``: AUROC / PRR per
  method and K (mean and standard deviation over random subsets), and the
  early-stop table.
- ``results/<model>/<dataset>_<n>/analysis/sample_count.md``: the same as Markdown tables.

The files go to ``analysis/`` because the summary table reads every ``*.json``
next to the method results as a method.
- ``results/plots/<dataset>_<n>_<model>_k_curve.png``: AUROC vs number of samples.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib as mpl

mpl.use("Agg")  # write files only, no window
from common.data import cache_path, load_jsonl
from common.evaluate import auroc, prr
from common.io import RESULTS_ROOT
from common.nli_matrix import load_nli_cache
from common.sample_count import METHODS, early_stop_scores, k_curve, tied
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "semantic_entropy": "SE",
    "discrete_semantic_entropy": "discrete SE",
    "lexical_similarity": "lexical similarity",
}


def load_model(model: str, dataset: str, n: int) -> tuple[list[dict], list, np.ndarray]:
    """Rows, NLI matrices and correctness labels of one model, checked to be aligned."""
    rows = load_jsonl(cache_path(model, n, dataset))
    cache = load_nli_cache(cache_path(model, n, dataset, suffix=".nli.npz"))
    if cache["ids"] != [r["id"] for r in rows]:
        msg = f"{model}: question ids of the NLI cache and the data cache differ."
        raise ValueError(msg)
    correct = np.array([bool(r["correct"]) for r in rows])
    return rows, cache["matrices"], correct


def curve_table(curve: dict[int, np.ndarray], correct: np.ndarray) -> list[dict]:
    """AUROC and PRR per method and K, averaged over the random subsets."""
    table = []
    for k, values in curve.items():
        for m, method in enumerate(METHODS):
            aurocs = [auroc(tied(draw[m]), correct) for draw in values]
            prrs = [prr(tied(draw[m]), correct) for draw in values]
            table.append(
                {
                    "method": method,
                    "k": k,
                    "draws": len(values),
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "prr_mean": float(np.mean(prrs)),
                }
            )
    return table


def early_stop_table(matrices: list, rows: list[dict], correct: np.ndarray, checks: list[int]) -> list[dict]:
    """Saved generations and AUROC change of the early-stop rule, per check point."""
    n_samples = len(rows[0]["samples"])
    full, _ = early_stop_scores(matrices, rows, n_samples)
    full = tied(full)
    table = []
    for check_at in checks:
        values, used = early_stop_scores(matrices, rows, check_at)
        values = tied(values)
        stopped = used < n_samples
        for m, method in enumerate(METHODS):
            table.append(
                {
                    "method": method,
                    "check_at": check_at,
                    "stopped_share": float(np.mean(stopped)),
                    "stopped_accuracy": float(correct[stopped].mean()) if stopped.any() else float("nan"),
                    "not_stopped_accuracy": float(correct[~stopped].mean()) if (~stopped).any() else float("nan"),
                    "samples_per_question": float(used.mean()),
                    "saved_generations_share": float(1 - used.mean() / n_samples),
                    "auroc": auroc(values[m], correct),
                    "auroc_full": auroc(full[m], correct),
                    "auroc_change": auroc(values[m], correct) - auroc(full[m], correct),
                }
            )
    return table


def to_markdown(model: str, run: str, curve: list[dict], stop: list[dict]) -> str:
    """Both tables as Markdown."""
    ks = sorted({r["k"] for r in curve})
    lines = [
        f"# Sample-count curve: `{model}/{run}`",
        "",
        "AUROC (mean over random subsets of the 10 cached samples; std in brackets).",
        "",
        "| method | " + " | ".join(f"K={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for method in METHODS:
        cells = [f"{r['auroc_mean']:.3f} ({r['auroc_std']:.3f})" for r in curve if r["method"] == method]
        lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Early stop",
        "",
        "Rule: generate `check_at` samples; stop if they all fall into one semantic cluster,",
        "otherwise generate all 10. AUROC change is against all 10 samples.",
        "",
        "Accuracy columns: share of correct greedy answers among the stopped and the not-stopped",
        "questions. A stopped wrong answer gets entropy 0 and ties with the stopped correct ones,",
        "which is where most of the AUROC change comes from.",
        "",
        (
            "| method | check_at | stopped | accuracy stopped | accuracy not stopped "
            "| samples / question | saved generations | AUROC | AUROC change |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {LABELS[r['method']]} | {r['check_at']} | {r['stopped_share']:.0%} | "
        f"{r['stopped_accuracy']:.0%} | {r['not_stopped_accuracy']:.0%} | "
        f"{r['samples_per_question']:.2f} | {r['saved_generations_share']:.0%} | "
        f"{r['auroc']:.3f} | {r['auroc_change']:+.3f} |"
        for r in stop
    ]
    return "\n".join(lines) + "\n"


def plot_curve(curve: list[dict], model: str, run: str) -> None:
    """AUROC vs number of samples, one line per method, band = one std over subsets."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHODS:
        part = [r for r in curve if r["method"] == method]
        k = np.array([r["k"] for r in part])
        mean = np.array([r["auroc_mean"] for r in part])
        std = np.array([r["auroc_std"] for r in part])
        (line,) = ax.plot(k, mean, marker="o", label=LABELS[method])
        ax.fill_between(k, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("samples per question (K)")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{model} {run}: quality vs number of samples")
    ax.grid(alpha=0.3)
    ax.legend()
    out = RESULTS_ROOT / "plots" / f"{run}_{model}_k_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    """Run the curve and the early-stop table for every model."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["qwen0.5b", "qwen1.5b"])
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--draws", type=int, default=20, help="random subsets per K")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checks", type=int, nargs="+", default=[2, 3, 4, 5], help="early-stop check points")
    a = ap.parse_args()

    run = f"{a.dataset}_{a.n}"
    for model in a.models:
        start = time.perf_counter()
        rows, matrices, correct = load_model(model, a.dataset, a.n)
        ks = list(range(2, len(rows[0]["samples"]) + 1))
        curve = curve_table(k_curve(matrices, rows, ks, a.draws, a.seed), correct)
        stop = early_stop_table(matrices, rows, correct, a.checks)

        out_dir = RESULTS_ROOT / model / run / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = {"draws": a.draws, "seed": a.seed, "checks": a.checks}
        record = {"model": model, "run": run, "settings": settings, "curve": curve, "early_stop": stop}
        (out_dir / "sample_count.json").write_text(json.dumps(record, indent=2) + "\n")
        (out_dir / "sample_count.md").write_text(to_markdown(model, run, curve, stop))
        plot_curve(curve, model, run)
        print(f"{model}: done in {time.perf_counter() - start:.0f}s -> {out_dir / 'sample_count.md'}")


if __name__ == "__main__":
    main()
