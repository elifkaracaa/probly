"""Decomposed semantic entropy on a 50-question subset (B4 example).

Clarifications cost extra generations (3 rephrasings + 15 answers per question),
so this runs the plugin on the first ``--questions`` cached questions of one
model instead of the full benchmark. Run from ``experiments/llm-probly``::

    uv run python -m scripts.decomposed_example --model qwen0.5b --questions 50

Writes ``results/<model>/<dataset>_<n>/analysis/decomposed_se.{json,md}``: mean
total / aleatoric / epistemic uncertainty, their AUROC for the greedy answer's
correctness, and the questions with the highest aleatoric and epistemic share.
"""

from __future__ import annotations

import argparse
import json
import time

from common.data import cache_path, load_jsonl
from common.evaluate import auroc
from common.io import RESULTS_ROOT
from common.nli_matrix import pick_device
from methods import decomposed_semantic_entropy as plugin
import numpy as np
from scripts.generate import MODELS

from probly.representer.sampler.huggingface import load_model

PARTS = ("uncertainty", "aleatoric", "epistemic")
LABELS = {"uncertainty": "total", "aleatoric": "aleatoric (ambiguous question)", "epistemic": "epistemic (model)"}


def examples(record: dict, rows: list[dict], key: str, k: int = 3) -> list[str]:
    """Markdown for the ``k`` questions with the highest ``key``."""
    order = np.argsort(record[key])[::-1][:k]
    lines = []
    for i in order:
        detail, row = record["details"][i], rows[i]
        lines += [
            (
                f"**{row['question']}** (greedy: {row['greedy_answer']!r}, "
                f"{'correct' if row['correct'] else 'wrong'}; total {record['uncertainty'][i]:.2f}, "
                f"aleatoric {record['aleatoric'][i]:.2f}, epistemic {record['epistemic'][i]:.2f})"
            ),
            "",
        ]
        n = len(detail["answers"]) // len(detail["clarifications"])
        for c, text in enumerate(detail["clarifications"]):
            answers = detail["answers"][c * n : (c + 1) * n]
            clusters = detail["clusters"][c * n : (c + 1) * n] if detail["clusters"] else []
            lines.append(f"- *{text}* -> {', '.join(f'{a} [{g}]' for a, g in zip(answers, clusters, strict=False))}")
        lines.append("")
    return lines


def to_markdown(model: str, run: str, record: dict, rows: list[dict], seconds: float) -> str:
    """Summary of the decomposition as Markdown."""
    correct = [bool(r["correct"]) for r in rows]
    lines = [
        f"# Decomposed semantic entropy: `{model}/{run}`, first {len(rows)} questions",
        "",
        "3 clarifications per question (probly's HFQuestionClarifier), 5 answers each, all 15 answers",
        "clustered with one NLI matrix; total = aleatoric + epistemic (Hou et al., 2024).",
        f"{seconds / len(rows):.1f} s per question on this machine (generation + NLI).",
        "",
        "| part | mean (nats) | AUROC (greedy answer wrong) |",
        "|---|---|---|",
    ]
    lines += [
        f"| {LABELS[key]} | {np.mean(record[key]):.3f} | {auroc(np.array(record[key]), correct):.3f} |" for key in PARTS
    ]
    share = np.sum(record["aleatoric"]) / max(np.sum(record["uncertainty"]), 1e-12)
    lines += [
        "",
        f"Aleatoric share of the total: {share:.0%}.",
        "",
        "## Highest aleatoric (the readings of the question lead to different answers)",
        "",
        *examples(record, rows, "aleatoric"),
        "## Highest epistemic (answers disagree even for one reading)",
        "",
        *examples(record, rows, "epistemic"),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the plugin on the subset and write the report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen0.5b", choices=sorted(MODELS))
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    plugin.SEED = a.seed  # the plugin seeds right before sampling (model loading must not shift it)

    rows = load_jsonl(cache_path(a.model, a.n, a.dataset))[: a.questions]
    model, tok = load_model(MODELS[a.model])
    model = model.to(pick_device())
    model.eval()

    start = time.perf_counter()
    record = plugin.score(rows, model, tok)
    seconds = time.perf_counter() - start

    run = f"{a.dataset}_{a.n}"
    out = RESULTS_ROOT / a.model / run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomposed_se.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "decomposed_se.md").write_text(to_markdown(a.model, run, record, rows, seconds))
    print(f"{a.model}: {len(rows)} questions in {seconds:.0f} s -> {out / 'decomposed_se.md'}")


if __name__ == "__main__":
    main()
