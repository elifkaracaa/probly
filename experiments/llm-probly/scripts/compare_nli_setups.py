"""Compare NLI setups for the semantic methods: NLI model x question in front of the answers.

For each setup ({deberta-base, deberta-large} x {answers only, question + answer}) and each model
(qwen0.5b, qwen1.5b, TriviaQA-200) the script

1. computes the NLI matrices of all questions on this machine (so the times are comparable) and
   saves them to ``results/nli_setups/<model>_<setup>.nli.npz``,
2. reads the decisions for the hand-labelled pairs (``data/nli_hand_labels.json``) from these
   matrices, with probly's rule and with the strict rule (entailment both ways), and reports the
   agreement with the labels, weighted to the real mix of pairs (``weighted_agreement``),
3. computes SE / DSE AUROC with a bootstrap interval, and the paired bootstrap difference of SE
   against the reference setup (deberta-base, answers only).

The table goes to ``results/nli_setups.md``. Run from ``experiments/llm-probly``::

    uv run python -m scripts.compare_nli_setups                         # all 200 questions (keep the machine awake)
    uv run python -m scripts.compare_nli_setups --limit 20              # quick check
    uv run python -m scripts.compare_nli_setups --reuse                 # new labels, no NLI run

``--reuse`` reads the matrices saved by an earlier full run (with their measured time per question)
instead of running the NLI models again, e.g. after adding hand labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING

from common.data import cache_path, load_jsonl
from common.evaluate import auroc, bootstrap_auroc, paired_bootstrap
from common.nli_labels import candidate_pairs, load_labels, stratum_shares, weighted_agreement
from common.nli_matrix import (
    NLI_MODELS,
    QUESTIONS_PER_BATCH,
    RULES,
    compute_matrices,
    load_nli_cache,
    pick_device,
    same_meaning,
    save_nli_cache,
    se_and_dse,
)
import numpy as np

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "nli_setups"
MODELS = ("qwen0.5b", "qwen1.5b")
REFERENCE = "deberta-base"


@dataclass
class SetupResult:
    """Everything measured for one NLI setup, per model (``agreement`` per rule)."""

    seconds: dict[str, float] = field(default_factory=dict)
    se: dict[str, np.ndarray] = field(default_factory=dict)
    dse: dict[str, np.ndarray] = field(default_factory=dict)
    matrices: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    agreement: dict[str, float] = field(default_factory=dict)


def setup_name(nli: str, *, use_question: bool) -> str:
    """Name of a setup, e.g. ``deberta-large+question``."""
    return f"{nli}+question" if use_question else nli


def shares_per_model() -> dict[str, dict[str, float]]:
    """Stratum shares of all candidate pairs per model (from the committed base / large caches)."""
    shares = {}
    for model in MODELS:
        rows = load_jsonl(cache_path(model, 200))
        base = load_nli_cache(cache_path(model, 200, suffix=".nli.npz"))["matrices"]
        large = load_nli_cache(cache_path(model, 200, suffix=".nli.deberta-large.npz"))["matrices"]
        shares[model] = stratum_shares(candidate_pairs(model, rows, base, large))
    return shares


def run_setup(
    clusterer: HFGreedySemanticClusterer, model: str, rows: list[dict], *, use_question: bool
) -> tuple[list[torch.Tensor], float]:
    """NLI matrices of all rows in batches; returns ``(matrices, seconds_per_question)``."""
    matrices = []
    t0 = time.time()
    for start in range(0, len(rows), QUESTIONS_PER_BATCH):
        chunk = rows[start : start + QUESTIONS_PER_BATCH]
        matrices += compute_matrices(clusterer, chunk, use_question=use_question)
    seconds = (time.time() - t0) / len(rows)
    print(f"  {model}: {seconds:.2f} s/question")
    return matrices, seconds


def load_saved(path: Path, n: int) -> tuple[list[torch.Tensor], float]:
    """Matrices and seconds per question of an earlier full run (``--reuse``)."""
    cache = load_nli_cache(path)
    if len(cache["matrices"]) < n:
        msg = f"{path.name} has {len(cache['matrices'])} questions, need {n}; run without --reuse"
        raise ValueError(msg)
    return cache["matrices"][:n], cache["seconds_per_question"]


def main(limit: int, *, reuse: bool = False) -> None:
    """Run (or, with ``reuse``, reload) every setup and write the comparison table."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shares = shares_per_model()
    data = {model: load_jsonl(cache_path(model, 200))[: limit or None] for model in MODELS}
    labels = [label for label in load_labels() if label["id"] < len(data[label["model"]])]
    results: dict[str, SetupResult] = {}
    device = pick_device()

    for nli, hf_name in NLI_MODELS.items():
        clusterer = None
        if not reuse:
            clusterer = HFGreedySemanticClusterer.from_model_name(hf_name, batch_size=32, max_length=256)
            clusterer.model.to(device)
        for use_question in (False, True):
            name = setup_name(nli, use_question=use_question)
            print(f"{name} " + ("(saved matrices)" if reuse else f"on {device}"))
            result = results[name] = SetupResult()
            for model, rows in data.items():
                path = OUT_DIR / f"{model}_{name}.nli.npz"
                if clusterer is None:
                    matrices, seconds = load_saved(path, len(rows))
                else:
                    matrices, seconds = run_setup(clusterer, model, rows, use_question=use_question)
                    save_nli_cache(
                        path,
                        matrices,
                        [row["id"] for row in rows],
                        nli_model=hf_name,
                        use_question=use_question,
                        seconds_per_question=seconds,
                    )
                scores = np.array(
                    [se_and_dse(m, row["samples_ll_mean"]) for m, row in zip(matrices, rows, strict=True)]
                )
                result.seconds[model] = seconds
                result.se[model], result.dse[model] = scores[:, 0], scores[:, 1]
                result.matrices[model] = matrices
            for rule in RULES:
                decisions = [
                    same_meaning(result.matrices[lab["model"]][lab["id"]], lab["index_a"], lab["index_b"], rule)
                    for lab in labels
                ]
                result.agreement[rule] = weighted_agreement(labels, decisions, shares)

    write_table(results, data)


def write_table(results: dict[str, SetupResult], data: dict[str, list[dict]]) -> None:
    """Print the comparison and save it as ``results/nli_setups.md``."""
    correct = {model: np.array([row["correct"] for row in rows]) for model, rows in data.items()}
    lines = [
        "| setup | agreement with hand labels (probly rule) | (strict rule) | "
        + " | ".join(f"SE AUROC {m} | DSE AUROC {m} | SE vs reference {m} | s/question {m}" for m in MODELS)
        + " |",
        "|" + "---|" * (3 + 4 * len(MODELS)),
    ]
    for name, result in results.items():
        cells = [name, f"{result.agreement['probly']:.3f}", f"{result.agreement['strict']:.3f}"]
        for model in MODELS:
            se, dse = result.se[model], result.dse[model]
            low, high = bootstrap_auroc(se, correct[model])
            cells.append(f"{auroc(se, correct[model]):.3f} [{low:.2f}-{high:.2f}]")
            cells.append(f"{auroc(dse, correct[model]):.3f}")
            if name == REFERENCE:
                cells.append("reference")
            else:
                diff = paired_bootstrap(se, results[REFERENCE].se[model], correct[model])
                cells.append(f"{diff.diff:+.3f} [{diff.ci_low:+.3f}, {diff.ci_high:+.3f}] {diff.verdict}")
            cells.append(f"{result.seconds[model]:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    n = {model: len(rows) for model, rows in data.items()}
    header = (
        f"# NLI setups\n\nTriviaQA questions per model: {n}. Agreement is weighted to the real mix of "
        'answer pairs (see `common/nli_labels.weighted_agreement`). "SE vs reference" is the paired bootstrap '
        "difference of SE AUROC against deberta-base without the question (positive = better; verdict A means "
        "this setup is better, B the reference, tie = interval contains 0).\n\n"
    )
    table = header + "\n".join(lines) + "\n"
    (ROOT / "results" / "nli_setups.md").write_text(table, encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions per model (quick check)")
    parser.add_argument("--reuse", action="store_true", help="read the saved matrices instead of running NLI")
    args = parser.parse_args()
    main(args.limit, reuse=args.reuse)
