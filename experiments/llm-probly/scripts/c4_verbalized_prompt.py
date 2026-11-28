"""Does an anchored prompt fix verbalized confidence? Runs it on the cached greedy answers.

The week 1 prompt ("a number between 0 and 100") makes the small models answer almost always "95"
or "100". This script asks the same question with two 0-10 prompts and compares them with the
saved ``verbalized_2s`` result of the same model: ``plain10`` (``ASK[10]``, only the scale changes)
and ``anchored`` (``common.verbalized.ASK_ANCHORED``, scale 0-10 with both ends explained), so the
effect of the scale and of the anchoring can be told apart. For each: AUROC with a bootstrap
interval, how many distinct replies the model gives, and the most common ones.

The new scores go to ``results/<run>/triviaqa_200/c4/verbalized_2s_<prompt>.json``. That folder is not
read by the summary table, so nothing in the main results changes.

Example:
    uv run python -m scripts.c4_verbalized_prompt --run qwen1.5b
    uv run python -m scripts.c4_verbalized_prompt --run qwen3b --device mps --dtype float16 --prompts plain100 anchored

The second line is for a 16 GB machine; add ``plain10`` too to run all three prompts in the same setup.
``--from-saved`` loads no model: it prints the table of docs/methods.md (AUROC, replies and the paired
bootstrap against the week 1 prompt) from the files already in ``c4/``::

    uv run python -m scripts.c4_verbalized_prompt --run qwen3b --from-saved
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from typing import TYPE_CHECKING

from common.data import load_data
from common.evaluate import bootstrap_auroc, evaluate, paired_bootstrap
from common.io import RESULTS_ROOT, load_scores
from common.verbalized import ASK, ASK_ANCHORED, verbalized_2s
import torch

from probly.representer.sampler.huggingface import load_model

if TYPE_CHECKING:
    from pathlib import Path

MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
SCALE = 10
PROMPTS = {"plain100": ASK[100], "plain10": ASK[SCALE], "anchored": ASK_ANCHORED}
"""``plain100`` is the week 1 prompt; rerun it when the model runs with another precision or device than
the saved ``verbalized_2s`` result, so that all prompts are compared under the same setup."""
SCALES = {"plain100": 100, "plain10": SCALE, "anchored": SCALE}


def summarize(label: str, replies: list[str], uncertainty: list[float], correct: list[bool]) -> None:
    """Print AUROC (with interval) and the reply distribution of one prompt."""
    scores = evaluate(uncertainty, correct)
    low, high = bootstrap_auroc(uncertainty, correct)
    counts = Counter(reply.strip() for reply in replies)
    print(f"{label}: AUROC {scores['auroc']:.3f} [{low:.3f}, {high:.3f}], {len(counts)} distinct replies")
    print(f"  most common: {counts.most_common(5)}")


def paired_report(folder: Path, ids: list, correct: list[bool]) -> list[str]:
    """Paired bootstrap of the saved 0-10 prompts against the week 1 prompt (same questions).

    The week 1 reference is ``c4/verbalized_2s_plain100.json`` when it exists (rerun in the same setup
    as the new prompts), otherwise the saved ``verbalized_2s.json``.

    Args:
        folder: Run folder, ``results/<run>/triviaqa_<n>``.
        ids: Question ids of the data cache, in order.
        correct: Correctness per question, same order.

    Returns:
        One printable line per comparison.
    """
    plain100 = folder / "c4" / "verbalized_2s_plain100.json"
    reference = plain100 if plain100.exists() else folder / "verbalized_2s.json"
    saved = {
        name: load_scores(folder / "c4" / f"verbalized_2s_{name}.json", expected_ids=ids)["uncertainty"]
        for name in ("plain10", "anchored")
        if (folder / "c4" / f"verbalized_2s_{name}.json").exists()
    }
    week1 = load_scores(reference, expected_ids=ids)["uncertainty"]
    pairs = [(name, "week 1", values, week1) for name, values in saved.items()]
    if len(saved) == len(("plain10", "anchored")):
        pairs.append(("anchored", "plain10", saved["anchored"], saved["plain10"]))
    lines = [f"paired bootstrap (AUROC difference, 95 % interval), week 1 = {reference.name}:"]
    for name_a, name_b, values_a, values_b in pairs:
        d = paired_bootstrap(values_a, values_b, correct)
        lines.append(f"  {name_a} - {name_b}: {d.diff:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]")
    return lines


def main() -> None:
    """Run the 0-10 prompts on one model and compare them with the saved week 1 result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="qwen1.5b", choices=sorted(MODELS))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--prompts", nargs="+", default=["plain10", "anchored"], choices=list(PROMPTS))
    parser.add_argument("--from-saved", action="store_true", help="no model: report the saved c4/ files")
    args = parser.parse_args()

    rows = load_data(args.run, args.n)
    ids = [row["id"] for row in rows]
    correct = [row["correct"] for row in rows]
    folder = RESULTS_ROOT / args.run / f"triviaqa_{args.n}"

    old = load_scores(folder / "verbalized_2s.json", expected_ids=ids)  # same questions, same order
    summarize("before (0-100)", old["replies"], old["uncertainty"], correct)

    if args.from_saved:
        for name in PROMPTS:
            path = folder / "c4" / f"verbalized_2s_{name}.json"
            if path.exists():
                saved = load_scores(path, expected_ids=ids)
                setup = f"{saved.get('device', '?')} {saved.get('dtype', '?')}"
                summarize(f"saved ({name}, {setup})", saved["replies"], saved["uncertainty"], correct)
        print("\n".join(paired_report(folder, ids, correct)))
        return

    model, tok = load_model(MODELS[args.run], model_kwargs={"dtype": getattr(torch, args.dtype)})
    model = model.to(args.device)
    model.eval()
    print(f"model on {args.device}, {args.dtype}")
    for name in args.prompts:
        ask = PROMPTS[name]
        uncertainty, failed, replies = [], [], []
        start = time.perf_counter()
        for row in rows:
            confidence, no_number, reply = verbalized_2s(
                model, tok, row["question"], row["greedy_answer"], scale=SCALES[name], ask=ask
            )
            uncertainty.append(1.0 if no_number else 1.0 - confidence)
            failed.append(no_number)
            replies.append(reply)
        seconds = (time.perf_counter() - start) / len(rows)
        summarize(f"after ({name}, {args.device} {args.dtype})", replies, uncertainty, correct)
        print(f"  unreadable replies: {sum(failed)}, {seconds:.2f} s per question")

        out = folder / "c4" / f"verbalized_2s_{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "method": f"verbalized_2s_{name}",
            "model": MODELS[args.run],
            "answer": "greedy",
            "probability": True,
            "scale": SCALES[name],
            "prompt": ask,
            "device": args.device,
            "dtype": args.dtype,
            "replies": replies,
            "ids": ids,
            "uncertainty": uncertainty,
            "failed": failed,
            "seconds_per_question": seconds,
            "model_calls_per_question": 1,
        }
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"  saved {out}")
    print("\n".join(paired_report(folder, ids, correct)))


if __name__ == "__main__":
    main()
