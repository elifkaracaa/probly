"""=====================================================
From an LLM answer to a calibrated confidence
=====================================================

Which uncertainty method should you trust for *your* language model, and what
does its score mean? This example walks through the full llm-probly pipeline
on a laptop CPU:

1. **Generate** an answer with a small instruct model and keep the
   log probability and entropy of every generated token.
2. **Methods**: turn that one generation into several uncertainty scores.
3. **Selector**: read the model's saved profile, which says which method won
   on a labelled calibration split of TriviaQA, and why.
4. **Calibrated confidence**: map the chosen method's raw score to a number in
   [0, 1] that estimates how likely the answer is correct.

Steps 3 and 4 do not refit anything. The selector and the calibration were run
once per model on the benchmark and saved as a profile
(``experiments/llm-probly/profiles/<model>.json``); this example and the
uncertainty API only read that file.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from probly.representer.sampler.huggingface import load_model

logging.getLogger("transformers").setLevel(logging.ERROR)
torch.manual_seed(0)

# %%
# Model and profile
# -----------------
#
# The confidence is only meaningful for the model the profile was fitted on, so
# the example uses the smallest benchmark model, Qwen2.5-0.5B-Instruct, and its
# profile. The prompt and generation settings are the ones the benchmark used.

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def find_repo_root() -> Path:
    """Find the probly checkout from the working directory.

    The docs build runs examples from their own folder and does not define
    ``__file__``, so search the working directory and its parents instead.
    """
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "experiments" / "llm-probly").is_dir():
            return folder
    msg = "Run this example from inside the probly repository."
    raise FileNotFoundError(msg)


PROFILE_PATH = find_repo_root() / "experiments" / "llm-probly" / "profiles" / "qwen0.5b.json"
INSTRUCTION = "Answer the following question with as few words as possible."
MAX_NEW_TOKENS = 32

with PROFILE_PATH.open() as file:
    profile = json.load(file)
if profile["schema_version"] != 1 or profile["model_id"] != MODEL_ID:
    msg = f"{PROFILE_PATH} is not a schema-1 profile for {MODEL_ID}"
    raise ValueError(msg)

model, tokenizer = load_model(MODEL_ID, model_kwargs={"dtype": torch.float32})
model.eval()

# %%
# Step 1: generate
# ----------------
#
# One greedy answer per question, generated exactly like the benchmark cache:
# the question goes through the chat template, generation stops at the first
# line break, and only the tokens before the stop are scored. For every scored
# token we keep its log probability and the entropy of the next-token
# distribution it was drawn from.

QUESTIONS = {  # question -> reference answer (None: there is no correct answer)
    "What is the capital of France?": "Paris",
    "What is the chemical symbol for gold?": "Au",
    "Who wrote the novel Pride and Prejudice?": "Jane Austen",
    "In which year did the Berlin Wall fall?": "1989",
    "Which planet has a moon called Titan?": "Saturn",
    "What is the capital of Burkina Faso?": "Ouagadougou",
    "Who painted The Garden of Earthly Delights?": "Hieronymus Bosch",
    "Which element has the atomic number 74?": "Tungsten",
    "Who was the first person to walk on Mars?": None,
}

generation_config = model.generation_config
eos = generation_config.eos_token_id
stop_ids = set(eos) if isinstance(eos, list) else {eos}
stop_ids |= {i for i, piece in enumerate(tokenizer.batch_decode([[i] for i in range(len(tokenizer))])) if "\n" in piece}
if tokenizer.pad_token_id is not None:
    stop_ids.add(tokenizer.pad_token_id)


def generate(question: str) -> dict:
    """Generate the greedy answer and its per-token statistics, as in the benchmark."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\nQuestion: {question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=sorted(stop_ids),
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        token_logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        step_logp = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.scores], dim=1)[0]
        token_entropies = -(step_logp.exp() * step_logp.nan_to_num(neginf=0.0)).sum(dim=-1)
    generated = out.sequences[0, inputs["input_ids"].shape[-1] :]
    is_stop = torch.isin(generated, torch.tensor(sorted(stop_ids)))
    n_scored = int((torch.cumsum(is_stop.long(), dim=0) == 0).sum())
    return {
        "answer": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "token_logprobs": token_logprobs[:n_scored].tolist(),
        "token_entropies": token_entropies[:n_scored].tolist(),
    }


generations = {question: generate(question) for question in QUESTIONS}

# %%
# Step 2: methods
# ---------------
#
# Four methods need nothing beyond this one generation. Each returns a raw
# uncertainty (higher = more uncertain) with the same definition as its plugin
# in ``experiments/llm-probly/methods``. The sample-based methods (lexical
# similarity, semantic entropy, ...) and the self-evaluation methods need extra
# model calls; the selector compared them too, but they are left out here to
# keep the example fast.


# An empty answer has no scored tokens and nothing to trust, so every method
# returns infinite uncertainty for it (calibrated confidence 0, as in the API).


def msp(gen: dict) -> float:
    """Maximum sequence probability: minus the log probability of the whole answer."""
    lps = gen["token_logprobs"]
    return -sum(lps) if lps else math.inf


def likelihood_mean(gen: dict) -> float:
    """Minus the mean token log probability (length-normalized likelihood)."""
    lps = gen["token_logprobs"]
    return -sum(lps) / len(lps) if lps else math.inf


def perplexity(gen: dict) -> float:
    """Exponential of the mean negative token log probability."""
    return math.exp(likelihood_mean(gen))


def mean_token_entropy(gen: dict) -> float:
    """Mean entropy of the next-token distributions, in nats."""
    ents = gen["token_entropies"]
    return sum(ents) / len(ents) if ents else math.inf


METHODS = {
    "msp": msp,
    "likelihood_mean": likelihood_mean,
    "perplexity": perplexity,
    "mean_token_entropy": mean_token_entropy,
}


def normalize(text: str) -> str:
    """Lower-case and drop punctuation, so 'Paris.' matches 'paris'."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def is_correct(answer: str, reference: str | None) -> bool:
    """Count an answer as correct if it contains the reference answer as whole words."""
    if reference is None:
        return False
    return re.search(rf"\b{re.escape(normalize(reference))}\b", normalize(answer)) is not None


scores = {question: {name: method(gen) for name, method in METHODS.items()} for question, gen in generations.items()}

for number, (question, reference) in enumerate(QUESTIONS.items(), start=1):
    answer = generations[question]["answer"]
    mark = "correct" if is_correct(answer, reference) else "wrong"
    print(f"Q{number}: {question}\n    answer: {answer}  [{mark}, reference: {reference}]")

print(f"\n{'':<4}" + "".join(f"{name:>19}" for name in METHODS))
for number, question in enumerate(QUESTIONS, start=1):
    print(f"Q{number:<3}" + "".join(f"{scores[question][name]:>19.3f}" for name in METHODS))

# %%
# Step 3: selector
# ----------------
#
# The four scores above disagree in scale and, more importantly, in how well
# they separate right from wrong answers. The selector settled this once for
# this model: on a labelled calibration split it ranked every method by AUROC
# with a bootstrap interval, then picked the cheapest method whose AUROC is not
# significantly worse than the best one. The profile stores the choice, the
# reason and every candidate's result.

method = profile["method"]
print(f"Chosen method: {method['name']} ({method['family']}, needs {method['access']}, cost {method['cost']:g})")
print(f"Reason: {profile['reason']}")

candidates = sorted(profile["candidates"].items(), key=lambda item: item[1]["auroc"])
names = [name for name, _ in candidates]
auroc = np.array([m["auroc"] for _, m in candidates])
low = np.array([m["auroc_low"] for _, m in candidates])
high = np.array([m["auroc_high"] for _, m in candidates])
colors = ["C1" if name == method["name"] else "C0" for name in names]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(names, auroc, color=colors)
ax.errorbar(auroc, names, xerr=[auroc - low, high - auroc], fmt="none", ecolor="black", capsize=3, linewidth=1)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xlim(0.4, 1.0)
ax.set_xlabel("AUROC on the calibration split (95% bootstrap interval)")
ax.set_title(f"Selector for {profile['model']}: chosen method highlighted")
fig.tight_layout()
plt.show()

# %%
# Many intervals overlap: on a calibration split of this size the top methods
# are often not distinguishable, and then the selector prefers the cheapest.
# When it picks a token-level method, one greedy pass per question is enough;
# there is no need to sample ten answers.
#
# Step 4: calibrated confidence
# -----------------------------
#
# The chosen method's raw score has no unit; a likelihood of 0.4 says nothing on
# its own. The profile also stores a calibration map fitted on the same split
# (Platt scaling or isotonic regression, picked by cross-validated Brier score),
# which turns the raw uncertainty into the probability that the answer is
# correct. Its expected calibration error on held-out questions is stored too.


def confidence(uncertainty: np.ndarray | float) -> np.ndarray:
    """Map the chosen method's raw uncertainty to a confidence in [0, 1] (as ``selector.profile.confidence``)."""
    u = np.asarray(uncertainty, dtype=float)
    calibration = profile["calibration"]
    params = calibration["parameters"]
    if calibration["type"] == "platt":
        return 1.0 / (1.0 + np.exp(params["a"] * u + params["b"]))
    if calibration["type"] == "isotonic":
        return np.interp(u, params["x"], params["y"])
    msg = f"profile calibration {calibration['type']!r} cannot be applied"
    raise ValueError(msg)


if method["name"] not in METHODS:
    msg = f"The profile chose {method['name']!r}; this example only computes {sorted(METHODS)}"
    raise ValueError(msg)
chosen = METHODS[method["name"]]
raw = np.array([chosen(generations[question]) for question in QUESTIONS])
conf = confidence(raw)
correct = np.array([is_correct(generations[q]["answer"], ref) for q, ref in QUESTIONS.items()])

ece_after = profile["calibration"]["ece_after"]
ece_text = "not measured yet" if ece_after is None else f"{ece_after:.3f}"
print(f"Calibration: {profile['calibration']['type']}, held-out ECE after calibration: {ece_text}")
print(f"{'':<4} {'answer':<8} {'raw':>7} {'confidence':>11}")
for number, (answer_ok, u, c) in enumerate(zip(correct, raw, conf, strict=True), start=1):
    print(f"Q{number:<3} {'correct' if answer_ok else 'wrong':<8} {u:>7.3f} {c:>11.2f}")

finite = np.isfinite(raw)
grid = np.linspace(0.0, max(raw[finite].max(initial=0.0) * 1.2, 1.0), 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, confidence(grid), color="gray", label="calibration map from the profile")
ok, wrong = finite & correct, finite & ~correct
ax.scatter(raw[ok], conf[ok], color="C2", zorder=3, label="correct answer")
ax.scatter(raw[wrong], conf[wrong], color="C3", marker="x", zorder=3, label="wrong or no correct answer")
ax.set_xlabel(f"raw uncertainty ({method['name']})")
ax.set_ylabel("calibrated confidence")
ax.set_ylim(0.0, 1.0)
ax.set_title(f"{profile['model']}: raw score to calibrated confidence")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The ceiling of the curve is the confidence of a perfectly certain answer,
# ``confidence(0)``. For a small model it can be well below 1: even its most
# confident TriviaQA answers are often wrong, and a calibrated score says so
# instead of reporting a near-certain probability.
#
# Nine questions are only an illustration; the calibration quality is measured
# by the ECE on the held-out split, not by this table. The uncertainty API
# (``POST /uncertainty``) runs the same four steps for one question and returns
# the answer, the calibrated confidence, the chosen method and its cost.

print(f"Highest possible confidence for {profile['model']}: {float(confidence(0.0)):.2f}")
