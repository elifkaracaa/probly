"""Semantic uncertainty of one question with SmolLM2-135M on a laptop CPU (B5).

This is the runnable snippet of docs/semantic_methods.md; tests/test_semantic_scores.py
checks that the two stay the same. Run from experiments/llm-probly::

    uv run python -m scripts.semantic_demo

The first run downloads SmolLM2-135M-Instruct (~270 MB) and microsoft/deberta-base-mnli (~560 MB).
"""

from __future__ import annotations

# --- snippet start ---
import logging

from common.generation import generate_answers, stop_at_newline_config
from common.semantic_scores import semantic_scores
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(format="%(message)s")
logging.getLogger("common.semantic_scores").setLevel(logging.INFO)  # progress messages of the helper only
torch.manual_seed(0)

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = PreTrainedTokenizerFast.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

question = "What is the capital of France?"
prompt = f"Answer the following question with as few words as possible.\nQuestion: {question}"
samples = generate_answers(
    model,
    tok,
    prompt,
    num_return_sequences=10,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=32,
    generation_config=stop_at_newline_config(model, tok),
)
samples = [s for s in samples if s["text"]]  # tiny models sometimes return an empty answer

scores = semantic_scores(
    [s["text"] for s in samples],
    sample_log_likelihoods=[s["ll_mean"] for s in samples],
)
print("samples: ", [s["text"] for s in samples])
print("clusters:", scores["clusters"])
for name in ("semantic_entropy", "discrete_semantic_entropy", "num_semantic_sets", "eig_val_laplacian"):
    print(f"{name:>26}: {scores[name]:.3f}")
# --- snippet end ---
