# Names glossary (draft for the Day 1 naming session)

During weeks 1-3 we used short role codes (A1, B3, D4, ...) for tasks. They mean nothing to an
outside reader, and some codes meant different things in different weeks. In code and docs we now
use the descriptive name only. This file is the one place that maps old codes to the new names, so
the learning plan, old notes and the oral exam can still be followed.

**Rule for the rename PRs:** most codes appear as a parenthesis after a sentence that already
describes the thing, e.g. "16 questions share the model batches (A3)". There we simply delete
"(A3)". Only where the code *is* the name ("## Shared data (DATA)", "the selector (D1)") we write
the descriptive name.

**Not role codes, do not change:** "ROUGE-L F1" (`methods/lexical_similarity.py`, the F1 score),
codes inside question text in `data/*.jsonl` (dataset content), "method A / method B" in
`common/evaluate.py`.

## Week 3 codes (current)

| Code | Descriptive name | Where it lives now |
|---|---|---|
| A1 | shared NLI matrix (one pairwise matrix per question) | `common/nli_matrix.py` |
| A2 | de-duplicated NLI pairs | `common/nli_matrix.py` |
| A3 | NLI batching across questions | `common/nli_matrix.py` |
| A4 | NLI setup (deberta-large-mnli, answers only) | `common/nli_matrix.py`, README |
| A5 | persistent NLI cache | `data/<model>/*.nli.<tag>.npz`, `scripts/build_nli_cache.py` |
| B1 | graph measures (from LM-Polygraph) | `common/graph_measures.py`, `methods/_graph_common.py` |
| B2 | sample-count optimization (stop sampling early) | `common/sample_count.py` |
| B3 | embedding backend (`--semantic-backend embedding`) | `common/nli_matrix.py` |
| B4 | decomposed semantic entropy (input clarification) | `common/decomposed_se.py` |
| B5 | semantic scores helper (all seven scores in one call) | `common/semantic_scores.py` |
| C1 | self-evaluation representer in probly | `src/probly/representer/self_evaluation/` (branch) |
| C2 | batched empirical p(True) | branch `c2/empirical-batched` |
| C3 | coverage column in the results table | branch `c3/coverage-column` |
| C4 | new verbalized-confidence prompt | branch `c4/verbalized-prompt` |
| C5 | hybrid scores (CoCoA) | `common/hybrid.py`, `methods/cocoa_msp.py`, `methods/cocoa_p_true.py` |
| D1 | method selector | `selector/select.py`, `selector/rank.py` |
| D2 | method combination (stacking) | `selector/combine.py` |
| D3 | calibration (Platt / isotonic) | `selector/calibrate.py` |
| D4 | per-model profile (`profiles/<model>.json`) | `selector/profile.py`, CONTRACT.md |
| E1 | pipeline cleanup | branch `e1/cleanup` |
| E2 | method registry (cost and access level per method) | `common/registry.py` |
| E3 | MMLU and 1000-question runs | `configs/*mmlu*`, `configs/*_1000.yaml` |
| E4 | paired bootstrap and report | `common/evaluate.py`, `common/summary.py`, `scripts/report.py` |
| E5 | larger-model runs (3B, 7B) | `configs/`, `results/` |
| F1 | uncertainty API | branch `f1-api-skeleton` |
| F2 | text-only backend (OpenAI-compatible API) | branch `f2-text-only-backend` |
| F3 | demo generator | branch `f3-demo` |
| F4 | runnable example in the docs | branch `f4-example-docs` |
| F5 | positioning against LM-Polygraph | branch `f5-positioning` |

## Week 2 and README section codes

| Code | Descriptive name | Where it lives now |
|---|---|---|
| PIPE | pipeline (runner, evaluation, results table) | `cli.py`, `scripts/run_benchmark.py`, `common/evaluate.py`, `common/summary.py` |
| DATA | shared data (generation, correctness labels, answer cache) | `common/data.py`, `common/generation.py`, `common/correctness.py`, `data/` |
| D | method selector (all of D1-D4) | `selector/`, `profiles/` |
| BIG | larger models and their cost (3B, 7B) | README section "Models & cost" |
| SELF | self-evaluation methods (p(True), verbalized) | `common/reflexive.py`, `methods/p_true_*`, `methods/verbalized_2s.py` |

## Week 1 codes (same letters, different meaning!)

| Code | Week 1 meaning | Notebook now | Proposed new notebook name |
|---|---|---|---|
| B1 | data and correctness check | `B1_data_correctness.ipynb` | `data_correctness.ipynb` |
| B2 | evaluation and score contract | `B2_evaluation.ipynb` | `evaluation.ipynb` |
| D1 | semantic entropy | `D1_semantic_entropy.ipynb` | `semantic_entropy.ipynb` |
| R1 | zero-shot p(True) | `R1_ptrue_zeroshot.ipynb` | `p_true_zeroshot.ipynb` |
| R2 | brainstorm p(True) | `R2_ptrue_brainstorm.ipynb` | `p_true_brainstorm.ipynb` |
| R3 | text-only self-evaluation (empirical p(True), verbalized) | `R3_text_only.ipynb` | `text_only_self_evaluation.ipynb` |
| - | comparison of p(True) prompt modes | `reflexive_mode_comparison.ipynb` | `p_true_prompt_modes.ipynb` |

## Files named after codes or days

| Now | Proposal | Owner |
|---|---|---|
| `scripts/r3_methods.py` | retire, or `common/self_evaluation.py` if kept | T3 Elif |
| `scripts/c4_verbalized_prompt.py` | `scripts/compare_verbalized_prompts.py` | T3 Elif |
| `results/<model>/triviaqa_200/c4/` | `results/<model>/triviaqa_200/verbalized_prompts/` | T3 Elif |
| `scripts/day3_queue.sh`, `scripts/day3_retry.sh` | delete if superseded by `cli run` | T5 Elçin |
| `archive/week1/` | delete (git keeps the history) | T6 Eylül |
| CONTRACT.md headings "Score contract (B2)", "Pipeline (PIPE, week 2)", "Profile (D4, week 3)" | "Score contract", "Pipeline", "Profile" | T6 Eylül |

## How we use this file

1. This file lives in the repo (`experiments/llm-probly/docs/names.md`). It is the one file allowed to
   contain codes; the "after" role-code count excludes it.
2. Each owner removes the codes in their own files, using the names above (one PR per task).
3. Notebook owners: comment on this PR if your notebook should be deleted instead of renamed.
