# Semantic methods: NLI matrix, cache and results

Agreement methods that compare the meaning of 10 sampled answers. All of them read one shared
pairwise NLI matrix per question, so asking for all of them costs one NLI pass. NLI model:
`microsoft/deberta-large-mnli`; matrices are cached in `data/<model>/<dataset>_<n>.nli.<tag>.npz`.
User guide with a runnable example: [`semantic_methods.md`](semantic_methods.md).

## Shared NLI matrix, NLI setup and cache

Every ordered pair of [greedy answer, 10 samples] is scored once per question, identical answers
only once, and 16 questions share the model batches on the GPU.

**NLI setup.** Default `deberta-large`, answers only: it agrees best with 80 hand-labelled answer
pairs (0.971 vs 0.937 for deberta-base; a second labeller agreed on 20/20). Change it per run in the config:

    nli_model: deberta-base      # or deberta-large (default), or any Hugging Face NLI model
    nli_use_question: true       # put the question in front of each answer (default false)

**NLI cache.** The matrices are saved next to the data, `data/<model>/<dataset>_<n>.nli.<setup>.npz`,
and reused automatically: a second run does no NLI work. Changed answers or another NLI setup are
detected (fingerprints) and only those questions are computed again.

    uv run python -m cli run --config configs/qwen1.5b_triviaqa_200.yaml \
        --methods semantic_entropy,discrete_semantic_entropy --force
    # NLI for the semantic methods: deberta-large, cache data/qwen1.5b/triviaqa_200.nli.deberta-large.npz (200/200 questions cached)

In your own code:

    from common.data import cache_path, load_data
    from common.nli_matrix import load_nli_cache, nli_cache_path, nli_settings

    rows = load_data("qwen1.5b", 200)
    cache = load_nli_cache(nli_cache_path(cache_path("qwen1.5b", 200), nli_settings()))
    cache["matrices"][0].shape   # (11, 11, 3): contradiction / neutral / entailment per ordered pair

**Effect over week 3** (TriviaQA-200): NLI pairs per question 204 → 59 (0.5B), 174 → 28 (1.5B),
164 → 14 (3B), 159 → 6 (7B), and 0 on a cached rerun; AUROC unchanged (at most 0.006).
Details, the hand labels and the decisions (e.g. no answer normalization before dedup):
[methods.md](methods.md).

Two more agreement scores read the same matrix (features for the selector's stacked combination): `largest_cluster_share` and
`greedy_cluster_share` (1 − share of the samples in the largest cluster / in the greedy answer's cluster).
They carry about the same information as discrete SE and are never chosen by the selector.

## Results and graph measures

AUROC (higher is better, 0.5 = random). NumSemSets, EigVal, DegMat and Eccentricity are the
LM-Polygraph graph measures; "embedding" is the cheap backend (`--semantic-backend embedding`).

| model | data | backend | SE | discrete SE | NumSemSets | EigVal | DegMat | Eccentricity |
|---|---|---|---|---|---|---|---|---|
| 0.5B | TriviaQA-200 | NLI | 0.744 | 0.751 | 0.759 | 0.758 | 0.751 | 0.750 |
| 0.5B | TriviaQA-200 | embedding | 0.745 | 0.755 | 0.763 | 0.774 | 0.762 | 0.751 |
| 0.5B | TriviaQA-1000 | NLI | 0.730 | 0.732 | 0.735 | **0.752** | 0.743 | 0.710 |
| 0.5B | TriviaQA-1000 | embedding | 0.724 | 0.728 | 0.727 | 0.724 | 0.720 | 0.710 |
| 0.5B | MMLU-200 | NLI | 0.504 | 0.517 | 0.533 | 0.522 | 0.503 | 0.518 |
| 1.5B | TriviaQA-200 | NLI | 0.776 | 0.779 | 0.775 | 0.784 | 0.779 | 0.748 |
| 1.5B | TriviaQA-200 | embedding | 0.744 | 0.749 | 0.748 | 0.752 | 0.751 | 0.733 |
| 1.5B | TriviaQA-1000 | NLI | 0.772 | 0.775 | 0.772 | **0.777** | 0.775 | 0.753 |
| 1.5B | TriviaQA-1000 | embedding | 0.759 | 0.764 | 0.763 | 0.764 | 0.759 | 0.743 |
| 1.5B | MMLU-200 | NLI | 0.520 | 0.520 | 0.526 | 0.524 | 0.524 | 0.550 |

Embedding rows for MMLU and all result files: `results/<model>/<dataset>_<n>[_embedding]/`.

**Takeaways**
1. The graph measures are as good as semantic entropy and need no likelihoods; EigVal is the best
   semantic score on TriviaQA-1000 for both models. With 200 questions the 95% interval is about ±0.05,
   so the differences between the semantic scores are not significant.
2. The embedding backend is 26–33x cheaper than deberta-large. It matches NLI on the model its
   threshold was chosen on (0.5B, TriviaQA-200) and is 0.006–0.03 lower elsewhere. NLI stays the default.
3. Five samples instead of ten cost 0.006 (0.5B) to 0.022 (1.5B) AUROC for SE; stopping when the first 2–3
   samples agree saves 18–48% of the generations but costs 0.03–0.10
   (`results/*/triviaqa_200/analysis/sample_count.md`).
4. Splitting the uncertainty by input clarification (decomposed semantic entropy, opt-in) shows *why* an answer is uncertain but does
   not detect wrong answers better: on the same first 50 questions of 0.5B, the total reaches 0.71 AUROC and
   plain SE 0.86 (`results/qwen0.5b/triviaqa_200/analysis/decomposed_se.md`).
5. On MMLU every method is near chance, the baselines included: the models are barely above guessing
   (accuracy 20% and 36% with four options), and with single-letter answers there is little meaning to compare.

One call for all seven scores of a list of answers:

```python
from common.semantic_scores import semantic_scores

scores = semantic_scores(["Paris", "It is Paris.", "Lyon"], sample_log_likelihoods=[-0.1, -0.3, -1.2])
```
