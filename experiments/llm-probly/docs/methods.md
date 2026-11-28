# Methods

All methods are plugins in `methods/` (see CONTRACT.md). Each one has a family, and each one reads the same cached generations.

Family, model access (`text_only` / `logprobs` / `white_box`) and cost per question (generation passes,
greedy answer included) are kept in one place, `common/registry.py`; `uv run python -m cli methods` prints them.
The selector reads the same registry.

| Method | Family | Needs logits? | Model calls per question |
|---|---|---|---|
| semantic_entropy | agreement | yes (sample log-likelihoods, `samples_ll_mean`) | 0 LLM calls (reads the 10 cached samples) + one shared NLI matrix per question |
| discrete_semantic_entropy | agreement | no (cluster sizes only) | 0 LLM calls + the same shared NLI matrix |
| largest_cluster_share | agreement | no | 1 - share of the 10 samples in their largest meaning cluster; 0 LLM calls + the same shared NLI matrix (feature for the selector, D2) |
| greedy_cluster_share | agreement | no | 1 - share of the 10 samples with the same meaning as the greedy answer; 0 LLM calls + the same shared NLI matrix (feature for the selector, D2) |
| lexical_similarity | agreement | no (text only, Rouge-L between samples) | 0 |
| p_true_zeroshot | self-eval | yes (probability of "True") | 1 extra LLM call |
| p_true_brainstorm | self-eval | yes (probability of the answer option "A" vs "B") | 1 extra LLM call (the 10 cached samples are shown in the prompt) |
| p_true_empirical | self-eval | yes (probability of "True") | 10 extra generations + scoring |
| verbalized_2s | self-eval | no (the model writes a confidence number, read from the text) | 1 short extra LLM call |
| msp | baseline | yes (cached greedy log-probs) | 0 |
| likelihood_mean | baseline | yes (cached greedy log-likelihood) | 0 |
| mean_token_entropy | baseline | yes (cached token entropies) | 0 |

## How the cost works

Generation happens once per model: 1 greedy answer and 10 samples per question are cached. Every method reads that cache, so the methods with 0 LLM calls only do arithmetic on saved data. The self-evaluation methods need the LLM again at scoring time, because they ask the model to judge its own answer.

The agreement methods (SE and discrete SE) also run an NLI model on the answers to group them by meaning.
This is a small classifier, not the LLM. Since week 3 (A1-A3) the NLI work is done once per question and
shared:

- **A1, one matrix:** the NLI model scores every ordered pair of [greedy answer, 10 samples] once
  (`HFSemanticClusterer.pairwise_nli` in probly); SE, discrete SE and later graph methods all read this matrix.
- **A2, dedup:** identical answers are sent to the NLI model only once (identical = entailment).
- **A3, batching:** the pairs of 16 questions share model batches, sorted by length; the model runs on the GPU
  (MPS / CUDA) when there is one.

Effect on qwen0.5b TriviaQA (50 questions, Mac, MPS): 110 -> 56 NLI pairs and 0.47 -> 0.20 s per question,
with identical SE / DSE scores. Over all 200 questions dedup skips 46 % (0.5B) and 74 % (1.5B) of the pairs.

**Decision: no answer normalization before dedup.** We measured whether treating answers that differ only in
case, punctuation or whitespace ("Paris" / "paris." ) as identical would help (deberta-large matrices,
TriviaQA-200, pairs equal after normalization set to entailment):

| | qwen0.5b | qwen1.5b |
|---|---|---|
| NLI pairs skipped, exact dedup (A2) | 46.1 % | 74.5 % |
| with normalization | 47.1 % | 77.9 % |
| SE / DSE AUROC | unchanged (0.744 / 0.751) | unchanged (0.776 / 0.779) |
| questions whose SE changes | 0 / 200 | 0 / 200 |

The NLI model already calls every such pair the same meaning, so normalization only saves 1-3 % more pairs and
changes no score. It would add a library option, tests and a cache rule, and has risks of its own: removing
punctuation joins numbers ("3.5" / "35"), removing articles joins different things ("The Maine" / "Maine").
So we keep exact dedup (agreed with the reviewer of A2).

## Before / after over week 3 (all four models, TriviaQA-200)

NLI pairs per question, time and AUROC of SE / DSE before week 3 (week-2 results, deberta-base) and after
A1-A5 (deberta-large, Mac, MPS). Week 2 ran SE and DSE separately, each clustering lazily (comparing a new
answer with one answer of each cluster found so far, both directions), and `build_nli_cache` built the full
matrix again for the analyses and graph measures. Pair counts are recomputed from the saved matrices
(the lazy counts of 3B / 7B from the deberta-large decisions, because they had no deberta-base matrix).

| model | NLI pairs / question, week 2: SE + DSE lazy + full matrix | week 3: one deduplicated matrix | reduction | SE + DSE seconds / question, week 2 | NLI seconds / question, week 3 (first run / cached run) | SE AUROC week 2 -> 3 | DSE AUROC week 2 -> 3 |
|---|---|---|---|---|---|---|---|
| qwen0.5b | 94 + 110 = 204 | 59.3 | -71 % | 3.25 | 0.67 / 0 | 0.740 -> 0.744 | 0.746 -> 0.751 |
| qwen1.5b | 64 + 110 = 174 | 28.1 | -84 % | 1.49 | 0.25 / 0 | 0.781 -> 0.776 | 0.783 -> 0.779 |
| qwen3b | 54 + 110 = 164 | 14.0 | -91 % | 0.55 | 0.13 / 0 | 0.759 -> 0.761 | 0.765 -> 0.765 |
| qwen7b | 49 + 110 = 159 | 6.2 | -96 % | 0.40 | 0.06 / 0 | 0.735 -> 0.729 | 0.734 -> 0.729 |

- **NLI pairs at least halved on every model**, and a second run does no NLI work (A5). Bigger models repeat
  their answers more often, so dedup removes more pairs (and the NLI time falls with model size).
- Compared with SE + DSE alone (94 / 64 / 54 / 49 lazy pairs), the shared matrix is cheaper from 1.5B on, but
  not on 0.5B (59 vs 94): its 0.5B answers are mostly different. The matrix pays off because SE, DSE, the graph
  measures and the hybrid scores all read it, and it is computed once and cached.
- The time comparison is not only the NLI changes: week 3 uses deberta-large (about 3x slower than deberta-base)
  on the GPU (A3), week 2 ran deberta-base as it was then.
- **AUROC changes are within noise** on all models (at most 0.006; the 95 % intervals are about ±0.06; the
  paired bootstrap for 0.5B and 1.5B is in the NLI setup table below; 0.5B DSE also gains from the tie fix).

## NLI setup (A4)

Which NLI model, and should the question go in front of each answer? Compared on qwen0.5b and qwen1.5b
TriviaQA-200 with `scripts/compare_nli_setups.py` (full table: `results/nli_setups.md`), against 80 answer
pairs labelled by hand (`data/nli_hand_labels.jsonl`, made with `scripts/label_nli_pairs.py`: 60 stratified by
how the NLI models decide + 20 near duplicates). The
agreement with the labels is weighted to the real mix of answer pairs (the labels over-sample the hard pairs).

| setup | agreement with hand labels | SE AUROC 0.5B / 1.5B | s per question 0.5B / 1.5B |
|---|---|---|---|
| deberta-base | 0.937 | 0.740 / 0.781 | 0.22 / 0.07 |
| deberta-base + question | 0.959 | 0.754 / 0.772 | 0.46 / 0.19 |
| **deberta-large** (default) | **0.971** | 0.744 / 0.776 | 0.71 / 0.23 |
| deberta-large + question | 0.938 | 0.772 / 0.781 | 1.38 / 0.55 |

- No setup changes SE AUROC beyond noise (paired bootstrap against deberta-base: every interval contains 0).
- **Default: deberta-large, answers only.** Its clusters agree best with the human labels (deberta-base merges
  answers people call different), AUROC is the same, and the extra time (about 3x) is paid once per cache.
  It is also LM-Polygraph's default. probly's library default stays deberta-base.
- The question in front of each answer doubles the time and gives no consistent gain.
- probly's rule (no contradiction either way, not neutral both ways) agrees better with the labels than the
  strict rule (entailment both ways) in every setup (0.94-0.97 vs 0.87-0.91): the strict rule splits answers
  like "Annie Hall" and "Annie Hall by John Williams".
- **The labels are reliable:** a second person (Eylül) labelled 20 of the pairs without seeing the first labels
  or the NLI decisions (`data/nli_hand_labels_eylul.jsonl`, `label_nli_pairs --second-labeller 20`): she agreed
  on 20/20 (Cohen's kappa 1.00; 14 of them were not near duplicates). The sample is small, so this shows the
  labelling guideline is clear, not that the labels are perfect.
- Change the setup with `nli_model` / `nli_use_question` in a config (see `docs/pipeline.md`).

## Persistent NLI cache (A5)

The NLI matrices are kept on disk, one file per data cache and NLI setup:
`data/<model>/<dataset>_<n>.nli.<tag>.npz` (e.g. `triviaqa_200.nli.deberta-large.npz`; deberta-base on
answers only keeps the week-2 name `.nli.npz`). The runner reads it before SE / DSE and writes new matrices
back, so **a second run on the same data does no NLI work** (`nli_pairs_per_question` is 0 in the record).

Measured on qwen0.5b TriviaQA-200 (deberta-large, Mac, MPS), two runs of
`run_benchmark --methods semantic_entropy,discrete_semantic_entropy --force`:

| run | questions cached | NLI pairs per question | SE seconds per question |
|---|---|---|---|
| 1st (file from before A5, no fingerprints) | 0/200 | 59.3 | 0.67 |
| 2nd | 200/200 | 0 | 0.00 (0.1 s for all 200, NLI model not loaded) |

Discrete SE shows 0 pairs in both runs because it reads the matrices SE has just made (A1).

`seconds_per_question` in a record is the time of that run, so it is about 0 when the matrices come from
the cache (as CONTRACT.md asks). The cost of the NLI step is kept in `nli_seconds_per_question`: what
building the matrices took (0.67 s per question above), measured in this run or stored in the cache file.
The registry cost of SE / DSE (12) already counts the NLI pass, so the report's picks are not affected.

A cached matrix is only reused when nothing it depends on has changed:

| changes | what happens |
|---|---|
| generator model or dataset (or n) | another data cache, so another file |
| NLI model or question conditioning | another file name; a file whose stored setup does not match is not used |
| the answers of a question (data regenerated) | that question's fingerprint changes, only it is computed again |
| file from before A5 (no fingerprints) | everything is computed again and the file is rewritten |

The fingerprint (`common.nli_matrix.matrix_fingerprint`) is a hash of the NLI model name, the question (only
when it is put in front of the answers) and the exact answer texts. Other code that needs the matrices
(e.g. graph measures, hybrid scores) should find the file with `nli_cache_path(data_file, nli_settings())`
and read it with `load_nli_cache`, instead of writing the file name by hand.

## Verbalized confidence prompt (C4)

The week-1 prompt ("a number between 0 and 100") makes the models give almost always the same
number: 1.5B says "95" for 170 of 200 questions, 3B says "100" for 145. `scripts/c4_verbalized_prompt.py`
asks the same question with two 0-10 prompts (`common/verbalized.py`): `plain10` (`ASK[10]`, only the
scale changes) and `anchored` (`ASK_ANCHORED`, 0 = certainly wrong, 10 = certainly correct).
Results in `results/<model>/triviaqa_200/c4/`; TriviaQA-200, AUROC, paired bootstrap against the
week-1 prompt on the same questions (1000 resamples). The table is reproduced from the saved files,
without a model, by `uv run python -m scripts.c4_verbalized_prompt --run qwen1.5b --from-saved`
(and `--run qwen3b`):

| model | 0-100 (week 1) | plain 0-10 | anchored 0-10 | anchored - week 1 | anchored - plain 0-10 |
|---|---|---|---|---|---|
| 1.5B (cpu, float32) | 0.538 | 0.600 | **0.644** | +0.107 [+0.040, +0.174] | +0.044 [-0.032, +0.117] |
| 3B (mps, float16) | 0.713 | 0.793 | **0.804** | +0.091 [+0.045, +0.139] | +0.011 [-0.030, +0.053] |

- 3B ran on MPS in float16 (12 GB in float32 does not fit next to the system on a 16 GB Mac). The
  week-1 prompt rerun in that setup (`plain100`) gives exactly the saved scores (same 200 values), so the
  comparison is fair.
- Most of the gain comes from the smaller scale; the anchoring adds a little more, but that part is
  inside the noise on both models.
- **Decision.** The C4 rule was: if AUROC stays below 0.65 on 1.5B *and* 3B, verbalized becomes a
  baseline only. With the anchored prompt 1.5B is just below (0.644) and 3B is clearly above (0.804),
  so verbalized stays a method, and the anchored prompt is the better prompt on both models. It is
  still far below zero-shot p(True) (0.825 / 0.877), which needs the same one extra model call.
- Not done yet: `verbalized_2s` in the main table still uses the week-1 prompt. Switching it to the
  anchored prompt needs a rerun on all four sizes (7B needs more than 16 GB), so it is a follow-up.
  0.5B is not part of the rule; its earlier anchored number (0.531) was made before the 0-10 parsing fix.
## Hybrid scores (C5, CoCoA)

CoCoA (Vashurin et al. 2025, https://arxiv.org/abs/2502.04964) multiplies a confidence signal of the
greedy answer with how much the samples disagree with it:
`uncertainty = u(greedy) * (1 - mean similarity(greedy, sample_i))`. Code: `common/hybrid.py`.

- `cocoa_msp`: `u = -log P(greedy)` (the `msp` score); cache only, no model call.
- `cocoa_p_true`: `u = 1 - p(True)` (the `p_true_zeroshot` score); the plugin runs p(True),
  `scripts/hybrid_from_results.py` builds the same record from a saved `p_true_zeroshot.json`.
- Similarity = mean entailment probability greedy -> sample and sample -> greedy, read from the shared
  NLI matrix (deberta-large, A5 cache), so neither method adds an NLI pass. The paper uses a
  cross-encoder similarity instead.

AUROC on TriviaQA-200 (`results/<model>/triviaqa_200/`), compared with the two ingredients
("inconsistency" = `1 - mean similarity` alone, computed with `common.hybrid.inconsistency`):

| method | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| msp | 0.680 | 0.748 | 0.753 | 0.824 |
| p_true_zeroshot | 0.669 | 0.825 | **0.877** | **0.870** |
| inconsistency alone | 0.736 | 0.766 | 0.738 | 0.724 |
| cocoa_msp | 0.715 | 0.759 | 0.737 | 0.733 |
| cocoa_p_true | 0.734 | **0.833** | 0.786 | 0.752 |

- The hybrid beats both ingredients only on 1.5B (`cocoa_p_true` 0.833 vs 0.825, inside the
  intervals of about ±0.06). On 0.5B it helps the weak p(True) (+0.065) but does not beat consistency.
- From 3B on it is clearly worse than p(True) alone (-0.09 on 3B, -0.12 on 7B; the paired
  intervals in `summary.md` exclude 0). Reason: the product is about 0 (score < 0.01) for
  54 / 46 / 57 / 79 % of the questions (0.5B / 1.5B / 3B / 7B), and then the ranking among those
  questions is lost. A factor near 0 has two causes: all samples repeat the greedy answer
  (inconsistency < 0.01: 6 / 18 / 38 / 60 % of the questions) or p(True) is about 1
  (1 - p(True) < 0.01: 44 / 25 / 42 / 68 %); some questions have both. On 0.5B it is almost only
  the confident p(True); on 3B and 7B both causes are about equally common and often come
  together: bigger models repeat their answers also when they are wrong (README finding 3) and
  are then also sure of them. (Thanks to Eylül's reviews of #82.)
- Decision: both stay in the table as the `hybrid` family; the selector should not prefer them
  over p(True) from 3B on. A weighted sum or the logistic stack (D2) avoids the zero product.

## Known limitations

- The hand labels come from one person (60 pairs); a second person labels a subset to measure how much
  people agree (`label_nli_pairs --second-labeller`).
- The results in `results/<model>/triviaqa_200/` were made with deberta-base; they are rerun with the new default
  on day 4.
- SE differs from discrete SE only in the likelihood weights, so the two give almost the same AUROC (0.740 vs 0.744 at 0.5B, 0.781 vs 0.783 at 1.5B).
- Perplexity is not a separate method: perplexity = exp(−likelihood_mean), and exp keeps the order, so it gives
  exactly the same ranking (AUROC, AURC and PRR are identical; checked on 0.5B, 1.5B, 3B and 7B). It was removed
  in week 3 so the selector does not get the same method twice.
