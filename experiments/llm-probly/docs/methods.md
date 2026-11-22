# Methods

All methods are plugins in `methods/` (see CONTRACT.md). Each one has a family, and each one reads the same cached generations.

Family, model access (`text_only` / `logprobs` / `white_box`) and cost per question (generation passes,
greedy answer included) are kept in one place, `common/registry.py`; `uv run python -m cli methods` prints them.
The selector reads the same registry.

| Method | Family | Needs logits? | Model calls per question |
|---|---|---|---|
| semantic_entropy | agreement | yes (sample log-likelihoods, `samples_ll_mean`) | 0 LLM calls (reads the 10 cached samples) + one shared NLI matrix per question |
| discrete_semantic_entropy | agreement | no (cluster sizes only) | 0 LLM calls + the same shared NLI matrix |
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

## NLI setup (A4)

Which NLI model, and should the question go in front of each answer? Compared on qwen0.5b and qwen1.5b
TriviaQA-200 with `scripts/compare_nli_setups.py` (full table: `results/nli_setups.md`), against 60 answer
pairs labelled by hand (`data/nli_hand_labels.jsonl`, made with `scripts/label_nli_pairs.py`). The
agreement with the labels is weighted to the real mix of answer pairs (the labels over-sample the hard pairs).

| setup | agreement with hand labels | SE AUROC 0.5B / 1.5B | s per question 0.5B / 1.5B |
|---|---|---|---|
| deberta-base | 0.936 | 0.740 / 0.781 | 0.22 / 0.07 |
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

## Known limitations

- The hand labels come from one person (60 pairs); a second person labels a subset to measure how much
  people agree (`label_nli_pairs --second-labeller`).
- The results in `results/<model>/triviaqa_200/` were made with deberta-base; they are rerun with the new default
  on day 4.
- SE differs from discrete SE only in the likelihood weights, so the two give almost the same AUROC (0.740 vs 0.744 at 0.5B, 0.781 vs 0.783 at 1.5B).
- Perplexity is not a separate method: perplexity = exp(−likelihood_mean), and exp keeps the order, so it gives
  exactly the same ranking (AUROC, AURC and PRR are identical; checked on 0.5B, 1.5B, 3B and 7B). It was removed
  in week 3 so the selector does not get the same method twice.
