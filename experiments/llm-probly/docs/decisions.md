# Decision log

One line per scope or design decision: what we decided, why, and where it happened. Details stay in
the linked PR or doc; this page is the index. Add new decisions at the top of their section, with the
PR that carries them. PR numbers refer to this fork (`elifkaracaa/probly`).

## Methods (which scores exist)

| date | decision | why | where |
|---|---|---|---|
| 28 Nov | Drop `likelihood_sum` and `scripts/baseline_likelihood.py`. | Same formula as `msp` (`-sum(greedy_token_logprobs)` = `-greedy_ll_sum`, equal on all 200 rows of qwen0.5b/triviaqa_200); no plugin, only a week-1 file on 50 questions. Nothing is lost: it is in every table as `msp`. | #98 |
| 28 Nov | The hybrid scores (`cocoa_msp`, `cocoa_p_true`) stay in the table as family `hybrid`, but are not preferred. | They win only on 1.5B (within noise) and lose to p(True) from 3B on; the product is about 0 when the samples agree or p(True) is about 1. The selector rerun changes no pick. | #82, #96, [methods.md](methods.md#hybrid-scores-c5-cocoa) |
| 28 Nov | Verbalized confidence stays a method; the anchored 0-10 prompt is the better prompt. | The rule for the new verbalized-confidence prompt (baseline only if AUROC < 0.65 on 1.5B and 3B) is not met: anchored gives 0.644 (1.5B) and 0.804 (3B), +0.107 / +0.091 over the week-1 prompt (paired intervals above 0). The main table keeps the week-1 prompt until all sizes are rerun. | #76, [methods.md](methods.md#verbalized-confidence-prompt-c4) |
| 20 Nov | Drop `perplexity`. | `perplexity = exp(likelihood_mean)` keeps the order, so AUROC, AURC and PRR are identical on all four sizes; the selector would get the same method twice. | #47 |
| 9 Nov | p(True) uses the `plain_ab` prompt (no chat template, next token A vs B). | Chosen by the week-1 sanity comparison against `chat_groups` (`notebooks/reflexive_mode_comparison.ipynb`). | #6 |

## Semantic methods and NLI

| date | decision | why | where |
|---|---|---|---|
| 23 Nov | No answer normalization (case, punctuation) before deduplication. | The NLI model already treats such pairs as the same meaning: 1-3 % fewer NLI pairs, no SE/DSE score changes on 0.5B/1.5B; normalization would join "3.5"/"35". | #64, [methods.md](methods.md) |
| 23 Nov | Default NLI setup: deberta-large on the answers only (probly's library default stays deberta-base). | Best agreement with 80 hand-labelled pairs (0.971, second labeller 20/20); same SE AUROC as the other setups; the extra time is paid once per cache. The question in front doubles the time with no consistent gain. | #56, [methods.md](methods.md#nli-setup-a4) |

## Self-evaluation code

| date | decision | why | where |
|---|---|---|---|
| 28 Nov | Keep `common/reflexive.py` and `scripts/r3_methods.py` as the week-1 reference, typed and linted; one verbalized implementation (`common/verbalized.py`). | The equivalence test of the self-evaluation representer, the bench script and the week-1 notebooks use them; deleting them would break those. | #80 |
| 28 Nov | Rerun `p_true_empirical` with the library code on 0.5B and 1.5B; 3B and 7B keep the week-2 run until the 24 GB machine reruns them. | The batched empirical p(True) sampling draws different random numbers, so the stored week-2 scores were not reproducible. New AUROC 0.512 / 0.765 (sampling difference only). 3B float32 does not fit on a 16 GB Mac. | #81 |
| 27 Nov | Empirical p(True) samples one question per `generate` call by default. | Several questions per call were 3-5x slower on CPU; the batch size stays a parameter for a GPU. | #75 |

## Data and runs

| date | decision | why | where |
|---|---|---|---|
| 23 Nov | Qwen2.5-7B on MMLU is dropped for this week. | mps runs out of memory on the long MMLU prompts, cpu bfloat16 is too slow, mps batches of 2 give NaN logits; all 7B TriviaQA results are complete. | #65, [benchmark_caches.md](benchmark_caches.md) |

## Repository and process

| date | decision | why | where |
|---|---|---|---|
| 28 Nov | `main` is protected by a ruleset: merging needs a PR with 1 approval and green `Code Quality Checks`, `Experiment Tests (ubuntu, py3.13)` and `Test Documentation Build / build`; no force pushes or deletion; only the repository admin can bypass, and only for pull requests. | It enforces the team rule that every PR is merged only after its reviewer approved it and the checks pass, instead of relying on everyone remembering it. | Repository settings (Rulesets) |
| 28 Nov | Deploy Docs is disabled on the fork (GitHub Actions setting, reversible). | It is upstream's deployment to pwhofman.github.io and needs a release asset the fork does not have, so it failed on every push to main. | Actions settings |
| 28 Nov | Docstring examples that need a model download are literal blocks (`::`), not `>>>` with `# doctest: +SKIP`. | sphinx-codeautolink cannot match the trimmed doctest flags, and the strict docs build (`-W`) turned that into a failure on every PR. | #85 |
| 28 Nov | Alexandru's week-3 f-branch PRs (#42, #45, #50, #55) are continued in #74, #77, #79 (his work kept or credited; #42's text goes into the README rewrite) and closed with a link once those are merged and he agrees. | The old branches conflict with main and would bring back problems already fixed (model download in the docs build, outdated positioning page, dead links). | #74, #77, #79 |
| 27 Nov | Erim's remaining self-evaluation tasks (self-evaluation representer in probly, batched empirical p(True), coverage column, new verbalized-confidence prompt, hybrid scores) are taken over by Elif; his commits stay unchanged, fixes go on top. | This work was not in main at the end of week 3. | #46, #75, #54, #76, #82 |
