# Models and cost

All runs: TriviaQA, 200 questions, on one MacBook Air (Apple M-series, 24 GB unified memory).

| model | precision | device | memory (python process) | generation s/question | accuracy |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | float32 | CPU | ~1 GB | 2.4 | 0.190 |
| Qwen2.5-1.5B | float32 | CPU | ~3 GB | 4.0 | 0.375 |
| Qwen2.5-3B | float32 | MPS | ~16 GB | 5.1 | 0.485 |
| Qwen2.5-7B | **float16** | MPS | ~17–21 GB | 9.2 | 0.590 |

Extra cost of each method, seconds per question on top of generation (`seconds_per_question` in the result files;
the LLM calls ran on CPU for 0.5B and 1.5B and on MPS for 3B and 7B, so compare within a column):

| method | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| cache-only (msp, likelihood_mean, mean_token_entropy, lexical_similarity) | ~0 | ~0 | ~0 | ~0 * |
| p_true_zeroshot | 0.11 | 0.31 | 0.24 | 0.34 |
| p_true_brainstorm | 0.29 | 0.50 | 0.47 | 0.68 |
| verbalized_2s | 0.19 | 0.56 | 0.77 | 0.91 |
| p_true_empirical (10 generations) | 0.57 † | 1.59 † | 3.16 | 5.93 |
| semantic_entropy + discrete SE (NLI, deberta-large on MPS, one shared matrix) | 0.67 | 0.25 | 0.13 | 0.06 |

The NLI time (`nli_seconds_per_question`) is paid once for SE, DSE and the graph measures together, and a
second run reads the NLI cache and pays 0. It falls with model size because bigger models give more
identical answers, which are scored only once. \* The 7B `lexical_similarity.json` records 7.75 s from its
first run; ROUGE-L on 10 short answers takes well under a second, so this is not the method's cost.
† 0.5B and 1.5B were rerun with the library's empirical p(True) (batched sampling, `HFSelfEvaluator.sample_judgments`); before: 0.85 / 2.25 s,
AUROC 0.518 / 0.770 (now 0.512 / 0.765, a sampling difference). 3B and 7B still have the week-2 run.

**Mac limits**
- 3B in float32 fits: memory pressure stays green while generating, with a short yellow peak while the weights load.
- 7B in float32 needs ~30 GB and does not fit, so 7B runs in float16 only. Precision changes answers (0.5B: 9 of the first 50 answers changed between bfloat16 and float32), so 7B is not fully comparable with the other sizes.
- Long runs get slower on the Air (no fan): 7B took 7.3 s/question on 5 questions and 9.2 s/question on 200.
- For long runs, keep the computer awake (macOS: put `caffeinate -i` in front of the command; Linux: `systemd-inhibit`; Windows: the power settings); `scripts.generate` saves after every question and resumes.
- Load test before a long run: `uv run python -m scripts.generate --model qwen3b --n 5 --device mps`, with Activity Monitor › Memory open.
