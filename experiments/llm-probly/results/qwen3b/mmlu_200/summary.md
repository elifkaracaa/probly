Run `qwen3b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.551 | 0.468 | 0.629 | worse | -0.218 | -0.330 | -0.111 | 0.333 | 0.150 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.601 | 0.553 | 0.650 | worse | -0.168 | -0.233 | -0.106 | 0.318 | 0.172 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.768 | 0.700 | 0.834 | tied | -0.001 | -0.005 | 0.003 | 0.209 | 0.587 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.769 | 0.702 | 0.834 | best | 0.000 | 0.000 | 0.000 | 0.208 | 0.592 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.768 | 0.700 | 0.834 | tied | -0.001 | -0.005 | 0.003 | 0.209 | 0.587 |  | 0 |  | 0.000 |
