Run `smollm2-1.7b/triviaqa_5`, 5 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.500 | 0.000 | 1.000 | tied | -0.333 | -1.000 | 0.000 | 0.383 | 0.198 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.833 | 0.250 | 1.000 | best | 0.000 | 0.000 | 0.000 | 0.196 | 0.753 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.667 | 0.000 | 1.000 | tied | -0.167 | -0.667 | 0.000 | 0.321 | 0.383 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.833 | 0.250 | 1.000 | tied | 0.000 | 0.000 | 0.000 | 0.196 | 0.753 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.667 | 0.000 | 1.000 | tied | -0.167 | -0.750 | 0.000 | 0.258 | 0.568 |  | 0 |  | 0.000 |
