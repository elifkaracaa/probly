Run `smollm2-1.7b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.527 | 0.440 | 0.605 | tied | -0.100 | -0.214 | 0.017 | 0.523 | 0.103 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.625 | 0.548 | 0.701 | tied | -0.002 | -0.064 | 0.067 | 0.460 | 0.289 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.583 | 0.510 | 0.668 | tied | -0.044 | -0.135 | 0.058 | 0.476 | 0.310 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.564 | 0.488 | 0.650 | tied | -0.063 | -0.164 | 0.050 | 0.498 | 0.221 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.627 | 0.546 | 0.702 | best | 0.000 | 0.000 | 0.000 | 0.452 | 0.366 |  | 0 |  | 0.000 |
