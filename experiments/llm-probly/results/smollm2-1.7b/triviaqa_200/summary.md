Run `smollm2-1.7b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.525 | 0.440 | 0.612 | worse | -0.292 | -0.406 | -0.172 | 0.613 | 0.098 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.796 | 0.726 | 0.854 | tied | -0.020 | -0.077 | 0.047 | 0.445 | 0.585 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.809 | 0.738 | 0.871 | tied | -0.007 | -0.030 | 0.015 | 0.423 | 0.685 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.816 | 0.741 | 0.879 | best | 0.000 | 0.000 | 0.000 | 0.424 | 0.694 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.795 | 0.726 | 0.855 | tied | -0.021 | -0.085 | 0.044 | 0.431 | 0.671 |  | 0 |  | 0.000 |
