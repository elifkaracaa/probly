Run `qwen3b/triviaqa_1000`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.460 | 0.425 | 0.495 | worse | -0.356 | -0.400 | -0.313 | 0.573 | -0.049 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.754 | 0.726 | 0.783 | worse | -0.062 | -0.085 | -0.038 | 0.397 | 0.452 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.798 | 0.772 | 0.824 | worse | -0.018 | -0.027 | -0.009 | 0.354 | 0.637 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.816 | 0.790 | 0.841 | best | 0.000 | 0.000 | 0.000 | 0.342 | 0.668 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.769 | 0.740 | 0.799 | worse | -0.047 | -0.070 | -0.023 | 0.362 | 0.606 |  | 0 |  | 0.000 |
