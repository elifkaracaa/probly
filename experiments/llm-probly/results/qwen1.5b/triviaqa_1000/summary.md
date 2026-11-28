Run `qwen1.5b/triviaqa_1000`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deg_mat | agreement | text_only | 12 | ok | 0.775 | 0.746 | 0.804 | tied | -0.024 | -0.055 | 0.007 | 0.451 | 0.520 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.775 | 0.745 | 0.803 | tied | -0.024 | -0.053 | 0.006 | 0.452 | 0.506 |  | 0 |  | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.503 | 0.467 | 0.540 | worse | -0.296 | -0.343 | -0.252 | 0.619 | 0.047 |  | 0 |  | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.753 | 0.721 | 0.785 | worse | -0.045 | -0.079 | -0.012 | 0.463 | 0.495 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.777 | 0.746 | 0.806 | tied | -0.021 | -0.053 | 0.010 | 0.449 | 0.525 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.773 | 0.743 | 0.802 | tied | -0.026 | -0.056 | 0.003 | 0.454 | 0.514 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.789 | 0.759 | 0.818 | tied | -0.010 | -0.020 | 0.000 | 0.444 | 0.605 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.799 | 0.770 | 0.827 | best | 0.000 | 0.000 | 0.000 | 0.436 | 0.632 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.739 | 0.706 | 0.771 | worse | -0.060 | -0.093 | -0.028 | 0.465 | 0.547 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.772 | 0.743 | 0.801 | tied | -0.026 | -0.055 | 0.002 | 0.453 | 0.503 |  | 0 |  | 0.000 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.772 | 0.742 | 0.801 | tied | -0.027 | -0.057 | 0.003 | 0.454 | 0.503 |  | 0 |  | 0.412 |
