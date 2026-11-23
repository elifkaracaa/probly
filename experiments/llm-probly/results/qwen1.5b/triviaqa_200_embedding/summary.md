Run `qwen1.5b/triviaqa_200_embedding`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.748 | 0.679 | 0.811 | tied | -0.004 | -0.031 | 0.020 | 0.466 | 0.471 |  | 0 | 0.043 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.744 | 0.674 | 0.809 | tied | -0.008 | -0.037 | 0.019 | 0.465 | 0.466 |  | 0 | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.748 | 0.679 | 0.810 | tied | -0.005 | -0.032 | 0.021 | 0.464 | 0.471 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.752 | 0.686 | 0.816 | best | 0.000 | 0.000 | 0.000 | 0.470 | 0.461 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.751 | 0.683 | 0.815 | tied | -0.002 | -0.011 | 0.008 | 0.471 | 0.456 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.733 | 0.661 | 0.801 | tied | -0.020 | -0.045 | 0.006 | 0.473 | 0.440 |  | 0 | 0.000 |
