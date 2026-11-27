Run `qwen1.5b/triviaqa_1000_embedding`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semantic_entropy | agreement | logprobs | 12 | ok | 0.759 | 0.728 | 0.789 | tied | -0.005 | -0.019 | 0.009 | 0.462 | 0.497 |  | 0 | 0.017 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.764 | 0.732 | 0.793 | tied | -0.001 | -0.014 | 0.012 | 0.458 | 0.504 |  | 0 | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.763 | 0.731 | 0.793 | tied | -0.002 | -0.015 | 0.012 | 0.457 | 0.502 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.764 | 0.733 | 0.794 | best | 0.000 | 0.000 | 0.000 | 0.462 | 0.496 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.759 | 0.728 | 0.790 | worse | -0.005 | -0.009 | -0.000 | 0.465 | 0.486 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.743 | 0.710 | 0.775 | worse | -0.022 | -0.033 | -0.011 | 0.472 | 0.465 |  | 0 | 0.000 |
