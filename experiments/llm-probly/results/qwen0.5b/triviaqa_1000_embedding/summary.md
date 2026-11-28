Run `qwen0.5b/triviaqa_1000_embedding`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deg_mat | agreement | text_only | 12 | ok | 0.720 | 0.682 | 0.756 | tied | -0.008 | -0.031 | 0.012 | 0.692 | 0.394 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.728 | 0.689 | 0.767 | best | 0.000 | 0.000 | 0.000 | 0.680 | 0.418 |  | 0 |  | 0.001 |
| eccentricity | agreement | text_only | 12 | ok | 0.709 | 0.673 | 0.747 | tied | -0.019 | -0.039 | 0.002 | 0.696 | 0.385 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.724 | 0.686 | 0.760 | tied | -0.004 | -0.027 | 0.016 | 0.690 | 0.401 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.727 | 0.687 | 0.764 | tied | -0.002 | -0.008 | 0.004 | 0.681 | 0.416 |  | 0 |  | 0.000 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.724 | 0.685 | 0.764 | tied | -0.004 | -0.008 | 0.000 | 0.682 | 0.412 |  | 0 |  | 0.027 |
