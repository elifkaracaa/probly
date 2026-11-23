Run `qwen0.5b/triviaqa_200_embedding`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.755 | 0.657 | 0.836 | tied | -0.019 | -0.071 | 0.029 | 0.690 | 0.472 |  | 0 | 0.043 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.745 | 0.644 | 0.833 | tied | -0.029 | -0.084 | 0.023 | 0.693 | 0.464 |  | 0 | 0.001 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.763 | 0.668 | 0.846 | tied | -0.011 | -0.062 | 0.038 | 0.683 | 0.486 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.774 | 0.687 | 0.847 | best | 0.000 | 0.000 | 0.000 | 0.682 | 0.500 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.762 | 0.669 | 0.837 | worse | -0.012 | -0.021 | -0.003 | 0.690 | 0.477 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.751 | 0.651 | 0.831 | tied | -0.023 | -0.062 | 0.014 | 0.689 | 0.482 |  | 0 | 0.000 |
