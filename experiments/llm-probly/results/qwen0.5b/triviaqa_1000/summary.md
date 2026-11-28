Run `qwen0.5b/triviaqa_1000`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deg_mat | agreement | text_only | 12 | ok | 0.743 | 0.704 | 0.785 | worse | -0.009 | -0.016 | -0.001 | 0.668 | 0.478 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.732 | 0.688 | 0.774 | worse | -0.021 | -0.032 | -0.009 | 0.671 | 0.458 |  | 0 |  | 0.001 |
| dummy_random | baseline | text_only | 1 | ok | 0.515 | 0.465 | 0.561 | worse | -0.237 | -0.298 | -0.174 | 0.780 | 0.065 |  | 0 |  | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.710 | 0.667 | 0.752 | worse | -0.042 | -0.062 | -0.024 | 0.684 | 0.426 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.752 | 0.712 | 0.794 | best | 0.000 | 0.000 | 0.000 | 0.663 | 0.498 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.681 | 0.642 | 0.717 | worse | -0.072 | -0.112 | -0.033 | 0.714 | 0.325 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.712 | 0.672 | 0.747 | worse | -0.040 | -0.085 | -0.000 | 0.689 | 0.401 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.696 | 0.656 | 0.735 | worse | -0.056 | -0.104 | -0.012 | 0.698 | 0.372 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.639 | 0.595 | 0.683 | worse | -0.113 | -0.150 | -0.076 | 0.720 | 0.302 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.735 | 0.693 | 0.778 | worse | -0.017 | -0.028 | -0.006 | 0.669 | 0.460 |  | 0 |  | 0.000 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.730 | 0.687 | 0.773 | worse | -0.022 | -0.036 | -0.009 | 0.671 | 0.456 |  | 0 |  | 1.254 |
