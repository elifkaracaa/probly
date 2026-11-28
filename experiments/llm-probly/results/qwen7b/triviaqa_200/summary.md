Run `qwen7b/triviaqa_200`, 200 questions, precision float16

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cocoa_msp | hybrid | logprobs | 12 | ok | 0.733 | 0.663 | 0.798 | worse | -0.137 | -0.216 | -0.053 | 0.278 | 0.414 |  | 0 |  | 0.000 |
| cocoa_p_true | hybrid | logprobs | 13 | ok | 0.752 | 0.685 | 0.816 | worse | -0.118 | -0.190 | -0.040 | 0.272 | 0.432 |  | 0 |  | 0.344 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.729 | 0.667 | 0.789 | worse | -0.141 | -0.215 | -0.071 | 0.275 | 0.391 |  | 0 |  | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.560 | 0.477 | 0.637 | worse | -0.311 | -0.414 | -0.208 | 0.372 | 0.168 |  | 0 |  | 0.000 |
| greedy_cluster_share | agreement | text_only | 12 | ok | 0.718 | 0.651 | 0.777 | worse | -0.152 | -0.229 | -0.078 | 0.279 | 0.374 |  | 0 |  | 0.000 |
| largest_cluster_share | agreement | text_only | 12 | ok | 0.728 | 0.664 | 0.787 | worse | -0.143 | -0.219 | -0.072 | 0.275 | 0.390 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.735 | 0.668 | 0.800 | worse | -0.136 | -0.213 | -0.052 | 0.275 | 0.418 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.817 | 0.752 | 0.871 | tied | -0.053 | -0.120 | 0.015 | 0.209 | 0.704 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.833 | 0.772 | 0.885 | tied | -0.037 | -0.104 | 0.026 | 0.204 | 0.724 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.824 | 0.764 | 0.877 | tied | -0.046 | -0.115 | 0.026 | 0.207 | 0.719 |  | 0 |  | 0.000 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.845 | 0.785 | 0.896 | tied | -0.025 | -0.075 | 0.019 | 0.196 | 0.693 | 0.255 | 0 | 1.000 | 0.679 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.824 | 0.763 | 0.886 | worse | -0.046 | -0.084 | -0.008 | 0.218 | 0.633 | 0.171 | 14 | 0.836 | 5.927 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.870 | 0.815 | 0.920 | best | 0.000 | 0.000 | 0.000 | 0.189 | 0.767 | 0.220 | 0 | 1.000 | 0.344 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.729 | 0.667 | 0.790 | worse | -0.141 | -0.215 | -0.070 | 0.274 | 0.391 |  | 0 |  | 0.062 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.734 | 0.672 | 0.796 | worse | -0.137 | -0.202 | -0.070 | 0.279 | 0.409 | 0.320 | 0 |  | 0.905 |
