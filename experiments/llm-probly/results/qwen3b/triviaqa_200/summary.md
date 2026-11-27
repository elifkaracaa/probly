Run `qwen3b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| largest_cluster_share | agreement | text_only | 12 | ok | 0.758 | 0.696 | 0.817 | worse | -0.119 | -0.183 | -0.059 | 0.355 | 0.460 |  | 0 | 0.000 |
| greedy_cluster_share | agreement | text_only | 12 | ok | 0.761 | 0.697 | 0.818 | worse | -0.117 | -0.178 | -0.056 | 0.354 | 0.464 |  | 0 | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.765 | 0.703 | 0.823 | worse | -0.112 | -0.173 | -0.052 | 0.353 | 0.467 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.500 | 0.417 | 0.584 | worse | -0.378 | -0.473 | -0.286 | 0.504 | 0.093 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.757 | 0.692 | 0.816 | worse | -0.121 | -0.189 | -0.056 | 0.363 | 0.457 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.786 | 0.719 | 0.843 | worse | -0.091 | -0.156 | -0.027 | 0.321 | 0.631 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.817 | 0.757 | 0.871 | worse | -0.061 | -0.123 | -0.000 | 0.299 | 0.684 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.753 | 0.682 | 0.818 | worse | -0.124 | -0.195 | -0.055 | 0.342 | 0.573 |  | 0 | 0.000 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.820 | 0.764 | 0.873 | worse | -0.057 | -0.107 | -0.012 | 0.299 | 0.689 | 0.259 | 0 | 0.471 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.830 | 0.775 | 0.880 | worse | -0.047 | -0.081 | -0.012 | 0.298 | 0.650 | 0.181 | 0 | 3.157 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.877 | 0.830 | 0.921 | best | 0.000 | 0.000 | 0.000 | 0.262 | 0.793 | 0.176 | 0 | 0.244 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.761 | 0.699 | 0.819 | worse | -0.116 | -0.176 | -0.057 | 0.355 | 0.463 |  | 0 | 0.125 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.713 | 0.659 | 0.766 | worse | -0.164 | -0.223 | -0.103 | 0.395 | 0.350 | 0.412 | 0 | 0.771 |
