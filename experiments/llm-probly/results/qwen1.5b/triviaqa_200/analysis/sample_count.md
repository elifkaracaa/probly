# Sample-count curve: `qwen1.5b/triviaqa_200`

NLI setup: `deberta-large`. AUROC (mean over random subsets of the 10 cached samples; std in brackets).

| method | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 | K=10 |
|---|---|---|---|---|---|---|---|---|---|
| SE | 0.690 (0.029) | 0.726 (0.014) | 0.749 (0.007) | 0.754 (0.011) | 0.765 (0.008) | 0.767 (0.008) | 0.773 (0.005) | 0.775 (0.004) | 0.776 (0.000) |
| discrete SE | 0.690 (0.027) | 0.727 (0.011) | 0.752 (0.008) | 0.758 (0.011) | 0.769 (0.009) | 0.770 (0.008) | 0.776 (0.007) | 0.778 (0.004) | 0.779 (0.000) |
| lexical similarity | 0.684 (0.019) | 0.715 (0.015) | 0.728 (0.010) | 0.733 (0.005) | 0.737 (0.006) | 0.744 (0.007) | 0.745 (0.004) | 0.748 (0.003) | 0.752 (0.000) |

## Early stop

Rule: generate `check_at` samples; stop if they all fall into one semantic cluster,
otherwise generate all 10. AUROC change is against all 10 samples.

Accuracy columns: share of correct greedy answers among the stopped and the not-stopped
questions. A stopped wrong answer gets entropy 0 and ties with the stopped correct ones,
which is where most of the AUROC change comes from.

| method | check_at | stopped | accuracy stopped | accuracy not stopped | samples / question | saved generations | AUROC | AUROC change |
|---|---|---|---|---|---|---|---|---|
| SE | 2 | 60% | 49% | 20% | 5.16 | 48% | 0.672 | -0.104 |
| discrete SE | 2 | 60% | 49% | 20% | 5.16 | 48% | 0.672 | -0.107 |
| lexical similarity | 2 | 60% | 49% | 20% | 5.16 | 48% | 0.657 | -0.094 |
| SE | 3 | 48% | 58% | 19% | 6.67 | 33% | 0.744 | -0.032 |
| discrete SE | 3 | 48% | 58% | 19% | 6.67 | 33% | 0.746 | -0.033 |
| lexical similarity | 3 | 48% | 58% | 19% | 6.67 | 33% | 0.722 | -0.030 |
| SE | 4 | 43% | 58% | 22% | 7.42 | 26% | 0.739 | -0.038 |
| discrete SE | 4 | 43% | 58% | 22% | 7.42 | 26% | 0.742 | -0.037 |
| lexical similarity | 4 | 43% | 58% | 22% | 7.42 | 26% | 0.723 | -0.029 |
| SE | 5 | 41% | 60% | 22% | 7.95 | 20% | 0.748 | -0.028 |
| discrete SE | 5 | 41% | 60% | 22% | 7.95 | 20% | 0.752 | -0.028 |
| lexical similarity | 5 | 41% | 60% | 22% | 7.95 | 20% | 0.730 | -0.022 |
