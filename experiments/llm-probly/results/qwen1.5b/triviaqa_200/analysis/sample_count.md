# Sample-count curve: `qwen1.5b/triviaqa_200`

AUROC (mean over random subsets of the 10 cached samples; std in brackets).

| method | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 | K=10 |
|---|---|---|---|---|---|---|---|---|---|
| SE | 0.690 (0.027) | 0.729 (0.012) | 0.751 (0.009) | 0.760 (0.008) | 0.770 (0.007) | 0.771 (0.008) | 0.778 (0.006) | 0.780 (0.004) | 0.781 (0.000) |
| discrete SE | 0.691 (0.025) | 0.729 (0.011) | 0.752 (0.010) | 0.762 (0.009) | 0.773 (0.007) | 0.772 (0.008) | 0.780 (0.007) | 0.781 (0.004) | 0.783 (0.000) |
| lexical similarity | 0.684 (0.019) | 0.715 (0.015) | 0.728 (0.010) | 0.733 (0.005) | 0.737 (0.006) | 0.744 (0.007) | 0.745 (0.004) | 0.748 (0.003) | 0.752 (0.000) |

## Early stop

Rule: generate `check_at` samples; stop if they all fall into one semantic cluster,
otherwise generate all 10. AUROC change is against all 10 samples.

Accuracy columns: share of correct greedy answers among the stopped and the not-stopped
questions. A stopped wrong answer gets entropy 0 and ties with the stopped correct ones,
which is where most of the AUROC change comes from.

| method | check_at | stopped | accuracy stopped | accuracy not stopped | samples / question | saved generations | AUROC | AUROC change |
|---|---|---|---|---|---|---|---|---|
| SE | 2 | 58% | 50% | 20% | 5.32 | 47% | 0.682 | -0.099 |
| discrete SE | 2 | 58% | 50% | 20% | 5.32 | 47% | 0.683 | -0.100 |
| lexical similarity | 2 | 58% | 50% | 20% | 5.32 | 47% | 0.658 | -0.094 |
| SE | 3 | 44% | 59% | 21% | 6.92 | 31% | 0.749 | -0.032 |
| discrete SE | 3 | 44% | 59% | 21% | 6.92 | 31% | 0.752 | -0.031 |
| lexical similarity | 3 | 44% | 59% | 21% | 6.92 | 31% | 0.730 | -0.022 |
| SE | 4 | 40% | 60% | 22% | 7.60 | 24% | 0.750 | -0.031 |
| discrete SE | 4 | 40% | 60% | 22% | 7.60 | 24% | 0.753 | -0.030 |
| lexical similarity | 4 | 40% | 60% | 22% | 7.60 | 24% | 0.735 | -0.017 |
| SE | 5 | 38% | 61% | 23% | 8.07 | 19% | 0.755 | -0.026 |
| discrete SE | 5 | 38% | 61% | 23% | 8.07 | 19% | 0.758 | -0.025 |
| lexical similarity | 5 | 38% | 61% | 23% | 8.07 | 19% | 0.736 | -0.015 |
