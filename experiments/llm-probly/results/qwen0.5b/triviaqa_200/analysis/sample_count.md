# Sample-count curve: `qwen0.5b/triviaqa_200`

AUROC (mean over random subsets of the 10 cached samples; std in brackets).

| method | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 | K=10 |
|---|---|---|---|---|---|---|---|---|---|
| SE | 0.624 (0.024) | 0.683 (0.026) | 0.709 (0.024) | 0.728 (0.025) | 0.730 (0.013) | 0.736 (0.009) | 0.736 (0.008) | 0.739 (0.003) | 0.740 (0.000) |
| discrete SE | 0.659 (0.026) | 0.702 (0.023) | 0.723 (0.022) | 0.738 (0.022) | 0.737 (0.012) | 0.743 (0.008) | 0.744 (0.007) | 0.746 (0.004) | 0.746 (0.000) |
| lexical similarity | 0.693 (0.028) | 0.714 (0.023) | 0.733 (0.015) | 0.748 (0.008) | 0.745 (0.008) | 0.745 (0.008) | 0.753 (0.008) | 0.755 (0.005) | 0.757 (0.000) |

## Early stop

Rule: generate `check_at` samples; stop if they all fall into one semantic cluster,
otherwise generate all 10. AUROC change is against all 10 samples.

Accuracy columns: share of correct greedy answers among the stopped and the not-stopped
questions. A stopped wrong answer gets entropy 0 and ties with the stopped correct ones,
which is where most of the AUROC change comes from.

| method | check_at | stopped | accuracy stopped | accuracy not stopped | samples / question | saved generations | AUROC | AUROC change |
|---|---|---|---|---|---|---|---|---|
| SE | 2 | 35% | 33% | 12% | 7.20 | 28% | 0.666 | -0.073 |
| discrete SE | 2 | 35% | 33% | 12% | 7.20 | 28% | 0.671 | -0.075 |
| lexical similarity | 2 | 35% | 33% | 12% | 7.20 | 28% | 0.678 | -0.078 |
| SE | 3 | 24% | 47% | 10% | 8.36 | 16% | 0.723 | -0.017 |
| discrete SE | 3 | 24% | 47% | 10% | 8.36 | 16% | 0.727 | -0.019 |
| lexical similarity | 3 | 24% | 47% | 10% | 8.36 | 16% | 0.742 | -0.015 |
| SE | 4 | 20% | 51% | 11% | 8.83 | 12% | 0.733 | -0.006 |
| discrete SE | 4 | 20% | 51% | 11% | 8.83 | 12% | 0.739 | -0.007 |
| lexical similarity | 4 | 20% | 51% | 11% | 8.83 | 12% | 0.766 | +0.009 |
| SE | 5 | 18% | 53% | 12% | 9.10 | 9% | 0.735 | -0.005 |
| discrete SE | 5 | 18% | 53% | 12% | 9.10 | 9% | 0.741 | -0.005 |
| lexical similarity | 5 | 18% | 53% | 12% | 9.10 | 9% | 0.763 | +0.006 |
