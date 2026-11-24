# Embedding backend vs NLI: `qwen0.5b/triviaqa_200`

Embedding model `sentence-transformers/all-mpnet-base-v2`; threshold chosen on `qwen0.5b`: **0.9**.

## Speed (this machine, same questions)

| backend | seconds per question | speed-up |
|---|---|---|
| NLI (deberta-large, A2 + A3) | 0.971 | 1x |
| embedding | 0.030 | 33x |

## AUROC

| backend | semantic_entropy | discrete_semantic_entropy | eig_val_laplacian | deg_mat | num_semantic_sets | mean clusters | same clusters as NLI |
|---|---|---|---|---|---|---|---|
| NLI (deberta-large) | 0.744 | 0.751 | 0.758 | 0.751 | 0.759 | 5.05 |  |
| embedding, t = 0.6 | 0.713 | 0.711 | 0.778 | 0.769 | 0.708 | 2.48 | 14% |
| embedding, t = 0.65 | 0.712 | 0.713 | 0.777 | 0.769 | 0.714 | 2.69 | 13% |
| embedding, t = 0.7 | 0.733 | 0.738 | 0.777 | 0.769 | 0.734 | 3.01 | 16% |
| embedding, t = 0.75 | 0.733 | 0.738 | 0.775 | 0.766 | 0.738 | 3.40 | 17% |
| embedding, t = 0.8 | 0.742 | 0.746 | 0.774 | 0.765 | 0.752 | 4.04 | 19% |
| embedding, t = 0.85 | 0.735 | 0.745 | 0.775 | 0.764 | 0.751 | 4.76 | 24% |
| embedding, t = 0.9 | 0.745 | 0.755 | 0.774 | 0.762 | 0.763 | 5.53 | 30% |
| embedding, t = 0.95 | 0.702 | 0.713 | 0.774 | 0.762 | 0.723 | 6.31 | 31% |
