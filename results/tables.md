# Results (5-fold CV, mean ± std)

*random* = stratified event-level folds (BiGCN protocol). *group* = leakage-free: no claim/story group in both train and test. Chance macro-F1 = 0.25.


## Macro-F1

| Model | T15 random | T15 group | T16 random | T16 group |
|---|---|---|---|---|
| Structure features + RF | 0.311 ± 0.023 | 0.290 ± 0.032 | 0.311 ± 0.024 | 0.295 ± 0.049 |
| Source text TF-IDF + LR | 0.837 ± 0.016 | 0.298 ± 0.056 | 0.849 ± 0.022 | 0.319 ± 0.067 |
| Source text TF-IDF + MLP | 0.797 ± 0.015 | 0.294 ± 0.055 | 0.844 ± 0.024 | 0.327 ± 0.074 |
| Cascade words (source+replies) + LR | 0.825 ± 0.015 | 0.445 ± 0.051 | 0.818 ± 0.005 | 0.497 ± 0.113 |
| Source text + structure + LR | 0.840 ± 0.016 | 0.324 ± 0.052 | 0.847 ± 0.023 | 0.404 ± 0.096 |
| BiGCN (re-impl.) | 0.829 ± 0.011 | 0.382 ± 0.060 | 0.842 ± 0.011 | 0.410 ± 0.105 |
| BiGCN + text (TF-IDF), gated fusion | 0.816 ± 0.026 | 0.367 ± 0.058 | 0.826 ± 0.014 | 0.400 ± 0.120 |

## Accuracy

| Model | T15 random | T15 group | T16 random | T16 group |
|---|---|---|---|---|
| Structure features + RF | 0.312 ± 0.023 | 0.294 ± 0.031 | 0.312 ± 0.022 | 0.296 ± 0.048 |
| Source text TF-IDF + LR | 0.836 ± 0.017 | 0.342 ± 0.052 | 0.847 ± 0.022 | 0.364 ± 0.046 |
| Source text TF-IDF + MLP | 0.795 ± 0.016 | 0.330 ± 0.046 | 0.842 ± 0.024 | 0.361 ± 0.055 |
| Cascade words (source+replies) + LR | 0.824 ± 0.015 | 0.472 ± 0.043 | 0.818 ± 0.005 | 0.519 ± 0.099 |
| Source text + structure + LR | 0.839 ± 0.017 | 0.367 ± 0.045 | 0.846 ± 0.023 | 0.439 ± 0.081 |
| BiGCN (re-impl.) | 0.828 ± 0.011 | 0.416 ± 0.056 | 0.842 ± 0.010 | 0.439 ± 0.094 |
| BiGCN + text (TF-IDF), gated fusion | 0.815 ± 0.025 | 0.402 ± 0.054 | 0.826 ± 0.015 | 0.429 ± 0.101 |

## LLM stage (same class-balanced events in every row of a dataset)

| Run | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Twitter16_classify_source_mock | 60 | 0.200 | 0.199 |

## Early detection (BiGCN, macro-F1 with only the first k responses at test time)

| Dataset / split | k=0 | k=5 | k=10 | k=20 | full |
|---|---|---|---|---|---|
| Twitter15 random | 0.729 | 0.794 | 0.801 | 0.815 | 0.829 |
| Twitter15 group | 0.364 | 0.368 | 0.371 | 0.373 | 0.382 |
| Twitter16 random | 0.776 | 0.806 | 0.813 | 0.827 | 0.842 |
| Twitter16 group | 0.377 | 0.394 | 0.393 | 0.405 | 0.410 |

## Explanation faithfulness – Twitter15_group_bigcn_none

| Nodes | Comprehensiveness ↑ (saliency) | (random) | Sufficiency ↓ (saliency) | (random) | n |
|---|---|---|---|---|---|
| top10% | 0.063 | 0.001 | -0.108 | 0.069 | 1351 |
| top20% | 0.107 | 0.002 | -0.115 | 0.053 | 1351 |
| top30% | 0.142 | 0.004 | -0.111 | 0.042 | 1351 |

## Explanation faithfulness – Twitter16_group_bigcn_none

| Nodes | Comprehensiveness ↑ (saliency) | (random) | Sufficiency ↓ (saliency) | (random) | n |
|---|---|---|---|---|---|
| top10% | 0.048 | 0.002 | -0.057 | 0.067 | 706 |
| top20% | 0.078 | 0.005 | -0.070 | 0.053 | 706 |
| top30% | 0.101 | 0.007 | -0.070 | 0.044 | 706 |
