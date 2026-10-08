# Phase 4 results: Logistic Regression vs Random Forest

5-fold GroupKFold on `split_group_id`; `random_state=42`. PR-AUC and ROC-AUC are mean ± std over folds; @k metrics use the pooled out-of-fold scores. Lift = precision@k / positive rate.

**Note:** `flagged_neighbour_companies` is kept as a model feature (it is in the scope) but is near-constant: non-zero for only 7 of 406,715 companies, none of them positive. Rule 1 therefore ranks almost exactly like rule 2.

## Main label (PEP + sanctions)

406,715 companies, 52 positives (rate 0.01279%). Positives per fold: [20, 9, 9, 7, 7]. There are fewer positives than every k, so the best possible precision@k is 1.000 (k=50), 0.520 (k=100), 0.104 (k=500), 0.052 (k=1000).

| Method | PR-AUC | ROC-AUC | P@50 | P@100 | P@500 | P@1000 | R@50 | R@100 | R@500 | R@1000 | Lift@50 | Lift@100 | Lift@500 | Lift@1000 | Acc@0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.00019 ± 0.00007 | 0.513 ± 0.157 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7624 |
| Random Forest | 0.00020 ± 0.00005 | 0.540 ± 0.135 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9060 |
| Baseline: random | 0.00017 ± 0.00008 | 0.506 ± 0.057 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| Baseline: rule 1 (neighbours, then address) | 0.00032 ± 0.00045 | 0.498 ± 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| Baseline: rule 2 (shared address) | 0.00032 ± 0.00045 | 0.498 ± 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |

**Does each model beat each baseline?**

- Logistic Regression vs Baseline: random: **does NOT clearly beat** -- higher PR-AUC, but within fold-to-fold noise (0.00019 vs 0.00017); the same number of positives in the top 100 (0 vs 0)
- Logistic Regression vs Baseline: rule 1 (neighbours, then address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00019 vs 0.00032); the same number of positives in the top 100 (0 vs 0)
- Logistic Regression vs Baseline: rule 2 (shared address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00019 vs 0.00032); the same number of positives in the top 100 (0 vs 0)
- Random Forest vs Baseline: random: **does NOT clearly beat** -- higher PR-AUC, but within fold-to-fold noise (0.00020 vs 0.00017); the same number of positives in the top 100 (0 vs 0)
- Random Forest vs Baseline: rule 1 (neighbours, then address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00020 vs 0.00032); the same number of positives in the top 100 (0 vs 0)
- Random Forest vs Baseline: rule 2 (shared address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00020 vs 0.00032); the same number of positives in the top 100 (0 vs 0)

## PEP-only label

406,709 companies, 46 positives (rate 0.01131%). Positives per fold: [18, 5, 8, 5, 10]. There are fewer positives than every k, so the best possible precision@k is 0.920 (k=50), 0.460 (k=100), 0.092 (k=500), 0.046 (k=1000).

| Method | PR-AUC | ROC-AUC | P@50 | P@100 | P@500 | P@1000 | R@50 | R@100 | R@500 | R@1000 | Lift@50 | Lift@100 | Lift@500 | Lift@1000 | Acc@0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.00020 ± 0.00015 | 0.482 ± 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7681 |
| Random Forest | 0.00021 ± 0.00016 | 0.493 ± 0.150 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9081 |
| Baseline: random | 0.00025 ± 0.00030 | 0.408 ± 0.076 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| Baseline: rule 1 (neighbours, then address) | 0.00031 ± 0.00044 | 0.521 ± 0.094 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |
| Baseline: rule 2 (shared address) | 0.00031 ± 0.00044 | 0.521 ± 0.094 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | n/a |

**Does each model beat each baseline?**

- Logistic Regression vs Baseline: random: **does NOT clearly beat** -- lower or equal PR-AUC (0.00020 vs 0.00025); the same number of positives in the top 100 (0 vs 0)
- Logistic Regression vs Baseline: rule 1 (neighbours, then address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00020 vs 0.00031); the same number of positives in the top 100 (0 vs 0)
- Logistic Regression vs Baseline: rule 2 (shared address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00020 vs 0.00031); the same number of positives in the top 100 (0 vs 0)
- Random Forest vs Baseline: random: **does NOT clearly beat** -- lower or equal PR-AUC (0.00021 vs 0.00025); the same number of positives in the top 100 (0 vs 0)
- Random Forest vs Baseline: rule 1 (neighbours, then address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00021 vs 0.00031); the same number of positives in the top 100 (0 vs 0)
- Random Forest vs Baseline: rule 2 (shared address): **does NOT clearly beat** -- lower or equal PR-AUC (0.00021 vs 0.00031); the same number of positives in the top 100 (0 vs 0)

## Interpretation (main label)

| Feature | LR coefficient (scaled log1p) | RF permutation importance (drop in AP) |
|---|---|---|
| flagged_neighbour_companies | -0.019 ± 0.002 | +0.00000 ± 0.00000 |
| shared_address_count | +0.039 ± 0.090 | +0.00004 ± 0.00004 |
| degree | +0.430 ± 0.688 | +0.00007 ± 0.00009 |
| component_size | -0.291 ± 1.795 | -0.00012 ± 0.00024 |

A beats-verdict needs a PR-AUC gap larger than the fold-to-fold std AND at least as many positives in the top 100.
