# Feature statistics (Phase 3, Checkpoint 3)

Generated 2026-10-08 by `src/features.py` from `data/processed/graph.pkl`.

## Companies and labels

- Companies: 406,715 in 387,944 components
- Positives (`label = 1`): 52 (0.0128%)
- By source: pep: 46, sanctions: 6
- `label_weak = 1` (strong OR weak flags): 4,288. **Reporting only, never a training label**: in Mode A most weak matches are different people with the same name.
- Companies in a hub component (PSC with 100+ companies): 1,490, of which positive: 0

## Address-key cap

- Address keys shared by more than **50 companies** are ignored in `flagged_neighbour_companies`, `shared_address_count` and `split_group_id`. They are virtual offices, formation agents and accountants, not real links.
- Capped keys: 206 of 304,153, covering 32,234 companies
- Largest capped keys: `n1 7gu | 20-22 wenlock road` (3,454); `wc2h 9jq | 71-75 shelton street` (2,835); `wc1n 3ax | 27 old gloucester street` (1,276); `ec1a 9ej | 1st floor` (1,177); `ec1v 2nx | kemp house` (1,118)

## Train/test split groups

- `split_group_id` = ownership-graph components merged by shared (capped) address keys: 305,990 groups (vs 387,944 components)
- Largest group: 619 companies; the 52 positives sit in 44 groups
- Phase 4 must use `split_group_id` (not `component_id`) for GroupKFold, so address-linked companies never land in different folds.

## Feature 1: companies with a flagged neighbour > 0

| Version | Companies > 0 | Positives > 0 |
|---|---|---|
| `flagged_neighbour_companies` (owner OR address) | 7 | 0 |
| `flagged_neighbour_via_owner` (plan's original) | 4 | 0 |

## `shortest_distance_to_flagged` (not a model feature)

Hops on the ownership graph from each company to the nearest strongly flagged owner, ignoring the company's own flagged owners (same rule as feature 1); capped at 6, -1 if none is reachable. Flagged nodes are owners and paths alternate company/owner, so only odd distances (3, 5) occur below the cap. Stored in `features.csv` for analysis; not used by the models.

| Distance | label 0 (companies, %) | label 1 (companies, %) |
|---|---|---|
| unreachable (-1) | 406,658 (99.999%) | 52 (100.0%) |
| 3 | 4 (0.001%) | 0 (0.0%) |
| 5 | 1 (0.000%) | 0 (0.0%) |

## Feature summary by label

| Feature | label | mean | std | min | 50% | 90% | 99% | max |
|---|---|---|---|---|---|---|---|---|
| flagged_neighbour_companies | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flagged_neighbour_companies | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| shared_address_count | 0 | 2.99 | 7.69 | 0 | 0 | 10 | 40 | 85 |
| shared_address_count | 1 | 5.52 | 10.2 | 0 | 0 | 17 | 38.96 | 41 |
| degree | 0 | 1.23 | 0.52 | 1 | 1 | 2 | 3 | 20 |
| degree | 1 | 1.48 | 0.94 | 1 | 1 | 2 | 4.98 | 6 |
| component_size | 0 | 3.59 | 17.23 | 2 | 2 | 3 | 15 | 414 |
| component_size | 1 | 3.69 | 2.59 | 2 | 2 | 9 | 9 | 9 |
| flagged_neighbour_via_owner | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| flagged_neighbour_via_owner | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Correlation with label (leakage check)

| Feature | Pearson | Spearman |
|---|---|---|
| flagged_neighbour_companies | -0.0000 | -0.0000 |
| shared_address_count | 0.0037 | 0.0020 |
| degree | 0.0055 | 0.0035 |
| component_size | 0.0001 | 0.0067 |
| flagged_neighbour_via_owner | -0.0000 | -0.0000 |

No feature is perfectly correlated with the label (all |r| < 0.99).
