# Graph statistics (Phase 2, Checkpoint 2)

Generated 2026-10-08 by `src/graph_builder.py` from `data/interim/psc_clean.csv` and `data/processed/matches.csv`.

## Size

| Node type | Count |
|---|---|
| company | 406,715 |
| person | 448,110 |
| corp | 30,624 |
| **total nodes** | **885,449** |
| **edges** (PSC -> company) | **499,142** |

- People who control 2+ companies (merged by name + postcode): 8,627
- Person nodes with no postcode (keyed by full address instead): 10,059

## Connected components (weakly connected)

- Number of components: 387,944
- Size (nodes): min 2, median 2, max 414
- Top 10 sizes: 414, 269, 261, 226, 178, 144, 135, 134, 114, 108
- Components by size (smallest sizes): 2 nodes: 304,759, 3 nodes: 69,079, 4 nodes: 10,061, 5 nodes: 2,314, 6 nodes: 778, 7 nodes: 344

## Flags

- Strongly flagged PSC nodes: 45 (pep: 39, sanctions: 6)
- Weakly flagged PSC nodes: 4,170

## Strong-match mapping check

All 52 strong matches were mapped to a node.
