# AML/KYC Graph Screening

A student project (FDS, Semester 7) that asks: **can the shape of a company's
ownership network predict whether it is linked to a sanctioned person or a
politically exposed person (PEP)?**

1. Take UK Companies House "persons with significant control" (PSC) data:
   who owns or controls each company.
2. Match those owners against the UK FCDO sanctions list and a UK PEP list.
3. Build a graph of owners and companies, and compute four graph features
   per company.
4. Compare Logistic Regression and Random Forest at ranking the risky
   companies, against simple baselines.

**Short answer: no, not with this data.** See [Results](#results).

## Folder layout

```
data/
  README.md          where each dataset came from, dates, row counts
  raw/               original downloads (git-ignored, not in this repo)
  interim/           cleaned CSVs: psc_clean, pep_clean, sanctions_clean (git-ignored)
  processed/         pipeline outputs: matches, graph, features, predictions (git-ignored)
src/
  name_normalization.py, text_encoding.py   name cleaning and encoding repair
  psc_loader.py, pep_cleaning.py, sanctions_loader.py   raw -> interim (preprocessing)
  matching.py        Phase 1: match owners to the PEP and sanctions lists
  spot_check.py      Phase 1b: sample matches for hand-checking, then score them
  graph_builder.py   Phase 2: build the ownership graph
  features.py        Phase 3: four features and the label per company
  train.py           Phase 4: Logistic Regression vs Random Forest vs baselines
  case_studies.py    Phase 5: explain individual results
scripts/
  run_preprocessing.py   raw -> interim (needs data/raw/, which is not available)
  run_pipeline.py        Phases 1-5 on data/interim/
tests/               pytest tests, one file per module
docs/
  implementation_plan.md   the phase plan and the decisions taken at each checkpoint
  preprocessing_log.md, graph_stats.md, feature_stats.md   per-phase statistics
  spot_check_sample.csv, match_precision.md                Phase 1b
  case_studies.md          Phase 5 write-up
  limitations.md           what the results can and cannot show
results/             metrics, charts and case-study drawings
```

## How to run

You need Python 3.10+ and the three cleaned files in `data/interim/`
(`psc_clean.csv`, `pep_clean.csv`, `sanctions_clean.csv`). They are not in
git; get them from a teammate (see `data/README.md`).

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py      # Phases 1-5, about 5 minutes
python -m pytest -q                 # tests
```

The pipeline writes `data/processed/` (matches, graph, features,
predictions), `results/` (metrics and charts) and the statistics files in
`docs/`. Every random step uses `random_state=42`, so reruns give identical
results.

**Manual spot-check (Phase 1b)** is separate, because it involves hand
labelling:

```bash
python src/spot_check.py sample     # writes docs/spot_check_sample.csv (refuses to overwrite labels)
# fill in the is_correct column by hand: y / n / ?
python src/spot_check.py score      # writes docs/match_precision.md
```

**Do not run `scripts/run_preprocessing.py`**: the raw files are not
available, and it would overwrite `data/interim/`.

## Results

Data: 1 of the 32 Companies House PSC files (499,971 owner records,
406,715 companies). Matching used **Mode A** (name + birth month and year).

| | |
|---|---|
| Strong matches to the lists | 52, on 45 owners |
| Risky companies (`label = 1`) | **52 of 406,715 (0.013%)**: 46 PEP, 6 sanctions |
| Ownership graph | 885,449 nodes, 499,142 edges, 387,944 connected groups |

Model comparison (5-fold cross-validation, grouped so linked companies never
sit in both training and test):

| Method | PR-AUC (mean ± std) | Risky companies in top 1,000 |
|---|---|---|
| Logistic Regression | 0.00019 ± 0.00007 | 0 |
| Random Forest | 0.00020 ± 0.00005 | 0 |
| Baseline: random | 0.00017 ± 0.00008 | 0 |
| Baseline: rank by shared addresses | 0.00032 ± 0.00045 | 0 |

**Neither model beats any baseline**, and no method puts a single risky
company in its top 1,000. A PEP-only run gives the same picture. The
case studies explain why:

- **Feature twins.** 56% of all companies have exactly the same four feature
  values (one owner, a two-node group, no shared address), and so do 22 of
  the 52 risky companies. A model that sees only these four numbers cannot
  tell them apart.
- **Random Forest memorises exact profiles.** Its top 8 companies are an
  unrelated corporate group (Travis Perkins subsidiaries) that has exactly
  the same profile as one flagged PEP's 8 companies (Peter Caruana).
- **The network feature is near-constant.** "Flagged neighbour companies"
  is non-zero for only 7 companies, none of them risky.

Details: `results/metrics.md`, `docs/case_studies.md`, `docs/limitations.md`.

**Still pending:** the manual spot-check (Phase 1b). Until it is labelled we
do not know what share of the 52 strong matches are really the listed person.
