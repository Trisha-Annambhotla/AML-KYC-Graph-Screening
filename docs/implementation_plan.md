Below is the full plan. Save it in your repo as **`docs/implementation_plan.md`**. Then, for each phase, tell Claude Code: *"Implement Phase N of docs/implementation_plan.md. Follow the rules in the 'Rules for every phase' section."* Do one phase per session, and paste each checkpoint result here if you want a second opinion.

---

# Implementation Plan: AML/KYC Graph Screening

## Current state

* Preprocessing is done. The three cleaned files are in `data/interim/`:
  * `psc_clean.csv`: 499,971 rows, from 1 of 32 PSC parts
  * `pep_clean.csv`: 7,990 rows (bishop and archbishop title rows removed)
  * `sanctions_clean.csv`: 19,610 rows (one row per name and per alias)
* Raw data is not available. Never run `scripts/run_preprocessing.py`, because it would overwrite the interim files and the log.
* **Problem 2 is open.** `psc_clean.csv` has no birth month or year, and `sanctions_clean.csv` has no `birth_date` or `addresses`. A teammate may deliver new versions of these two files later. **Every phase must work with both the current files and the future ones**, choosing automatically based on which columns exist.

## Rules for every phase

1. Never read a full CSV into the conversation. Inspect files with pandas `nrows=` or `head`.
2. Never modify anything in `data/interim/`.
3. Large outputs go in `data/processed/`, which git ignores. Small results (metrics CSVs, charts, markdown summaries) go in `results/` or `docs/` and are committed.
4. Each phase is one module in `src/`, with tests in `tests/`. Also add the phase as a step in `scripts/run_pipeline.py`, so the whole pipeline (Phases 1–4) can be rerun with one command when new data arrives.
5. Set `random_state=42` everywhere randomness is used.
6. At every **checkpoint**, show the summary and wait for approval before continuing.
7. At the end of every phase, run all tests, then run:
   `PYTHONUTF8=1 gitingest . -e "data/" -e "digest.txt" -o digest.txt`
   Then list the changed files so I can commit them. Don't commit automatically.

---

## Phase 1: Name matching

**Goal:** Find which company owners (PSCs) appear on the PEP or sanctions lists.

**Build:** `src/matching.py`, with output `data/processed/matches.csv`.

1. **Give every PSC row an ID.** Use `psc_row_id`, the row's position in `psc_clean.csv`. Later phases use it to find that owner in the graph.
2. **Compare like with like:**
   * PSC rows whose `kind` starts with `individual` are compared with PEP rows and sanctions rows with schema `Person`.
   * PSC rows whose `kind` starts with `corporate` are compared with sanctions rows with schema `Organization`, `LegalEntity` or `Company`.
   * Ignore other PSC kinds, such as legal-person or super-secure entries, and log how many were skipped.
3. **Speed:**
   * First do an exact-match pass on `normalized_name` using a dictionary lookup.
   * Then do a fuzzy pass with blocking. For people, only compare names with the same last word (surname). For companies, only compare names sharing their least common word.
4. **Scoring:** use rapidfuzz `token_sort_ratio` on `normalized_name`, and keep candidates scoring at least 90.
5. **Strong vs weak.** Detect which mode applies from the columns present.
   * **Mode A (birth dates available):** the PSC file has `birth_month`/`birth_year` and the list row has `birth_date`. Strong means score ≥ 90 and the same birth year and month. Sanctions `birth_date` may hold several dates separated by `;`, and any one matching counts.
   * **Mode B (current files, fallback rule):**
     * *People:* strong means an exact normalised name match, at least 3 name words, and a nationality consistent with the list's `countries`. Use a small mapping, such as British → gb, Russian → ru, Iranian → ir. An unknown or missing nationality is never strong.
     * *Companies:* strong means an exact normalised name match.
   * Everything else scoring ≥ 90 is weak.
   * Record in a `rule_used` column whether Mode A or B produced each label.
6. **Output columns:** `psc_row_id`, `company_number`, `kind`, `psc_name`, `nationality`, `list_name`, `list_source` (pep or sanctions), `seed_source` (canonical or alias, for sanctions), `list_entity_id`, `score`, `tier`, `rule_used`.
7. **Tests:**
   * an exact match is strong in Mode B
   * a two-word name is never strong in Mode B
   * a person is never matched to an organisation
   * Mode A is chosen when birth columns exist

**Checkpoint 1:** show the count of strong and weak matches per list, the number of **distinct companies** with at least one strong match, and 10 example matches per tier.

**Gate:** if fewer than about 30 companies have a strong match, stop. Present these options:
* treat sanctions and PEP as one combined label (the default)
* add a sensitivity run that includes weak matches scoring at least 95
* report the project explicitly as a small pilot

## Phase 1b: Manual spot-check

**Goal:** Measure how often the matches are actually correct. This is evidence the scope document requires.

**Build:** `src/spot_check.py`, which has two commands:

1. **`sample`:** writes `docs/spot_check_sample.csv`. It takes 50 random strong and 50 random weak matches (or all of them, if there are fewer), with an empty `is_correct` column. *We fill in this column by hand, writing y or n.*
2. **`score`:** reads the filled-in file and writes `docs/match_precision.md`, with estimated precision per tier and per list, and the sample sizes.

**Done when:** `match_precision.md` exists. The human labelling is our job, not Claude Code's.

## Phase 2: Build the ownership graph

**Goal:** Turn the PSC rows into a network of who controls what.

**Build:** `src/graph_builder.py`. Outputs:
* `data/processed/graph.pkl`, saved with pickle, since `write_gpickle` was removed in networkx 3
* `data/processed/nodes.csv` and `data/processed/edges.csv`

1. **Nodes:**
   * Companies: `company:<company_number>`.
   * Individual PSCs: `person:<normalized_name>|<postcode>`. Extract the UK postcode from `address` with a regex; if none is found, use the normalised full address. This merges the same person across the companies they control.
   * Corporate PSCs: `corp:<normalized_name>`.
2. **Edges:** from each PSC node to its company node, with `natures_of_control` as an attribute.
3. **Flags:**
   * Map every strong match to its node through `psc_row_id`, and set `flagged=True` and `flag_source` (pep, sanctions or both).
   * Keep weak flags separately in a `weak_flagged` attribute.
4. **Address key:**
   * Store each PSC's normalised address key: postcode plus the first address line, lowercased with spaces collapsed.
   * Don't add address nodes to the graph.
5. **Tests:**
   * the same person controlling two companies becomes one node with two edges
   * a flagged match lands on the right node

**Checkpoint 2:** write `docs/graph_stats.md` containing:
* node counts by type, and the edge count
* the number of connected components and the size distribution (min, median, max, and the top 10 sizes)
* the number of flagged nodes
* confirmation that **every** strong match was mapped to a node, or a list of those that weren't

## Phase 3: Features and labels

**Goal:** One row per company, with its 4 features and a label.

**Build:** `src/features.py`, with output `data/processed/features.csv`.

1. **Label:** `label = 1` if the company is directly connected to at least one strongly flagged PSC.
   * Also store `label_weak`, using weak flags as well, for sensitivity analysis.
   * Also store `label_source`: pep, sanctions or both.
2. **Feature 1, `flagged_neighbour_companies`.** This feature must **not** contain the label.
   * Make a copy of the graph with all flagged PSC nodes removed.
   * For each company, count the *other* companies that have `label = 1` and share a (non-flagged) PSC with it.
   * Never count the company's own flagged owners. *(This replaces the scope's "flagged 1-hop neighbours", which would equal the label and leak the answer. Document this in the report.)*
3. **Feature 2, `shared_address_count`:** the number of *other* companies with at least one PSC at the same address key.
4. **Feature 3, `degree`:** the number of PSCs connected to the company, on the full graph.
5. **Feature 4, `component_size`:** the size of the company's connected component in the full graph.
6. **Extra columns** (not features): `company_number` and `component_id`, the full-graph component ID used for the train/test split.
7. **Tests:**
   * a company whose only link to risk is its own flagged owner gets `flagged_neighbour_companies = 0`
   * shared address counts exclude the company itself

**Checkpoint 3:** show the number of companies, the number and percentage of positives, summary statistics for each feature split by label, and a check that no feature is perfectly correlated with the label.

## Phase 4: Train and compare the models

**Goal:** Logistic Regression vs Random Forest, judged honestly.

**Build:** `src/train.py`. Outputs:
* `results/metrics.csv`
* `results/pr_curves.png`
* `results/feature_importance.png`
* `data/processed/predictions.csv` (out-of-fold scores)

*(Updated 2026-10-08 after Checkpoint 3: split on `split_group_id`, no `label_weak` run, PEP-only sensitivity only, extra k values and lift, second rule baseline.)*

1. **Split:** `GroupKFold` with 5 folds, grouped by **`split_group_id`** (not `component_id`). `split_group_id` is the ownership-graph component merged with companies that share a (capped) address key, so neither owner-linked nor address-linked companies end up in both the training and test sets.
2. **Logistic Regression:**
   * apply `log1p` to all four features, then `StandardScaler`
   * `class_weight="balanced"`, `max_iter=1000`
3. **Random Forest:**
   * raw features
   * `n_estimators=300`, `min_samples_leaf=5`, `class_weight="balanced_subsample"`
4. **Features:** the four from Phase 3. Keep `flagged_neighbour_companies` as a model feature (it is in the scope), but note in the results that it is near-constant (non-zero for only 7 companies, none positive).
5. **Baselines:**
   * (a) random scores
   * (b) rule 1: rank by `flagged_neighbour_companies`, then by `shared_address_count`
   * (c) rule 2: rank by `shared_address_count` alone (added because feature 1 is almost always 0)
6. **Metrics:**
   * **Main:** PR-AUC (`average_precision_score`), reported as the mean ± std across folds.
   * **On the pooled out-of-fold scores**, for k = 50, 100, 500 and 1000: precision@k, recall@k, and lift over random (precision@k divided by the positive rate).
   * **Secondary:** ROC-AUC.
   * **Context only:** accuracy at a 0.5 threshold.
   * There are fewer positives than k, so report k alongside the positive count.
7. **Interpretation:** report the Logistic Regression coefficients and the Random Forest feature importances (permutation importance on held-out folds).
8. **Sensitivity run** (add rows to `metrics.csv`):
   * **PEP-only label** (46 positives).
   * No sanctions-only run (only 6 positives).
   * **No `label_weak` run.** `label_weak` is for reporting only and is never a training label, because in Mode A most weak matches are different people with the same name.

**Checkpoint 4:** show the metrics table and state plainly whether each model beats each baseline.

## Phase 5: Case studies

**Goal:** Explain individual results; the scope requires explainability.

**Build:** `src/case_studies.py`, with outputs `results/case_*.png` and `docs/case_studies.md`.

1. Pick 4 companies from the out-of-fold predictions:
   * 2 high-scoring true positives
   * 1 high-scoring false positive
   * 1 positive the models missed
2. For each one, draw the 2-hop neighbourhood. Colour companies, people and flagged nodes differently, and label each node.
3. In `case_studies.md`, write each company's feature values, the model scores, and a short plain-English explanation of why it scored that way.

## Phase 6: Documentation and report material

1. **`README.md`:** what the project does, the folder layout, how to run the pipeline (`python scripts/run_pipeline.py`), and a summary of results.
2. **`data/README.md`:**
   * fill in the PEP counts: 769,044 before filtering, 8,009 after the gb filter, and 7,990 after removing titles
   * note that the work used interim files from a teammate, which is why the hashes are blank
3. **`docs/limitations.md`:**
   * only 1 of 32 PSC parts was used
   * sanctions and PEP status is a stand-in for risk, not proof of wrongdoing
   * the strong-match rule used (Mode A or B) and its estimated precision
   * the PSC service address is used rather than the company's registered address
   * corporate PSCs can't be linked to their own company records
   * the data is a single point-in-time snapshot
   * the "flagged 1-hop" feature was redefined to avoid leakage

## When the Problem 2 files arrive

1. Replace `psc_clean.csv` and `sanctions_clean.csv` in `data/interim/`.
2. Check the new columns exist: `birth_month` and `birth_year` in PSC, and `birth_date` and `addresses` in sanctions.
3. Run `python scripts/run_pipeline.py`. Matching switches to Mode A automatically.
4. Redo the Phase 1b spot-check on a new sample, since the strong matches will have changed.
5. Update the results and limitations to state that Mode A was used.

---
