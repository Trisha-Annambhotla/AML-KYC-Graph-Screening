# Preprocessing Log — 2026-09-09 12:53

## Sanctions (gb_fcdo_sanctions)

- canonical_names: 6261
- aliases: 13349
- total_seed_rows: 19610

## PEP dataset

- total_rows: 769044
- kept_gb: 8009
- dropped_not_gb: 760688
- dropped_positional_title: 347
- dropped_empty_name: 0

## PSC snapshot

- parts_processed: 1
- records_written: 499971
- records_skipped_no_name: 0
- parts_found: 1
- parts_expected: 32
- **PARTIAL COVERAGE**: only 1/32 PSC parts processed. Any downstream match counts are a partial-coverage pilot result, not a full-register figure (Charter Section 1.1, Step 4).

## Post-fix: PEP positional titles (2026-10-08)

- Removed 19 more rows from `pep_clean.csv` whose name contains "bishop of"
  or "archbishop" (e.g. "The Rt Rev. the Lord Bishop of St. Albans"). These
  are job titles, not personal names; the start-anchored regex missed them
  because "The Rt Rev." comes first.
- "rt rev" was not used as a pattern, so "The Rt Rev. the Lord Harries of
  Pentregarth" (a real person) is kept.
- Removed 2 title-only aliases: "The Rt Rev. and the Rt Hon. Lord Chartres
  GCVO" (Richard Chartres) and "The Rt Rev. and the Rt Hon. Lord Sentamu"
  (John Sentamu). Kept "Archbishop Paul Richard Gallagher", which contains
  his real name.
- PEP rows: 8,009 -> 7,990. Original kept as `pep_clean_backup.csv`.
