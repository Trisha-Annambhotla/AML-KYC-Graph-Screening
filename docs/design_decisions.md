# Design decisions

Every parameter below was chosen or changed at a checkpoint, based on what
the pilot data showed. "Evidence" is the observation that drove the choice;
"Checkpoint" says when it was decided (the phases are in
[implementation_plan.md](implementation_plan.md)). Where a value was set in
the plan and not tuned, the table says so.

## Main decisions

| # | Decision | Value chosen | Evidence | Checkpoint |
|---|---|---|---|---|
| 1 | **Fuzzy match score** | rapidfuzz `token_sort_ratio` >= 90 on normalised names | Set in the plan, **not tuned**. It only decides which pairs are *candidates*; whether a candidate counts as a label is decided by the strong-match rule (rows 2-4). In the final (Mode A) run, 1,773 of the 4,845 weak matches are fuzzy (score < 100), e.g. "Ian Wright" vs "Iain Wright" (95); 2 of the 52 strong matches are fuzzy too (scores 90.3 and 95.2), accepted because the birth month and year agree. | Plan, Phase 1 |
| 2 | **Strong-match rule: Mode A vs Mode B** | **Mode A** (name score >= 90 + same birth month and year) when both files have birth columns; otherwise Mode B (exact name + >= 3 words + nationality consistent with the list). Chosen automatically per list. | The first interim files had no birth dates, so Mode B was the only option: it gave 17 companies. After the PSC and sanctions loaders kept birth month/year, Mode A gave **52 companies** (46 PEP, 6 sanctions). Birth dates are much stronger evidence of "same person" than a name alone. | Checkpoint 1; Mode A active after the 2026-10-08 re-processing |
| 3 | **Apostrophe rule** (Mode B) | Count name words with apostrophes removed | Found in the Mode B run: "Paul O'Neill" normalises to "paul o neill" = 3 words, so it passed the 3-word rule. **9 companies** (O'Neill, O'Brien, O'Toole) were strong only because of this, all very common names. | Checkpoint 1 |
| 4 | **Short-alias rule** (companies) | A strong company match needs >= 2 real words (tokens of 2+ characters) | **8 of the 10** strong company matches were 1-3 letter or single-word sanctions aliases ("CP", "IAP", "SIG", "VSK", "NALU", "AMANA", "B&H") matching unrelated UK companies such as "Sig Plc". Only "Nord Gold Plc" looked genuine. With rules 3 and 4, Mode B strong companies fell from 35 to 17. | Checkpoint 1 |
| 5 | **Gate of ~30 companies** | Stop and choose a fallback if fewer than ~30 companies have a strong match | Set in the plan as the minimum for a meaningful model comparison. Mode B gave **17** (below the gate): the project was declared a pilot, with the combined PEP + sanctions label. Mode A later gave **52**, above the gate. | Plan; triggered at Checkpoint 1 (Mode B) |
| 6 | **Rejected sensitivity label "weak >= 95"** | Not used | A weak-match breakdown (Mode B run) showed **2,739** weak matches fail only because the name has two words: exact common names like "David Williams", mostly different people. A score-95 rule would have added thousands of these as positives. | Phase 1 gate decision |
| 7 | **Address cap** | Ignore any address key shared by **more than 50 companies** (in feature 1, `shared_address_count` and `split_group_id`) | Before the cap, one positive's co-owners at the virtual office **71-75 Shelton Street, WC2H 9JQ** (~3,000 companies) gave **2,841** companies feature 1 = 1, none of them risky. The cap removes **206** keys (32,234 companies), the largest being virtual offices: 20-22 Wenlock Road (3,454), Shelton Street (2,835), 27 Old Gloucester Street (1,276). After it, feature 1 is non-zero for 7 companies. 50 is a judgement call, not tuned. | Checkpoint 3 |
| 8 | **`split_group_id` for cross-validation** | 5-fold `GroupKFold` grouped by ownership component **merged with companies sharing a (capped) address key** | Companies linked only through an address sit in different ownership components, so a component-based split could put them in train and test at once. Merging gives **305,990** groups (from 387,944 components); the 52 positives sit in 44 groups. Case studies later confirmed the value of grouping: Peter Caruana's 8 companies always share a test fold, so they cannot vouch for each other. | Checkpoint 3 |
| 9 | **`label_weak` is report-only** | Never a training label; no `label_weak` sensitivity run | 4,288 companies have a strong or weak match (`label_weak`), against 52 with a strong one. In Mode A a weak match means the name is similar but the birth month/year do not agree (or are missing): mostly **different people** with the same name. | Checkpoint 3 |
| 10 | **Network view limits** (dashboard) | At most **2 hops** from the selected company, at most **150 nodes**, closest and flagged nodes first; the page says when the view is cut | The largest ownership component has **414** nodes and one owner (Peter Valaitis) controls **227** companies: 2 hops around one of his companies already reaches 228 nodes, too many to read. 2 hops shows a company's owners and their other companies, which is what an investigator checks first. The limits are for readability, not from model evidence. | Dashboard build |

## Other decisions

| Decision | Value chosen | Evidence | Checkpoint |
|---|---|---|---|
| PEP job-title rows | Remove names containing "bishop of" / "archbishop" (19 rows); do **not** use "rt rev" | "The Rt Rev. the Lord Bishop of St. Albans" is a job title, not a person. "rt rev" would also remove "The Rt Rev. the Lord Harries of Pentregarth", a real person. | Data fix before Phase 1 |
| Person node identity | `normalised name + postcode` | Name + nationality would merge every "John Smith" (British) into one fake hub. Name + postcode merges the same person across companies (**8,627** people control 2+ companies) while keeping namesakes apart. | Plan, Phase 2 |
| Feature 1 leakage rule | Flagged owners removed on **both** sides of owner and address links | The scope's "flagged 1-hop neighbour" equals the label. Removing only a company's own flagged owner would still let it count a sister company through that same owner's address. | Checkpoint 3 |
| Feature 1 stays a model feature | Kept, and reported as near-constant | It is in the project scope; it is non-zero for only 7 companies, none risky. | Phase 4 brief |
| Second rule baseline | Rank by `shared_address_count` alone | Feature 1 is almost always 0, so rule 1 (feature 1, then shared address) ranks almost exactly like rule 2; both are shown. | Phase 4 brief |
| Sensitivity runs | PEP-only (46 positives); no sanctions-only run | Only 6 sanctions positives, about 1 per test fold: too few to train on. | Phase 4 brief |
| `shortest_distance_to_flagged` | Stored, **not** added to the models | All 52 positives are unreachable from any other flagged owner (value -1); only 5 companies in the dataset can reach one, none risky. | Phase 7 |
| Hub component | Owner controlling **100+** companies | Marks formation-agent clusters such as the Valaitis group (227 companies); 1,490 companies, no positives. Analysis only. | Checkpoint 3 |
