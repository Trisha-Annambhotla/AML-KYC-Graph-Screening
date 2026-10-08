# Limitations

What the results of this project can and cannot show. Numbers are from the
final pipeline run (2026-10-08); sources are given for each.

## Data

1. **Only 1 of 32 PSC files was used.** 499,971 owner records and 406,715
   companies, about 1/32 of the register. This was a deliberate sample, but
   it cuts most ownership links: a person who controls companies in
   different files appears as unrelated owners, so the graph is far sparser
   than the real one (79% of connected groups are just one company and one
   owner; `docs/graph_stats.md`). Every count here is a sample figure, not a
   full-register figure.
2. **Single point-in-time snapshot.** PSC snapshot of 2026-08-02; PEP and
   sanctions lists downloaded 2026-08-02. Ownership and list membership
   change over time; nothing here describes any other date.
3. **Service addresses, not registered offices.** The address used for the
   address features is the PSC's correspondence address, not the company's
   registered office. Many PSCs use an accountant's or a virtual office's
   address, so shared addresses often reflect a shared service provider,
   not a real connection. Addresses shared by more than 50 companies are
   therefore ignored (206 keys covering 32,234 companies; the cap of 50 is a
   judgement call; `docs/feature_stats.md`).
4. **Corporate owners are not linked to their own company records.** A
   corporate PSC ("X Holdings Ltd") is a node identified by its name; it is
   not joined to the company node for X Holdings Ltd, because the PSC data
   does not give its company number. Corporate ownership chains are broken.
5. **People are identified by name + postcode.** The same person at two
   postcodes becomes two nodes; two different people with the same name at
   the same postcode become one.

## Labels

6. **Sanctions/PEP status is a stand-in for risk, not proof of
   wrongdoing.** Being a PEP is not an offence, and a company with a PEP
   owner is not necessarily doing anything wrong. The label means "linked to
   a listed person", nothing more.
7. **Strong-match rule (Mode A) and its precision.** A PSC is a strong match
   when the name scores >= 90 (rapidfuzz token-sort ratio) and the birth
   month and year agree with the list entry. Companies House publishes only
   month and year, so two different people with similar names born in the
   same month can still match. Company owners have no birth date; for them
   a strong match is an exact normalised name of at least two real words.
   **The precision of these labels is not yet known**: the manual
   spot-check (Phase 1b, `docs/spot_check_sample.csv`, 100 matches) is still
   unlabelled, so `docs/match_precision.md` has no figures. If many of the
   52 strong matches are wrong, the label itself is noisy.
8. **Very few positives.** 52 risky companies out of 406,715 (0.013%): 46
   PEP, 6 sanctions. The 52 sit in 44 independent groups, about 10 per test
   fold. Every metric is therefore very uncertain (e.g. for the rule
   baselines the PR-AUC std across folds is larger than the mean), and 6
   sanctions positives were too few for a sanctions-only run.
9. **Weak matches are not used as labels.** `label_weak` (4,288 companies)
   is reported only: in Mode A most weak matches are different people with
   the same name.

## Features and models

10. **The "flagged 1-hop neighbour" feature was redefined to avoid
    leakage.** The scope's version (does the company have a flagged
    neighbour?) is the label itself. Feature 1 instead counts *other* risky
    companies reachable through a non-flagged co-owner or a shared (capped)
    address, with every flagged owner removed first.
11. **Feature 1 is near-constant.** It is non-zero for only 7 of 406,715
    companies, none of them risky (`docs/feature_stats.md`). It stayed in
    the models because it is in the scope, but it carries almost no
    information. It does not even fire around the one flagged owner with
    several companies (Peter Caruana): his companies have no other owner,
    so once he is removed there is nothing to link through
    (`docs/case_studies.md`, case 4).
12. **Feature twins: the four features cannot separate risky companies.**
    56% of all companies (229,197) have exactly the same four values (one
    owner, a two-node group, no shared address), including 22 of the 52
    risky companies. Any model that sees only these four numbers must give
    them all the same score. Even the best-ranked risky company has 4,917
    exact twins, of which only 3 are risky (`docs/case_studies.md`, cases
    1 and 3).
13. **Random Forest memorises exact profiles.** Its 8 highest-scoring
    companies are subsidiaries of one corporate owner (Travis Perkins
    Financing Company No.3 Limited) that have exactly the same profile as
    Peter Caruana's 8 risky companies, which were in the training folds at
    the time. In the other direction, Caruana's own companies rank around
    382,000 of 406,715, because when they were scored the training data held
    only the 8 non-risky Travis Perkins twins. The forest learns one group's
    shape, not a general pattern (`docs/case_studies.md`, cases 2 and 4).
    The grouped cross-validation is what exposes this: with an ordinary
    random split, Caruana's companies would likely have been in training
    and test at the same time and scored each other highly.
14. **Neither model beats the baselines.** PR-AUC: Logistic Regression
    0.00019, Random Forest 0.00020, random 0.00017, shared-address rule
    0.00032, all within fold-to-fold noise of each other. No method puts a
    single risky company in its top 1,000 (`results/metrics.md`). Accuracy
    (0.76 and 0.91) only reflects how rare positives are and is not
    evidence of skill.
15. **The conclusion is about this sample and these four features.** It
    does not show that ownership networks carry no AML signal; it shows
    that four coarse graph features on 1/32 of the register, with 52
    positives, do not. A full-register graph, linked corporate chains,
    registered-office addresses and richer features (e.g. officer data,
    company age, jurisdiction) would be needed to test the idea properly.
