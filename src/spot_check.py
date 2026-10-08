"""
Phase 1b: manual spot-check of name matches (docs/implementation_plan.md).

Two commands:

    python src/spot_check.py sample
        Reads data/processed/matches.csv and writes docs/spot_check_sample.csv:
        ALL strong matches (up to --n-strong, default 50 -> random if more) plus
        --n-weak (default 50) weak matches stratified by score (90-94.9 and
        95+, half each). Weak PEP rows dominate the population, so the weak
        sample is also split across list_source. `is_correct` is left empty:
        fill it by hand with y (same person/company), n (different) or ?
        (cannot tell).

    python src/spot_check.py score
        Reads the filled-in file and writes docs/match_precision.md:
        precision per tier and per list, with sample sizes. '?' and blank rows
        are excluded from precision and reported separately. Weak precision
        is also given as a population-weighted estimate, because the weak
        sample is stratified, not uniform.

To make hand-checking easier, the sample also shows the PSC birth year/month
(psc_clean.csv) and the list entry's birth_date (pep_clean / sanctions_clean).
random_state=42 throughout.
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERIM = os.path.join(ROOT, "data", "interim")
PROCESSED = os.path.join(ROOT, "data", "processed")
DOCS = os.path.join(ROOT, "docs")

MATCHES = os.path.join(PROCESSED, "matches.csv")
PSC = os.path.join(INTERIM, "psc_clean.csv")
PEP = os.path.join(INTERIM, "pep_clean.csv")
SANCTIONS = os.path.join(INTERIM, "sanctions_clean.csv")
SAMPLE = os.path.join(DOCS, "spot_check_sample.csv")
PRECISION = os.path.join(DOCS, "match_precision.md")

SEED = 42
WEAK_SPLIT_SCORE = 95.0

SAMPLE_COLS = [
    "company_number", "psc_name", "nationality", "psc_birth_year", "psc_birth_month",
    "list_name", "list_source", "list_birth_date", "tier", "score", "stratum",
    "stratum_size", "kind", "seed_source", "rule_used", "psc_row_id",
    "list_entity_id", "is_correct",
]


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def _take(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=SEED)


def draw_sample(matches: pd.DataFrame, n_strong: int = 50, n_weak: int = 50) -> pd.DataFrame:
    """Strong: all (or n_strong random). Weak: stratified by score band and list."""
    m = matches.copy()
    m["score"] = m["score"].astype(float)

    strong = m[m["tier"] == "strong"].copy()
    strong["stratum"] = "strong|" + strong["list_source"]
    strong_pop = strong.groupby("stratum").size().to_dict()
    strong_s = _take(strong, n_strong)

    weak = m[m["tier"] == "weak"].copy()
    band = (weak["score"] >= WEAK_SPLIT_SCORE).map({True: "95+", False: "90-94.9"})
    weak["stratum"] = "weak|" + weak["list_source"] + "|" + band
    weak_pop = weak.groupby("stratum").size().to_dict()

    # Spread n_weak evenly over non-empty strata; leftovers go to larger strata.
    strata = sorted(weak_pop)
    parts = []
    if strata:
        quota = {s: min(weak_pop[s], n_weak // len(strata)) for s in strata}
        spare = n_weak - sum(quota.values())
        for s in sorted(strata, key=lambda k: -weak_pop[k]):
            if spare <= 0:
                break
            add = min(spare, weak_pop[s] - quota[s])
            quota[s] += add
            spare -= add
        for s in strata:
            parts.append(_take(weak[weak["stratum"] == s], quota[s]))
    weak_s = pd.concat(parts) if parts else weak.iloc[0:0]

    out = pd.concat([strong_s, weak_s], ignore_index=True)
    pops = {**strong_pop, **weak_pop}
    out["stratum_size"] = out["stratum"].map(pops)
    out["is_correct"] = ""
    return out


def _attach_context(sample: pd.DataFrame, psc_path=PSC, pep_path=PEP,
                    sanctions_path=SANCTIONS) -> pd.DataFrame:
    """Add PSC birth year/month and list birth_date as read-only helper columns."""
    sample = sample.copy()
    sample["psc_birth_year"] = ""
    sample["psc_birth_month"] = ""
    sample["list_birth_date"] = ""

    if os.path.exists(psc_path):
        header = pd.read_csv(psc_path, nrows=0).columns
        cols = [c for c in ("birth_year", "birth_month") if c in header]
        if cols:
            psc = pd.read_csv(psc_path, dtype=str, keep_default_na=False, usecols=cols)
            ids = sample["psc_row_id"].astype(int)
            for c in cols:
                sample[f"psc_{c}"] = psc[c].reindex(ids).to_numpy()

    for source, path, id_col in (("pep", pep_path, "id"),
                                 ("sanctions", sanctions_path, "source_id")):
        if not os.path.exists(path):
            continue
        header = pd.read_csv(path, nrows=0).columns
        if "birth_date" not in header:
            continue
        lst = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=[id_col, "birth_date"])
        # sanctions has several seed rows per entity, all with the same birth_date
        lookup = lst.drop_duplicates(id_col).set_index(id_col)["birth_date"]
        mask = sample["list_source"] == source
        sample.loc[mask, "list_birth_date"] = sample.loc[mask, "list_entity_id"].map(lookup).fillna("")
    return sample


def cmd_sample(args):
    matches = pd.read_csv(args.matches, dtype=str, keep_default_na=False)
    if matches.empty:
        sys.exit("matches.csv is empty - run scripts/run_pipeline.py first.")
    sample = draw_sample(matches, args.n_strong, args.n_weak)
    sample = _attach_context(sample)
    # Strong first, then weak; within a tier keep rows from the same company together.
    sample["_t"] = (sample["tier"] != "strong").astype(int)
    sample = sample.sort_values(["_t", "list_source", "company_number"]).drop(columns="_t")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sample[SAMPLE_COLS].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(sample)} rows to {args.output}")
    print(sample.groupby(["tier", "list_source"]).size().to_string())
    print("\nFill the is_correct column with y / n / ?  (open with a text editor or "
          "import into Excel as TEXT - leading zeros in company_number must survive).")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def precision_table(sample: pd.DataFrame) -> pd.DataFrame:
    """Per (tier, list_source) and per tier: n sampled, y, n, ?, precision."""
    s = sample.copy()
    s["label"] = s["is_correct"].astype(str).str.strip().str.lower().replace({"yes": "y", "no": "n"})
    rows = []
    groups = [(t, l, g) for (t, l), g in s.groupby(["tier", "list_source"])]
    groups += [(t, "all", g) for t, g in s.groupby("tier")]
    for tier, source, g in groups:
        y, n = (g["label"] == "y").sum(), (g["label"] == "n").sum()
        unsure = (g["label"] == "?").sum()
        blank = len(g) - y - n - unsure
        rows.append({
            "tier": tier, "list": source, "sampled": len(g), "y": y, "n": n,
            "unsure": unsure, "unlabelled": blank,
            "precision": y / (y + n) if (y + n) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["tier", "list"], ascending=[False, True]).reset_index(drop=True)


def weighted_weak_precision(sample: pd.DataFrame):
    """Population-weighted precision for weak matches (strata sizes from the sample file)."""
    w = sample[sample["tier"] == "weak"].copy()
    w["label"] = w["is_correct"].astype(str).str.strip().str.lower()
    w = w[w["label"].isin(["y", "n"])]
    if w.empty:
        return None
    total, acc = 0.0, 0.0
    for _, g in w.groupby("stratum"):
        size = float(g["stratum_size"].iloc[0])
        acc += size * (g["label"] == "y").mean()
        total += size
    return acc / total if total else None


def _fmt(p):
    return "n/a" if p != p or p is None else f"{p:.0%}"


def write_report(sample: pd.DataFrame, path: str):
    t = precision_table(sample)
    lines = [
        "# Match precision (manual spot-check)", "",
        "Hand-labelled sample of name matches: `y` = same person/company, `n` = different, "
        "`?` = cannot tell. Precision = y / (y + n); `?` and blank rows are excluded.", "",
        "| Tier | List | Sampled | y | n | ? | Unlabelled | Precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in t.itertuples():
        lines.append(f"| {r.tier} | {r.list} | {r.sampled} | {r.y} | {r.n} | {r.unsure} | "
                     f"{r.unlabelled} | {_fmt(r.precision)} |")

    strong_all = t[(t.tier == "strong") & (t.list == "all")]
    if len(strong_all):
        r = strong_all.iloc[0]
        lines += ["", f"**Strong matches were correct {r.y} out of {r.y + r.n} judged times "
                      f"({_fmt(r.precision)}).**"]
        if r.sampled and r.unlabelled:
            lines += ["", f"Warning: {r.unlabelled} strong rows are still unlabelled."]
    wp = weighted_weak_precision(sample)
    if wp is not None:
        lines += ["", f"Weak matches, population-weighted over score bands and lists: ~{wp:.1%} "
                      "(the weak sample is stratified, so the raw weak row above is not "
                      "representative of all weak matches)."]
    lines += [
        "", "## Notes", "",
        "- Strong rows are a census (all strong matches are labelled) unless there were more "
        "than the sample cap, so strong precision has no sampling error.",
        "- Weak sample sizes are small; treat weak precision as indicative only.",
        "- The labels were judged against Companies House birth month/year and the list's "
        "birth date where available; `?` is used when neither side gives enough evidence.",
        "- If strong precision is below ~70%, tighten the strong rule before freezing labels.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return t


def cmd_score(args):
    sample = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "is_correct" not in sample.columns:
        sys.exit("No is_correct column in the sample file.")
    t = write_report(sample, args.output)
    print(t.to_string(index=False))
    print(f"\nWrote {args.output}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("sample", help="write docs/spot_check_sample.csv")
    p1.add_argument("--matches", default=MATCHES)
    p1.add_argument("--output", default=SAMPLE)
    p1.add_argument("--n-strong", type=int, default=50)
    p1.add_argument("--n-weak", type=int, default=50)
    p1.set_defaults(func=cmd_sample)

    p2 = sub.add_parser("score", help="write docs/match_precision.md from the labelled sample")
    p2.add_argument("--input", default=SAMPLE)
    p2.add_argument("--output", default=PRECISION)
    p2.set_defaults(func=cmd_score)

    a = ap.parse_args()
    a.func(a)
