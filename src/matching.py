"""
Phase 1: name matching (docs/implementation_plan.md).

Finds PSC rows (company owners) whose name matches a PEP or sanctions list
entry, and labels each match STRONG or WEAK.

Like is compared with like:
    - individual-* PSCs  vs  PEP rows and sanctions rows with schema Person
    - corporate-* PSCs   vs  sanctions rows with schema Organization /
                             LegalEntity / Company
    - every other PSC kind (legal-person-*, super-secure, ...) is skipped
      and counted.

Candidates come from two passes over `normalized_name`:
    1. exact pass  -- dictionary lookup
    2. fuzzy pass  -- rapidfuzz token_sort_ratio >= 90, with blocking:
         people:    the PSC surname (last word) must appear in the list name.
                    Not "same last word", because list names come in both
                    orders ("Ivan ... TYRYSHKIN" and the Cyrillic alias
                    "BULAVKO Anatolii ...").
         companies: the list name must contain the PSC name's least common
                    word (rarest across PSC corporate + list company names).

Strong vs weak (rule_used records which mode labelled the row):
    Mode A -- PSC has birth_month/birth_year AND the list has birth_date:
        person strong = score >= 90 and same birth year and month (any one of
        the list's ';'-separated dates may match).
    Mode B -- fallback, chosen per list when Mode A's columns are missing:
        person strong = exact name match, >= 3 name words, and a PSC
        nationality consistent with the list row's countries. Words are
        counted with apostrophes removed first, so "Paul O'Neill" is 2 words,
        not 3 ("paul o neill" after normalization).
    Companies (both modes, recorded as B): strong = exact name match AND
        >= 2 real words (tokens of 2+ characters, legal suffixes already
        removed), so short aliases like "CP", "SIG" or "B&H" stay weak.
    Everything else scoring >= 90 is weak.

Choices not spelled out in the plan (agreed at Checkpoint 1):
    - Person blocking: the PSC surname may appear anywhere in the list name,
      not only as its last word (see above).
    - PEP: only the main `name` is matched, not PEP `aliases` (mostly
      "Surname, L." forms).
    - Aliases are deduplicated: each output row is one (PSC row, list
      entity) pair. If several aliases of the same sanctioned entity match,
      the best one is kept (strong over weak, then higher score, then
      canonical over alias).

Run directly:
    python src/matching.py
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz, process

from name_normalization import normalize_name

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERIM = os.path.join(ROOT, "data", "interim")
PROCESSED = os.path.join(ROOT, "data", "processed")

PSC_INPUT = os.path.join(INTERIM, "psc_clean.csv")
PEP_INPUT = os.path.join(INTERIM, "pep_clean.csv")
SANCTIONS_INPUT = os.path.join(INTERIM, "sanctions_clean.csv")
MATCHES_OUTPUT = os.path.join(PROCESSED, "matches.csv")

SCORE_CUTOFF = 90
MIN_STRONG_PERSON_TOKENS = 3
MIN_STRONG_COMPANY_TOKENS = 2
MIN_REAL_TOKEN_LEN = 2

PERSON_SCHEMAS = {"Person"}
COMPANY_SCHEMAS = {"Organization", "LegalEntity", "Company"}

PSC_BIRTH_COLS = {"birth_month", "birth_year"}
LIST_BIRTH_COL = "birth_date"

PSC_BASE_COLS = ["company_number", "kind", "name", "normalized_name", "nationality"]

OUTPUT_FIELDS = [
    "psc_row_id", "company_number", "kind", "psc_name", "nationality",
    "list_name", "list_source", "seed_source", "list_entity_id", "score",
    "tier", "rule_used",
]

# PSC nationality (free text demonym) -> ISO-2 code used by the lists'
# `countries` field. Covers the common PSC nationalities and the countries
# that dominate the sanctions list. Anything not here is "unknown", which is
# never strong under Mode B.
DEMONYM_TO_ISO = {
    # UK and its nations
    "british": "gb", "english": "gb", "scottish": "gb", "welsh": "gb",
    "northern irish": "gb", "united kingdom": "gb", "uk": "gb",
    "cymro": "gb", "cymraes": "gb", "british citizen": "gb",
    # main sanctions countries
    "russian": "ru", "syrian": "sy", "iranian": "ir", "ukrainian": "ua",
    "belarusian": "by", "belarussian": "by", "afghan": "af", "afghani": "af",
    "north korean": "kp", "iraqi": "iq", "pakistani": "pk", "chinese": "cn",
    "emirati": "ae", "indonesian": "id", "panamanian": "pa", "libyan": "ly",
    "saudi": "sa", "saudi arabian": "sa", "congolese": "cd", "sudanese": "sd",
    "algerian": "dz", "burmese": "mm", "myanmar": "mm", "yemeni": "ye",
    "somali": "so", "tunisian": "tn", "turkish": "tr", "egyptian": "eg",
    "filipino": "ph", "central african": "cf", "malian": "ml",
    "lebanese": "lb", "american": "us", "venezuelan": "ve",
    "jordanian": "jo", "rwandan": "rw", "georgian": "ge", "israeli": "il",
    "cambodian": "kh", "ugandan": "ug", "south sudanese": "ss",
    "cypriot": "cy", "palestinian": "ps", "kyrgyz": "kg", "kuwaiti": "kw",
    "moroccan": "ma", "uzbek": "uz", "south korean": "kr", "kenyan": "ke",
    "qatari": "qa", "kazakh": "kz", "kazakhstani": "kz", "tanzanian": "tz",
    "bosnian": "ba", "serbian": "rs", "armenian": "am", "azerbaijani": "az",
    "moldovan": "md", "tajik": "tj", "turkmen": "tm", "nicaraguan": "ni",
    "cuban": "cu", "zimbabwean": "zw", "eritrean": "er", "ethiopian": "et",
    "nigerian": "ng", "ghanaian": "gh", "south african": "za",
    # other common PSC nationalities
    "irish": "ie", "indian": "in", "romanian": "ro", "polish": "pl",
    "italian": "it", "french": "fr", "german": "de", "bulgarian": "bg",
    "spanish": "es", "dutch": "nl", "australian": "au", "lithuanian": "lt",
    "portuguese": "pt", "swedish": "se", "greek": "gr", "hungarian": "hu",
    "canadian": "ca", "norwegian": "no", "bangladeshi": "bd",
    "latvian": "lv", "belgian": "be", "new zealander": "nz", "czech": "cz",
    "danish": "dk", "malaysian": "my", "slovak": "sk", "swiss": "ch",
    "sri lankan": "lk", "brazilian": "br", "jamaican": "jm",
    "austrian": "at", "albanian": "al", "vietnamese": "vn",
    "hong konger": "hk", "japanese": "jp", "singaporean": "sg",
    "thai": "th", "estonian": "ee", "finnish": "fi", "croatian": "hr",
    "slovenian": "si", "maltese": "mt", "mauritian": "mu", "nepalese": "np",
}

_NATIONALITY_SPLIT_RE = re.compile(r"[,/;&]|\band\b", re.IGNORECASE)
_YEAR_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})")


# ---------------------------------------------------------------------------
# Small helpers (unit-tested)
# ---------------------------------------------------------------------------

def nationality_codes(nationality: str) -> set:
    """'British,Indian' -> {'gb', 'in'}. Unknown parts are dropped."""
    codes = set()
    for part in _NATIONALITY_SPLIT_RE.split(nationality or ""):
        code = DEMONYM_TO_ISO.get(" ".join(part.lower().split()))
        if code:
            codes.add(code)
    return codes


def country_codes(countries: str) -> set:
    """'gb;gb-sct;in' -> {'gb', 'in'} (subdivisions folded into the country)."""
    return {c.strip().lower().split("-")[0] for c in (countries or "").split(";") if c.strip()}


def nationality_consistent(nationality: str, countries: str) -> bool:
    """True only if a known PSC nationality appears in the list's countries."""
    return bool(nationality_codes(nationality) & country_codes(countries))


def birth_year_months(birth_date: str) -> set:
    """'1962-07-24;1963-01' -> {(1962, 7), (1963, 1)}. Year-only dates are dropped."""
    out = set()
    for part in (birth_date or "").split(";"):
        m = _YEAR_MONTH_RE.match(part.strip())
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def psc_year_month(birth_year: str, birth_month: str):
    try:
        return int(float(birth_year)), int(float(birth_month))
    except (TypeError, ValueError):
        return None


def detect_mode(psc_columns, list_columns) -> str:
    """'A' if both sides carry birth data, else 'B'."""
    if PSC_BIRTH_COLS <= set(psc_columns) and LIST_BIRTH_COL in set(list_columns):
        return "A"
    return "B"


_APOSTROPHE_RE = re.compile(r"['’`]")


def person_word_count(raw_name: str) -> int:
    """Words in a person's name, counted before the apostrophe split:
    "Paul O'Neill" -> 2 (normalization alone would give "paul o neill")."""
    return len(normalize_name(_APOSTROPHE_RE.sub("", raw_name or ""))["tokens"])


def company_real_word_count(normalized: str) -> int:
    """Tokens of 2+ characters: "b h" (from "B&H") -> 0, "nord gold" -> 2."""
    return sum(len(t) >= MIN_REAL_TOKEN_LEN for t in (normalized or "").split())


def person_tier(mode: str, score: float, exact: bool, psc_name: str,
                nationality: str, countries: str,
                psc_ym=None, list_birth_date: str = "") -> str:
    if mode == "A":
        strong = score >= SCORE_CUTOFF and psc_ym is not None and \
            psc_ym in birth_year_months(list_birth_date)
    else:
        strong = exact and person_word_count(psc_name) >= MIN_STRONG_PERSON_TOKENS and \
            nationality_consistent(nationality, countries)
    return "strong" if strong else "weak"


def company_tier(exact: bool, normalized: str) -> str:
    strong = exact and company_real_word_count(normalized) >= MIN_STRONG_COMPANY_TOKENS
    return "strong" if strong else "weak"


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _surname_key(name: str) -> str:
    return name.split()[-1]


def _rarest_token_key_fn(names):
    freq = Counter(t for n in names for t in set(n.split()))
    # rarest first; ties -> longer token (more specific), then alphabetical
    return lambda name: min(name.split(), key=lambda t: (freq[t], -len(t), t))


def find_candidates(psc_names, list_names, block_key) -> dict:
    """
    psc_names, list_names: unique, non-empty normalized names.
    block_key(psc_name) -> the token a list name must contain to be compared.
    Returns {(psc_name, list_name): score} for every pair scoring >= cutoff.
    """
    list_set = set(list_names)
    pairs = {}

    # Pass 1: exact
    for name in psc_names:
        if name in list_set:
            pairs[(name, name)] = 100.0

    # Pass 2: fuzzy, blocked
    token_index = defaultdict(set)
    for name in list_set:
        for tok in name.split():
            token_index[tok].add(name)

    blocks = defaultdict(list)
    for name in psc_names:
        key = block_key(name)
        if key in token_index:
            blocks[key].append(name)

    for key, group in blocks.items():
        cands = sorted(token_index[key])
        scores = process.cdist(group, cands, scorer=fuzz.token_sort_ratio,
                               score_cutoff=SCORE_CUTOFF, workers=-1)
        for i, j in zip(*scores.nonzero()):
            pairs.setdefault((group[i], cands[j]), round(float(scores[i, j]), 1))
    return pairs


def _pairs_frame(pairs: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [(p, l, s) for (p, l), s in pairs.items()],
        columns=["psc_norm", "list_norm", "score"],
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _prepare_lists(pep: pd.DataFrame, sanctions: pd.DataFrame) -> dict:
    """Returns {list_source: (persons_df, companies_df)} in a common shape."""
    pep_std = pd.DataFrame({
        "list_norm": pep["normalized_name"],
        "list_name": pep["name"],
        "list_entity_id": pep["id"],
        "seed_source": "canonical",
        "countries": pep.get("countries", ""),
        "list_birth_date": pep[LIST_BIRTH_COL] if LIST_BIRTH_COL in pep else "",
    })
    sanc_std = pd.DataFrame({
        "list_norm": sanctions["normalized_name"],
        "list_name": sanctions["seed_name"],
        "list_entity_id": sanctions["source_id"],
        "seed_source": sanctions["seed_source"].str.replace("sanctions_", "", regex=False),
        "countries": sanctions.get("countries", ""),
        "list_birth_date": sanctions[LIST_BIRTH_COL] if LIST_BIRTH_COL in sanctions else "",
        "schema": sanctions["schema"],
    })
    pep_std = pep_std[pep_std["list_norm"] != ""]
    sanc_std = sanc_std[sanc_std["list_norm"] != ""]
    return {
        "pep": (pep_std, pep_std.iloc[0:0]),  # PEP list is people only
        "sanctions": (
            sanc_std[sanc_std["schema"].isin(PERSON_SCHEMAS)],
            sanc_std[sanc_std["schema"].isin(COMPANY_SCHEMAS)],
        ),
    }


def _expand(pairs: dict, psc_side: pd.DataFrame, list_side: pd.DataFrame) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame()
    df = _pairs_frame(pairs)
    df = df.merge(psc_side, left_on="psc_norm", right_on="normalized_name")
    df = df.merge(list_side, on="list_norm")
    df["exact"] = df["psc_norm"] == df["list_norm"]
    return df


def match(psc: pd.DataFrame, pep: pd.DataFrame, sanctions: pd.DataFrame):
    """
    Core matching on in-memory frames. `psc` must contain psc_row_id plus
    PSC_BASE_COLS (and birth_month/birth_year for Mode A).
    Returns (matches DataFrame with OUTPUT_FIELDS, counts dict).
    """
    is_ind = psc["kind"].str.startswith("individual")
    is_corp = psc["kind"].str.startswith("corporate")
    has_name = psc["normalized_name"] != ""
    people = psc[is_ind & has_name]
    corps = psc[is_corp & has_name]

    counts = {
        "psc_rows": len(psc),
        "psc_individual_rows": int(is_ind.sum()),
        "psc_corporate_rows": int(is_corp.sum()),
        "psc_skipped_other_kind": int((~is_ind & ~is_corp).sum()),
        "psc_skipped_empty_name": int(((is_ind | is_corp) & ~has_name).sum()),
    }
    skipped_kinds = psc.loc[~is_ind & ~is_corp, "kind"].value_counts().to_dict()

    lists = _prepare_lists(pep, sanctions)
    list_columns = {"pep": pep.columns, "sanctions": sanctions.columns}
    people_names = people["normalized_name"].unique().tolist()
    corp_names = corps["normalized_name"].unique().tolist()

    frames = []
    for source, (persons, companies) in lists.items():
        mode = detect_mode(psc.columns, list_columns[source])
        counts[f"mode_{source}"] = mode

        # People
        pairs = find_candidates(people_names, persons["list_norm"].unique().tolist(),
                                _surname_key)
        df = _expand(pairs, people, persons)
        if len(df):
            if mode == "A":
                yms = [psc_year_month(y, m) for y, m in zip(df["birth_year"], df["birth_month"])]
            else:
                yms = [None] * len(df)
            df["tier"] = [
                person_tier(mode, s, e, n, nat, c, ym, bd)
                for s, e, n, nat, c, ym, bd in zip(
                    df["score"], df["exact"], df["name"], df["nationality"],
                    df["countries"], yms, df["list_birth_date"])
            ]
            df["rule_used"] = mode
            frames.append(df.assign(list_source=source))

        # Companies (exact-name + 2 real words rule in both modes)
        if len(companies):
            all_names = corp_names + companies["list_norm"].unique().tolist()
            pairs = find_candidates(corp_names, companies["list_norm"].unique().tolist(),
                                    _rarest_token_key_fn(all_names))
            df = _expand(pairs, corps, companies)
            if len(df):
                df["tier"] = [company_tier(e, n) for e, n in zip(df["exact"], df["psc_norm"])]
                df["rule_used"] = "B"
                frames.append(df.assign(list_source=source))

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.rename(columns={"name": "psc_name"})
        # One row per (PSC row, list entity): strong > weak, high score, canonical first.
        out["_tier_rank"] = (out["tier"] != "strong").astype(int)
        out["_alias_rank"] = (out["seed_source"] != "canonical").astype(int)
        out = out.sort_values(["psc_row_id", "list_source", "list_entity_id",
                               "_tier_rank", "score", "_alias_rank"],
                              ascending=[True, True, True, True, False, True])
        out = out.drop_duplicates(["psc_row_id", "list_source", "list_entity_id"])
        out = out[OUTPUT_FIELDS].reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=OUTPUT_FIELDS)

    counts["skipped_kinds"] = skipped_kinds
    return out, counts


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_psc(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = PSC_BASE_COLS + sorted(PSC_BIRTH_COLS & set(header))
    psc = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)
    psc.insert(0, "psc_row_id", range(len(psc)))  # row position in psc_clean.csv
    return psc


def run(psc_path=PSC_INPUT, pep_path=PEP_INPUT, sanctions_path=SANCTIONS_INPUT,
        output_path=MATCHES_OUTPUT):
    psc = load_psc(psc_path)
    pep = pd.read_csv(pep_path, dtype=str, keep_default_na=False)
    sanctions = pd.read_csv(sanctions_path, dtype=str, keep_default_na=False)

    matches, counts = match(psc, pep, sanctions)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matches.to_csv(output_path, index=False)
    counts["matches_written"] = len(matches)
    return matches, counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--psc", default=PSC_INPUT)
    ap.add_argument("--pep", default=PEP_INPUT)
    ap.add_argument("--sanctions", default=SANCTIONS_INPUT)
    ap.add_argument("--output", default=MATCHES_OUTPUT)
    args = ap.parse_args()

    _, result_counts = run(args.psc, args.pep, args.sanctions, args.output)

    print("Matching summary:", file=sys.stderr)
    for k, v in result_counts.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Matches written to: {args.output}", file=sys.stderr)
