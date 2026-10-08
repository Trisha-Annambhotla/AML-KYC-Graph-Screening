import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matching import (  # noqa: E402
    birth_year_months, detect_mode, match, nationality_consistent,
)


def _psc(rows, birth=False):
    cols = ["company_number", "kind", "name", "normalized_name", "nationality"]
    if birth:
        cols += ["birth_year", "birth_month"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "psc_row_id", range(len(df)))
    return df


def _pep(rows=(), birth=True):
    cols = ["id", "name", "normalized_name", "countries"] + (["birth_date"] if birth else [])
    return pd.DataFrame(list(rows), columns=cols)


def _sanctions(rows=()):
    cols = ["source_id", "schema", "seed_name", "seed_source", "normalized_name", "countries"]
    return pd.DataFrame(list(rows), columns=cols)


IND = "individual-person-with-significant-control"
CORP = "corporate-entity-person-with-significant-control"


def test_exact_match_is_strong_in_mode_b():
    psc = _psc([("001", IND, "Ivan Petrovich Sidorov", "ivan petrovich sidorov", "Russian")])
    sanc = _sanctions([("S1", "Person", "Ivan Petrovich SIDOROV", "sanctions_canonical",
                        "ivan petrovich sidorov", "ru")])
    out, counts = match(psc, _pep(), sanc)
    assert counts["mode_sanctions"] == "B"
    assert len(out) == 1
    assert out.loc[0, "tier"] == "strong"
    assert out.loc[0, "rule_used"] == "B"
    assert out.loc[0, "list_source"] == "sanctions"


def test_two_word_name_is_never_strong_in_mode_b():
    psc = _psc([("001", IND, "Ivan Sidorov", "ivan sidorov", "Russian")])
    sanc = _sanctions([("S1", "Person", "Ivan SIDOROV", "sanctions_canonical",
                        "ivan sidorov", "ru")])
    out, _ = match(psc, _pep(), sanc)
    assert len(out) == 1
    assert out.loc[0, "tier"] == "weak"


def test_unknown_nationality_is_never_strong_in_mode_b():
    psc = _psc([("001", IND, "Ivan Petrovich Sidorov", "ivan petrovich sidorov", "")])
    sanc = _sanctions([("S1", "Person", "Ivan Petrovich SIDOROV", "sanctions_canonical",
                        "ivan petrovich sidorov", "ru")])
    out, _ = match(psc, _pep(), sanc)
    assert out.loc[0, "tier"] == "weak"


def test_person_is_never_matched_to_an_organisation():
    psc = _psc([
        ("001", IND, "Electra Pro Trading", "electra pro trading", "Russian"),
        ("002", CORP, "Ivan Petrovich Sidorov Ltd", "ivan petrovich sidorov", ""),
    ])
    sanc = _sanctions([
        ("S1", "Organization", "Electra Pro Trading LLC", "sanctions_canonical",
         "electra pro trading", "ru"),
        ("S2", "Person", "Ivan Petrovich SIDOROV", "sanctions_canonical",
         "ivan petrovich sidorov", "ru"),
    ])
    out, _ = match(psc, _pep(), sanc)
    assert out.empty


def test_company_exact_match_is_strong():
    psc = _psc([("001", CORP, "Electra Pro Ltd", "electra pro", "")])
    sanc = _sanctions([("S1", "Organization", "ELECTRA PRO LLC", "sanctions_canonical",
                        "electra pro", "ru")])
    out, _ = match(psc, _pep(), sanc)
    assert out.loc[0, "tier"] == "strong"


def test_fuzzy_match_is_weak():
    psc = _psc([("001", IND, "Ilya Vladimirovich Sidorov", "ilya vladimirovich sidorov", "Russian")])
    sanc = _sanctions([("S1", "Person", "Ilia Vladimirovich Sidorov", "sanctions_canonical",
                        "ilia vladimirovich sidorov", "ru")])
    out, _ = match(psc, _pep(), sanc)
    assert len(out) == 1
    assert out.loc[0, "score"] >= 90
    assert out.loc[0, "tier"] == "weak"


def test_mode_a_is_chosen_when_birth_columns_exist():
    assert detect_mode(["birth_month", "birth_year", "name"], ["birth_date"]) == "A"
    assert detect_mode(["name"], ["birth_date"]) == "B"
    assert detect_mode(["birth_month", "birth_year"], ["name"]) == "B"

    # End to end: a two-word name with matching birth month/year is strong in Mode A.
    psc = _psc([("001", IND, "John Smith", "john smith", "British", "1962", "7")], birth=True)
    pep = _pep([("P1", "John Smith", "john smith", "gb", "1962-07-24")])
    out, counts = match(psc, pep, _sanctions())
    assert counts["mode_pep"] == "A"
    assert counts["mode_sanctions"] == "B"  # sanctions has no birth_date column
    assert out.loc[0, "rule_used"] == "A"
    assert out.loc[0, "tier"] == "strong"


def test_mode_a_wrong_birth_month_is_weak():
    psc = _psc([("001", IND, "John Smith", "john smith", "British", "1962", "8")], birth=True)
    pep = _pep([("P1", "John Smith", "john smith", "gb", "1962-07-24")])
    out, _ = match(psc, pep, _sanctions())
    assert out.loc[0, "tier"] == "weak"


def test_skipped_kinds_are_counted():
    psc = _psc([("001", "legal-person-person-with-significant-control", "X", "x", "")])
    _, counts = match(psc, _pep(), _sanctions())
    assert counts["psc_skipped_other_kind"] == 1


def test_helpers():
    assert nationality_consistent("British,Indian", "gb;in")
    assert nationality_consistent("Scottish", "gb-sct")
    assert not nationality_consistent("Martian", "ru")
    assert birth_year_months("1962-07-24;1963-01-02;1970") == {(1962, 7), (1963, 1)}
