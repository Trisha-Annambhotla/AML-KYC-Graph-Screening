import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spot_check import draw_sample, precision_table, weighted_weak_precision  # noqa: E402


def _matches():
    rows = []
    for i in range(10):
        rows.append((i, f"0000000{i}", "strong", "pep", 100.0))
    for i in range(10, 300):
        rows.append((i, f"1{i:07d}", "weak", "pep" if i % 3 else "sanctions", 90.0 + (i % 10)))
    df = pd.DataFrame(rows, columns=["psc_row_id", "company_number", "tier", "list_source", "score"])
    df["list_entity_id"] = "E" + df["psc_row_id"].astype(str)
    return df.astype(str)


def test_all_strong_kept_and_weak_capped():
    s = draw_sample(_matches(), 50, 50)
    assert (s["tier"] == "strong").sum() == 10
    assert (s["tier"] == "weak").sum() == 50
    assert (s["is_correct"] == "").all()


def test_sample_is_reproducible():
    a = draw_sample(_matches())
    b = draw_sample(_matches())
    assert a["psc_row_id"].tolist() == b["psc_row_id"].tolist()


def test_precision_ignores_unsure_and_blank():
    s = pd.DataFrame({
        "tier": ["strong"] * 5, "list_source": ["pep"] * 5,
        "is_correct": ["y", "y", "n", "?", ""],
    })
    t = precision_table(s)
    row = t[(t.tier == "strong") & (t.list == "all")].iloc[0]
    assert row.y == 2 and row.n == 1 and row.unsure == 1 and row.unlabelled == 1
    assert abs(row.precision - 2 / 3) < 1e-9


def test_weighted_weak_precision():
    s = pd.DataFrame({
        "tier": ["weak"] * 4, "stratum": ["a", "a", "b", "b"],
        "stratum_size": ["100", "100", "300", "300"],
        "is_correct": ["y", "n", "n", "n"],
    })
    assert abs(weighted_weak_precision(s) - 0.125) < 1e-9
