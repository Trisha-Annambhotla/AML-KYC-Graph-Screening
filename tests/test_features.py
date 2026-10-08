import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import compute_features, feature_summary, write_summary  # noqa: E402
from graph_builder import build_graph  # noqa: E402

IND = "individual-person-with-significant-control"
SHARES = "ownership-of-shares-75-to-100-percent"

# Addresses with distinct postcodes, so each gives a distinct address key.
ADDR = {k: f"{i} Main Street, Town, AB{i} 1CD" for i, k in enumerate("PQRSTUVWXYZ", start=1)}


def _graph(psc_rows, flagged=(), weak=()):
    """psc_rows: (company_number, person_letter, address_letter).
    flagged / weak: (company_number, person_letter, list_source) PSC rows to flag."""
    rows = [(co, IND, f"Person {p} Name", f"person {p.lower()} name", ADDR[a], SHARES)
            for co, p, a in psc_rows]
    psc = pd.DataFrame(rows, columns=["company_number", "kind", "name", "normalized_name",
                                      "address", "natures_of_control"])
    psc.insert(0, "psc_row_id", range(len(psc)))
    m = []
    for tier, items in (("strong", flagged), ("weak", weak)):
        for co, p, src in items:
            rid = psc.index[(psc.company_number == co) & (psc.name == f"Person {p} Name")][0]
            m.append((rid, co, f"Person {p} Name", src, "x", tier))
    matches = pd.DataFrame(m, columns=["psc_row_id", "company_number", "psc_name",
                                       "list_source", "list_name", "tier"])
    graph, _, _, unmapped = build_graph(psc, matches)
    assert unmapped.empty
    return compute_features(graph)[0].set_index("company_number")


def test_own_flagged_owner_gives_zero_flagged_neighbours():
    """P is flagged and controls A and C: both labelled, but P is the only link,
    so neither may count the other (that would leak the label)."""
    f = _graph([("A", "P", "P"), ("C", "P", "P")], flagged=[("A", "P", "pep")])
    assert f.loc["A", "label"] == 1 and f.loc["C", "label"] == 1
    for co in ("A", "C"):
        assert f.loc[co, "flagged_neighbour_companies"] == 0
        assert f.loc[co, "flagged_neighbour_via_owner"] == 0


def test_shared_address_count_excludes_itself():
    # A has two PSCs at the same address, B has one there too.
    f = _graph([("A", "Q", "Q"), ("A", "R", "Q"), ("B", "S", "Q"), ("D", "T", "T")])
    assert f.loc["A", "shared_address_count"] == 1
    assert f.loc["B", "shared_address_count"] == 1
    assert f.loc["D", "shared_address_count"] == 0


def test_shared_non_flagged_owner_counts_labelled_company():
    # L is labelled via flagged P; Q (not flagged) controls both L and A.
    f = _graph([("L", "P", "P"), ("L", "Q", "Q"), ("A", "Q", "Q")],
               flagged=[("L", "P", "sanctions")])
    assert f.loc["A", "flagged_neighbour_via_owner"] == 1
    assert f.loc["A", "flagged_neighbour_companies"] == 1
    assert f.loc["L", "flagged_neighbour_companies"] == 0  # A is not labelled


def test_shared_address_link_counts_only_in_new_version():
    # L labelled via flagged P; L's other PSC S lives at R's address; R owns A.
    f = _graph([("L", "P", "P"), ("L", "S", "R"), ("A", "R", "R")],
               flagged=[("L", "P", "pep")])
    assert f.loc["A", "flagged_neighbour_via_owner"] == 0
    assert f.loc["A", "flagged_neighbour_companies"] == 1


def test_flagged_psc_address_is_not_used():
    # A's PSC lives at the flagged P's address; that alone must not link A to L.
    f = _graph([("L", "P", "P"), ("A", "Q", "P")], flagged=[("L", "P", "pep")])
    assert f.loc["A", "flagged_neighbour_companies"] == 0
    assert f.loc["A", "shared_address_count"] == 1  # feature 2 still sees the address


def test_hub_component_and_label_source():
    rows = [(f"H{i:03d}", "Z", "Z") for i in range(100)] + [("B", "X", "X"), ("B", "Y", "Y")]
    f = _graph(rows, flagged=[("B", "X", "pep"), ("B", "Y", "sanctions")], weak=[("H000", "Z", "pep")])
    assert f.loc["H000", "is_hub_component"] and not f.loc["B", "is_hub_component"]
    assert f.loc["B", "label_source"] == "both"
    assert f.loc["B", "degree"] == 2 and f.loc["B", "component_size"] == 3
    assert f.loc["H050", "label"] == 0 and f.loc["H050", "label_weak"] == 1


def test_summary_writes_report(tmp_path):
    f = _graph([("L", "P", "P"), ("L", "Q", "Q"), ("A", "Q", "Q"), ("D", "T", "T")],
               flagged=[("L", "P", "pep")]).reset_index()
    s = feature_summary(f)
    assert s["n_positive"] == 1
    assert s["nonzero"]["flagged_neighbour_via_owner"] == {"companies": 1, "positives": 0}
    write_summary(s, tmp_path / "stats.md")
    assert "Correlation with label" in (tmp_path / "stats.md").read_text(encoding="utf-8")


def test_address_shared_by_more_than_50_companies_is_ignored():
    # L is labelled; L's co-owner Q and the owner of 50 other companies share
    # address Q -> 51 companies at that key (> 50), so it links nothing.
    rows = [("L", "P", "P"), ("L", "Q", "Q")]
    rows += [(f"C{i:02d}", "R", "Q") for i in range(50)]
    f = _graph(rows, flagged=[("L", "P", "pep")])
    assert f.loc["C00", "flagged_neighbour_companies"] == 0
    assert f.loc["C00", "shared_address_count"] == 0
    assert f.loc["L", "shared_address_count"] == 0


def test_address_shared_by_50_companies_still_counts():
    rows = [("L", "P", "P"), ("L", "Q", "Q")]
    rows += [(f"C{i:02d}", "R", "Q") for i in range(49)]  # L + 49 = exactly 50 companies
    f = _graph(rows, flagged=[("L", "P", "pep")])
    assert f.loc["C00", "flagged_neighbour_companies"] == 1
    assert f.loc["C00", "shared_address_count"] == 49


def test_split_group_merges_address_linked_components():
    # A and B have different owners (separate components) at the same address;
    # D is unrelated.
    f = _graph([("A", "Q", "Q"), ("B", "R", "Q"), ("D", "T", "T")])
    assert f.loc["A", "component_id"] != f.loc["B", "component_id"]
    assert f.loc["A", "split_group_id"] == f.loc["B", "split_group_id"]
    assert f.loc["D", "split_group_id"] != f.loc["A", "split_group_id"]


def test_split_group_ignores_capped_addresses():
    # 51 companies at address Q -> capped, so the address must not merge them.
    rows = [(f"C{i:02d}", "PQRSTUVWXYZ"[i % 11], "Q") for i in range(51)]
    f = _graph(rows)
    # Owners repeat every 11 companies, so C00 and C11 share an owner (same
    # component) but C00 and C01 do not, and the capped address must not join them.
    assert f.loc["C00", "split_group_id"] == f.loc["C11", "split_group_id"]
    assert f.loc["C00", "split_group_id"] != f.loc["C01", "split_group_id"]
