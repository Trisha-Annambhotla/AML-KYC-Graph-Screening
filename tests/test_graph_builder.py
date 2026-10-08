import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_builder import (  # noqa: E402
    address_key, build_graph, extract_postcode, first_address_line, psc_node_id,
)

IND = "individual-person-with-significant-control"
CORP = "corporate-entity-person-with-significant-control"
SHARES = "ownership-of-shares-75-to-100-percent"


def _psc(rows):
    df = pd.DataFrame(rows, columns=["company_number", "kind", "name", "normalized_name",
                                     "address", "natures_of_control"])
    df.insert(0, "psc_row_id", range(len(df)))
    return df


def _matches(rows=()):
    return pd.DataFrame(list(rows), columns=["psc_row_id", "company_number", "psc_name",
                                             "list_source", "list_name", "tier"])


JOHN_HULL = ("John Paul Smith", "john paul smith", "Alvenga, Godmans Lane, Hull, HU10 7NY, England")


def test_same_person_two_companies_is_one_node_with_two_edges():
    psc = _psc([
        ("00000001", IND, *JOHN_HULL, SHARES),
        ("00000002", IND, *JOHN_HULL, SHARES),
    ])
    g, nodes, edges, _ = build_graph(psc, _matches())
    person = "person:john paul smith|HU10 7NY"
    assert (nodes["node_type"] == "person").sum() == 1
    assert g.out_degree(person) == 2
    assert set(g.successors(person)) == {"company:00000001", "company:00000002"}


def test_same_name_different_postcode_stays_separate():
    psc = _psc([
        ("00000001", IND, *JOHN_HULL, SHARES),
        ("00000002", IND, "John Paul Smith", "john paul smith", "1 High St, Leeds, LS7 3QB", SHARES),
    ])
    _, nodes, _, _ = build_graph(psc, _matches())
    assert (nodes["node_type"] == "person").sum() == 2


def test_flagged_match_lands_on_right_node():
    psc = _psc([
        ("00000001", IND, *JOHN_HULL, SHARES),
        ("00000002", IND, "Jane Doe Brown", "jane doe brown", "2 Low Rd, York, YO1 7HH", SHARES),
        ("00000002", CORP, "Electra Pro Ltd", "electra pro", "Moscow", SHARES),
    ])
    matches = _matches([
        (2, "00000002", "Electra Pro Ltd", "sanctions", "ELECTRA PRO LLC", "strong"),
        (0, "00000001", "John Paul Smith", "pep", "John Paul Smith", "strong"),
        (0, "00000001", "John Paul Smith", "sanctions", "John Paul SMITH", "strong"),
        (1, "00000002", "Jane Doe Brown", "pep", "Jane Brown", "weak"),
    ])
    g, _, _, unmapped = build_graph(psc, matches)
    assert unmapped.empty
    assert g.nodes["corp:electra pro"]["flagged"] is True
    assert g.nodes["corp:electra pro"]["flag_source"] == "sanctions"
    assert g.nodes["person:john paul smith|HU10 7NY"]["flag_source"] == "both"
    jane = g.nodes["person:jane doe brown|YO1 7HH"]
    assert jane["flagged"] is False and jane["weak_flagged"] is True
    assert g.nodes["company:00000002"]["flagged"] is False


def test_stale_match_is_reported_not_flagged():
    psc = _psc([("00000001", IND, *JOHN_HULL, SHARES)])
    matches = _matches([(0, "00000001", "Someone Else", "pep", "Someone Else", "strong")])
    g, _, _, unmapped = build_graph(psc, matches)
    assert len(unmapped) == 1
    assert g.nodes["person:john paul smith|HU10 7NY"]["flagged"] is False


def test_duplicate_psc_rows_merge_into_one_edge():
    psc = _psc([
        ("00000001", IND, *JOHN_HULL, SHARES),
        ("00000001", IND, *JOHN_HULL, "voting-rights-75-to-100-percent"),
    ])
    g, _, edges, _ = build_graph(psc, _matches())
    assert len(edges) == 1
    data = g.edges["person:john paul smith|HU10 7NY", "company:00000001"]
    assert data["natures_of_control"] == f"{SHARES};voting-rights-75-to-100-percent"
    assert data["psc_row_ids"] == "0;1"


def test_address_helpers():
    assert extract_postcode("The Manor, Boddington, GL51 0tj, England") == "GL51 0TJ"
    assert extract_postcode("Im Grund 12, 55491, Germany") == ""
    assert first_address_line("3, Vicarage Crescent, London, SW11 3LP") == "3 Vicarage Crescent"
    assert first_address_line("Third Floor, 207 Regent Street, London") == "Third Floor"
    assert address_key("3,  Vicarage Crescent, London, SW11 3LP") == "sw11 3lp | 3 vicarage crescent"
    assert address_key("Im Grund 12, Germany") == "im grund 12, germany"
    assert psc_node_id(CORP, "electra pro", "Moscow") == "corp:electra pro"
    assert psc_node_id(IND, "a b c", "Im Grund 12") == "person:a b c|im grund 12"
