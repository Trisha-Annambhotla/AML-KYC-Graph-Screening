import os
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import investigate as inv  # noqa: E402


def _row(**kw):
    base = {"label": 0, "degree": 1, "shared_address_count": 0,
            "flagged_neighbour_companies": 0, "component_size": 2,
            "shortest_distance_to_flagged": -1, "is_hub_component": False}
    base.update(kw)
    return pd.Series(base)


def _matches(rows=()):
    return pd.DataFrame(list(rows), columns=["psc_name", "list_source", "list_name", "tier",
                                             "rule_used", "score", "kind"])


TWIN = {"n": 229197, "positives": 22, "rate": 22 / 229197}


def test_strong_pep_match_mode_a_is_explained():
    m = _matches([("Jack Dromey", "pep", "Jack Dromey", "strong", "A", 100.0,
                   "individual-person-with-significant-control")])
    facts = inv.explain(_row(label=1, degree=3), m, TWIN)
    assert any("matched to the PEP list entry 'Jack Dromey'" in f
               and "birth month/year agree" in f for f in facts)
    assert any(f.startswith("Labelled risky") for f in facts)
    assert "Has 3 owners (PSCs)." in facts


def test_unreachable_distance_says_no_connection():
    facts = inv.explain(_row(), _matches(), TWIN)
    assert "No connection to any other flagged owner." in facts


def test_distance_and_cap_wording():
    assert "3 hops from a flagged owner (not counting its own owners)." in \
        inv.explain(_row(shortest_distance_to_flagged=3), _matches(), TWIN)
    assert any(f.startswith("7 or more hops") for f in
               inv.explain(_row(shortest_distance_to_flagged=6), _matches(), TWIN))


def test_shared_address_and_twins_wording():
    facts = inv.explain(_row(shared_address_count=17), _matches(), TWIN)
    assert "Shares an owner address with 17 other companies." in facts
    assert any(f.startswith("229,196 other companies have exactly the same feature profile")
               and "22 of the 229,197" in f for f in facts)
    one = inv.explain(_row(shared_address_count=1), _matches(), {"n": 2, "positives": 0})
    assert "Shares an owner address with 1 other company." in one
    assert any(f.startswith("1 other company has exactly the same feature profile")
               for f in one)


def test_no_shared_address_mentions_cap():
    facts = inv.explain(_row(), _matches(), TWIN)
    assert any("Shares no owner address" in f and "more than 50 companies" in f for f in facts)


def test_weak_and_company_matches():
    m = _matches([
        ("Electra Pro Ltd", "sanctions", "ELECTRA PRO LLC", "strong", "B", 100.0,
         "corporate-entity-person-with-significant-control"),
        ("John Smith", "pep", "John Smyth", "weak", "A", 91.0,
         "individual-person-with-significant-control"),
        ("John Smith", "sanctions", "John SMITH", "weak", "A", 100.0,
         "individual-person-with-significant-control"),
    ])
    facts = inv.explain(_row(label=1), m, TWIN)
    assert any("sanctions list entry 'ELECTRA PRO LLC'" in f
               and "exact company name match" in f for f in facts)
    assert "2 weak name matches (similar names only; not used as labels)." in facts


def test_hub_and_feature1_facts():
    facts = inv.explain(_row(is_hub_component=True, flagged_neighbour_companies=1),
                        _matches(), TWIN)
    assert any("controls 100+ companies" in f for f in facts)
    assert any(f.startswith("Linked to 1 other risky company") for f in facts)


def test_normalise_company_number():
    assert inv.normalise_company_number(" 7434180 ") == "07434180"
    assert inv.normalise_company_number("sc447141") == "SC447141"


def test_ego_network_caps_nodes_and_hops():
    g = nx.DiGraph()
    g.add_node("company:C", node_type="company")
    g.add_node("person:P", node_type="person", name="P", flagged=False)
    g.add_edge("person:P", "company:C")
    for i in range(200):  # P controls 200 other companies (2 hops from C)
        g.add_node(f"company:X{i:03d}", node_type="company")
        g.add_edge("person:P", f"company:X{i:03d}")
    g.add_node("person:Far", node_type="person", name="Far", flagged=True)
    g.add_edge("person:Far", "company:X000")  # 3 hops from C: never shown
    sub, total, truncated = inv.ego_network(g, "company:C", max_hops=5)
    assert "person:Far" not in sub          # max_hops is clamped to 2
    assert len(sub) == inv.MAX_NODES and truncated
    assert total == 202                     # C + P + 200 companies
    assert "company:C" in sub and "person:P" in sub


def test_search_owners():
    owners = pd.DataFrame({"node_id": ["a", "b", "c"], "name": ["Peter Caruana", "Peter Pan", "Al"],
                           "node_type": "person", "flagged": [True, False, False],
                           "flag_source": ["pep", "", ""], "n_companies": [8, 1, 1]})
    assert inv.search_owners(owners, "peter")["node_id"].tolist() == ["a", "b"]
    assert inv.search_owners(owners, "al").empty  # fewer than 3 letters
