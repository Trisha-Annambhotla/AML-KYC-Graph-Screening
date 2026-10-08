import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import case_studies as cs  # noqa: E402
from features import compute_features  # noqa: E402
from graph_builder import build_graph  # noqa: E402

IND = "individual-person-with-significant-control"
SHARES = "ownership-of-shares-75-to-100-percent"


def _addr(i):
    return f"{i} Main Street, Town, AB{i} 1CD"


def _world():
    """Flagged 'Peter Caruana' owns C1..C3 (risky) with non-flagged co-owner Q, who
    also owns N1 (feature 1 fires there). Plus ordinary companies and one flagged
    person with a single company."""
    rows = [(f"C{i}", "Peter Caruana", 1) for i in range(1, 4)]
    rows += [("C1", "Quinn Co Owner", 2), ("N1", "Quinn Co Owner", 2)]
    rows += [("S1", "Sam Single Flag", 3)]
    rows += [(f"O{i:02d}", f"Owner Number {i}", 10 + i) for i in range(30)]
    psc = pd.DataFrame(
        [(co, IND, name, name.lower(), _addr(a), SHARES) for co, name, a in rows],
        columns=["company_number", "kind", "name", "normalized_name", "address",
                 "natures_of_control"])
    psc.insert(0, "psc_row_id", range(len(psc)))
    m = pd.DataFrame(
        [(0, "C1", "Peter Caruana", "pep", "x", "strong"),
         (5, "S1", "Sam Single Flag", "sanctions", "x", "strong")],
        columns=["psc_row_id", "company_number", "psc_name", "list_source", "list_name", "tier"])
    graph, _, _, unmapped = build_graph(psc, m)
    assert unmapped.empty
    feats, _ = compute_features(graph)
    return graph, feats


def _predictions(feats):
    rng = np.random.default_rng(0)
    p = feats[["company_number"]].copy()
    p["fold"] = 0
    for m in cs.RANK_METHODS:
        p[f"score_{m}"] = rng.random(len(p))
    p.loc[p["company_number"] == "O05", "score_random_forest"] = 2.0  # top false positive
    p.loc[p["company_number"] == "S1", "score_random_forest"] = 1.5   # best positive
    return p


def test_ranks_follow_scores():
    df = pd.DataFrame({f"score_{m}": [0.1, 0.9, 0.5] for m in cs.RANK_METHODS})
    out = cs.add_ranks(df)
    assert out["rank_random_forest"].tolist() == [3, 1, 2]


def test_feature_twins_counts_identical_profiles():
    df = pd.DataFrame({"flagged_neighbour_companies": [0, 0, 0, 1],
                       "shared_address_count": [2, 2, 2, 2], "degree": [1, 1, 1, 1],
                       "component_size": [2, 2, 3, 2], "label": [1, 0, 0, 0]})
    t = cs.feature_twins(df, df.iloc[0])
    assert t == {"n": 2, "positives": 1, "rate": 0.5}


def test_neighbourhood_folds_extra_neighbours():
    g = nx.DiGraph()
    g.add_node("p", node_type="person")
    for i in range(12):
        g.add_node(f"c{i}", node_type="company")
        g.add_edge("p", f"c{i}")
    nodes, _, more = cs.neighbourhood(g, "p", 1, important={"c11"}, max_per_node=3)
    assert "c11" in nodes and len(nodes) == 4
    assert more == {"more:p": ("p", 9)}


def test_select_cases_and_caruana(tmp_path):
    graph, feats = _world()
    df = cs.add_ranks(feats.merge(_predictions(feats), on="company_number"))
    cases = cs.select_cases(df, graph)
    assert cases["top_false_pos"] == "company:O05"
    assert cases["best_positive"] == "company:S1"
    assert cases["caruana_group"].startswith("person:peter caruana")
    links = cs.feature1_links(graph, "company:N1",
                              {"company:C1", "company:C2", "company:C3", "company:S1"},
                              cs.address_key_sizes(graph))
    # Quinn co-owns C1 and N1 at the same address, so both links apply (one row).
    assert links == [("company:C1",
                      "shared owner Quinn Co Owner + shared address 'ab2 1cd | 2 main street'")]


def test_run_writes_doc_and_images(tmp_path):
    graph, feats = _world()
    paths = {k: tmp_path / f"{k}" for k in ("graph.pkl", "features.csv", "predictions.csv",
                                            "metrics.csv")}
    with open(paths["graph.pkl"], "wb") as f:
        pickle.dump(graph, f)
    feats.to_csv(paths["features.csv"], index=False)
    _predictions(feats).to_csv(paths["predictions.csv"], index=False)
    pd.DataFrame([{"run": "main", "method": m, "pr_auc_mean": 0.1, "pr_auc_std": 0.01,
                   "roc_auc_mean": 0.5, "accuracy_at_0_5": 0.9}
                  for m in cs.RANK_METHODS]).to_csv(paths["metrics.csv"], index=False)
    out = cs.run(paths["graph.pkl"], paths["features.csv"], paths["predictions.csv"],
                 paths["metrics.csv"], tmp_path / "results", tmp_path / "case_studies.md")
    for name in ("case_best_positive.png", "case_top_false_pos.png", "case_typical_pos.png",
                 "case_caruana_group.png", "case_model_comparison.png"):
        assert (tmp_path / "results" / name).exists(), name
    doc = (tmp_path / "case_studies.md").read_text(encoding="utf-8")
    assert "Random Forest vs Logistic Regression" in doc
    assert out["caruana_group"]["companies"] == 3
    assert out["caruana_group"]["feature1_via_group"] == 1
