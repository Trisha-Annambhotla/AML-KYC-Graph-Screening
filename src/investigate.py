"""
Data logic for the investigation dashboard (app/dashboard.py).

Kept free of Streamlit so it can be tested: search, a company's owners and
list matches, the plain-English "Why this score?" facts, and a 2-hop network
neighbourhood capped at a maximum number of nodes.
"""

import os
import pickle

import networkx as nx
import pandas as pd

import case_studies
from features import ADDRESS_KEY_MAX_COMPANIES, FEATURES, MAX_FLAG_DISTANCE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
GRAPH_INPUT = os.path.join(PROCESSED, "graph.pkl")
MATCHES_INPUT = os.path.join(PROCESSED, "matches.csv")
METRICS_INPUT = os.path.join(ROOT, "results", "metrics.csv")
METRICS_DOC = os.path.join(ROOT, "results", "metrics.md")

MAX_HOPS = 2
MAX_NODES = 150

LIST_NAMES = {"pep": "PEP", "sanctions": "sanctions"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: str = GRAPH_INPUT) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_companies() -> pd.DataFrame:
    """features.csv + out-of-fold scores + ranks (same tie-break as Phase 4)."""
    return case_studies.load_table()


def load_matches(path: str = MATCHES_INPUT) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"company_number": str}, keep_default_na=False)


def load_metrics(path: str = METRICS_INPUT) -> pd.DataFrame:
    return pd.read_csv(path)


def owner_index(graph: nx.DiGraph) -> pd.DataFrame:
    """One row per owner (PSC) node: id, name, type, flags, number of companies."""
    rows = [(n, d["name"], d["node_type"], d.get("flagged", False), d.get("flag_source", ""),
             graph.out_degree(n))
            for n, d in graph.nodes(data=True) if d["node_type"] != "company"]
    return pd.DataFrame(rows, columns=["node_id", "name", "node_type", "flagged",
                                       "flag_source", "n_companies"])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def normalise_company_number(text: str) -> str:
    """'7434180' -> '07434180'; 'sc447141' -> 'SC447141'."""
    t = (text or "").strip().upper().replace(" ", "")
    return t.zfill(8) if t.isdigit() else t


def search_owners(owners: pd.DataFrame, query: str, limit: int = 50) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if len(q) < 3:
        return owners.iloc[0:0]
    hits = owners[owners["name"].str.lower().str.contains(q, regex=False)]
    return hits.sort_values(["flagged", "n_companies"], ascending=False).head(limit)


# ---------------------------------------------------------------------------
# One company
# ---------------------------------------------------------------------------

def company_owners(graph: nx.DiGraph, company_number: str) -> pd.DataFrame:
    node = "company:" + company_number
    rows = []
    for p in graph.predecessors(node):
        d, e = graph.nodes[p], graph.edges[p, node]
        rows.append({
            "owner": d["name"], "type": d["node_type"],
            "flagged": d.get("flag_source", "") or "",
            "weak_flagged": d.get("weak_flag_source", "") or "",
            "controls_companies": graph.out_degree(p),
            "address_key": e.get("address_key", ""),
            "natures_of_control": e.get("natures_of_control", ""),
        })
    return pd.DataFrame(rows)


def company_matches(matches: pd.DataFrame, company_number: str) -> pd.DataFrame:
    m = matches[matches["company_number"] == company_number]
    return m.sort_values(["tier", "score"], ascending=[True, False])


def match_reason(row) -> str:
    """Plain-English reason a list match is strong or weak."""
    who = "company name" if str(row["kind"]).startswith("corporate") else "name"
    if row["tier"] == "strong":
        if row["rule_used"] == "A":
            return f"{who} similarity {float(row['score']):.0f}/100 and birth month/year agree"
        if who == "company name":
            return "exact company name match (2+ real words)"
        return "exact name (3+ words) and nationality consistent with the list"
    return (f"{who} similarity {float(row['score']):.0f}/100 only -- not enough to count "
            "as the same person (weak; not used as a label)")


def twins(companies: pd.DataFrame, row) -> dict:
    """Companies with exactly the same four model features."""
    return case_studies.feature_twins(companies, row)


def explain(row, matches: pd.DataFrame, twin: dict) -> list:
    """
    Plain-English facts behind a company's label and score, built only from
    the data. `row`: the company's features.csv row (+ ranks); `matches`:
    its rows from matches.csv; `twin`: feature_twins() output.
    """
    facts = []
    strong = matches[matches["tier"] == "strong"]
    weak = matches[matches["tier"] == "weak"]
    for r in strong.itertuples():
        facts.append(f"Owner {r.psc_name} matched to the {LIST_NAMES.get(r.list_source, r.list_source)} "
                     f"list entry '{r.list_name}' ({match_reason(r._asdict())}).")
    if len(weak):
        facts.append(f"{len(weak)} weak name match{'es' if len(weak) > 1 else ''} "
                     "(similar names only; not used as labels).")
    if int(row["label"]) == 1:
        facts.append("Labelled risky: it is directly owned by a strongly flagged owner.")
    else:
        facts.append("Not labelled risky: none of its owners is strongly flagged.")

    deg = int(row["degree"])
    facts.append(f"Has {deg} owner{'s' if deg != 1 else ''} (PSCs).")

    sa = int(row["shared_address_count"])
    if sa:
        facts.append(f"Shares an owner address with {sa:,} other compan{'y' if sa == 1 else 'ies'}.")
    else:
        facts.append("Shares no owner address with other companies (addresses used by more "
                     f"than {ADDRESS_KEY_MAX_COMPANIES} companies are ignored).")

    fn = int(row["flagged_neighbour_companies"])
    if fn:
        facts.append(f"Linked to {fn} other risky compan{'y' if fn == 1 else 'ies'} through a "
                     "non-flagged owner or a shared address.")

    dist = int(row["shortest_distance_to_flagged"])
    if dist == -1:
        facts.append("No connection to any other flagged owner.")
    elif dist >= MAX_FLAG_DISTANCE:
        facts.append(f"{MAX_FLAG_DISTANCE + 1} or more hops from a flagged owner "
                     "(not counting its own owners).")
    else:
        facts.append(f"{dist} hops from a flagged owner (not counting its own owners).")

    cs = int(row["component_size"])
    facts.append(f"Its ownership group has {cs:,} nodes (companies and owners).")
    if bool(row.get("is_hub_component", False)):
        facts.append("Its group contains an owner who controls 100+ companies "
                     "(often a formation agent).")

    others = twin["n"] - 1
    facts.append(f"{others:,} other compan{'y has' if others == 1 else 'ies have'} exactly the same "
                 f"feature profile ({twin['positives']} of the {twin['n']:,} with this profile "
                 "are risky), so a model using these four features cannot tell them apart.")
    return facts


# ---------------------------------------------------------------------------
# Network view
# ---------------------------------------------------------------------------

def ego_network(graph: nx.DiGraph, center: str, max_hops: int = MAX_HOPS,
                max_nodes: int = MAX_NODES):
    """
    Nodes within `max_hops` of `center` (undirected), at most `max_nodes`.
    Closer nodes first; within a hop, flagged owners and risky-looking nodes
    first, then by id (deterministic). Returns (subgraph, total_in_reach,
    truncated).
    """
    max_hops = min(max_hops, MAX_HOPS)  # never more than 2 hops
    und = graph.to_undirected(as_view=True)
    levels, seen, frontier = [[center]], {center}, [center]
    for _ in range(max_hops):
        nxt = set()
        for n in frontier:
            nxt.update(m for m in und.neighbors(n) if m not in seen)
        seen |= nxt
        frontier = sorted(nxt, key=lambda m: (not graph.nodes[m].get("flagged"), m))
        levels.append(frontier)
    total = len(seen)
    keep = []
    for level in levels:
        for n in level:
            if len(keep) >= max_nodes:
                break
            keep.append(n)
    sub = graph.subgraph(keep).copy()
    return sub, total, total > len(keep)


def score_table(row, n: int) -> pd.DataFrame:
    names = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
    return pd.DataFrame([
        {"model": names[m], "score": round(float(row[f"score_{m}"]), 4),
         "rank": f"{int(row[f'rank_{m}']):,} of {n:,}"}
        for m in names
    ])


def feature_table(row) -> pd.DataFrame:
    cols = FEATURES + ["shortest_distance_to_flagged"]
    return pd.DataFrame({"feature": cols,
                         "value": [int(row[c]) for c in cols],
                         "used by the models": ["yes"] * len(FEATURES) + ["no (analysis only)"]})


def overview_numbers(companies: pd.DataFrame, matches: pd.DataFrame) -> dict:
    return {
        "companies": len(companies),
        "positives": int(companies["label"].sum()),
        "positives_by_source": companies.loc[companies["label"] == 1, "label_source"]
        .value_counts().to_dict(),
        "matches": pd.crosstab(matches["list_source"], matches["tier"], margins=True,
                               margins_name="total"),
    }


def pick_cases(companies: pd.DataFrame, graph: nx.DiGraph) -> list:
    """The 4 Phase 5 cases as (title, company_number, one-line summary)."""
    cases = case_studies.select_cases(companies, graph)
    caruana = cases["caruana_group"]
    car_first = sorted(graph.successors(caruana))[0].removeprefix("company:")
    by = companies.set_index("company_number")

    def ranks(c):
        r = by.loc[c]
        return f"RF rank {int(r['rank_random_forest']):,}, LR rank {int(r['rank_logistic_regression']):,}"

    best = cases["best_positive"].removeprefix("company:")
    fp = cases["top_false_pos"].removeprefix("company:")
    typ = cases["typical_pos"].removeprefix("company:")
    return [
        ("1. The best-ranked risky company", best,
         f"The risky company the models ranked highest -- still only {ranks(best)}."),
        ("2. The highest-scoring false positive", fp,
         f"Random Forest's top company is not risky ({ranks(fp)}): it has exactly the same "
         "profile as Peter Caruana's risky companies."),
        ("3. A typical risky company", typ,
         f"Has the most common profile of all; indistinguishable from ordinary companies "
         f"({ranks(typ)})."),
        ("4. Peter Caruana's group", car_first,
         f"The one flagged owner with several companies ({graph.out_degree(caruana)}); his "
         f"companies rank near the bottom ({ranks(car_first)})."),
    ]
