"""
Phase 2: build the ownership graph (docs/implementation_plan.md).

Turns PSC rows into a directed graph of who controls what:

Nodes
    company:<company_number>            one per company
    person:<normalized_name>|<postcode> individual PSCs. The same name at the
                                        same postcode is merged into one node
                                        across all the companies it controls.
                                        No postcode -> normalised full address.
    corp:<normalized_name>              corporate PSCs. legal-person PSCs
                                        (e.g. government bodies; skipped in
                                        matching) are also corp nodes, so a
                                        company's degree counts all its PSCs.
Edges
    PSC node -> company node, with natures_of_control and the psc_row_ids
    behind the edge. A PSC listed more than once for the same company becomes
    one edge (natures of control merged).
Flags
    Strong matches from data/processed/matches.csv are mapped to nodes through
    psc_row_id: flagged=True and flag_source = pep / sanctions / both.
    Weak matches are kept separately in weak_flagged / weak_flag_source.
    Each match is checked against the PSC row it points at (same company
    number and name), so a matches.csv that is stale relative to
    psc_clean.csv is reported instead of silently flagging the wrong node.
Address key
    Per PSC row: postcode + first address line, lowercased with spaces
    collapsed (normalised full address if there is no postcode). Stored on
    edges and on PSC nodes (most common key across the node's rows). No
    address nodes are added to the graph.

Outputs: data/processed/graph.pkl (pickle; write_gpickle is gone in
networkx 3), data/processed/nodes.csv, data/processed/edges.csv, and the
Checkpoint 2 summary docs/graph_stats.md.

Run directly:
    python src/graph_builder.py
"""

import argparse
import os
import pickle
import re
import sys
from datetime import date

import networkx as nx
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERIM = os.path.join(ROOT, "data", "interim")
PROCESSED = os.path.join(ROOT, "data", "processed")

PSC_INPUT = os.path.join(INTERIM, "psc_clean.csv")
MATCHES_INPUT = os.path.join(PROCESSED, "matches.csv")
GRAPH_OUTPUT = os.path.join(PROCESSED, "graph.pkl")
NODES_OUTPUT = os.path.join(PROCESSED, "nodes.csv")
EDGES_OUTPUT = os.path.join(PROCESSED, "edges.csv")
STATS_OUTPUT = os.path.join(ROOT, "docs", "graph_stats.md")

PSC_COLS = ["company_number", "kind", "name", "normalized_name", "address",
            "natures_of_control"]

NODE_FIELDS = ["node_id", "node_type", "name", "kind", "postcode", "address_key",
               "n_psc_rows", "flagged", "flag_source", "weak_flagged", "weak_flag_source"]
EDGE_FIELDS = ["source", "target", "natures_of_control", "address_key", "psc_row_ids"]

# UK postcode: outward code (A9, A99, A9A, AA9, AA99, AA9A) + inward code (9AA).
_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})\b", re.IGNORECASE)
# A first segment that is only a house number, e.g. "3" in "3, Vicarage Crescent".
_HOUSE_NUMBER_RE = re.compile(r"^\d+[a-z]?(-\d+[a-z]?)?$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Address and node-ID helpers (unit-tested)
# ---------------------------------------------------------------------------

def extract_postcode(address: str) -> str:
    """Last UK postcode in the address, as 'GL51 0TJ'; '' if none."""
    found = _POSTCODE_RE.findall(address or "")
    if not found:
        return ""
    outward, inward = found[-1]
    return f"{outward.upper()} {inward.upper()}"


def normalize_address(address: str) -> str:
    return " ".join((address or "").lower().split())


def first_address_line(address: str) -> str:
    """First comma segment; a bare house number is joined to the street."""
    parts = [p.strip() for p in (address or "").split(",") if p.strip()]
    if not parts:
        return ""
    if _HOUSE_NUMBER_RE.match(parts[0]) and len(parts) > 1:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def address_key(address: str) -> str:
    postcode = extract_postcode(address)
    if postcode:
        return normalize_address(f"{postcode} | {first_address_line(address)}")
    return normalize_address(address)


def company_node_id(company_number: str) -> str:
    return f"company:{company_number}"


def psc_node_id(kind: str, normalized_name: str, address: str) -> str:
    if kind.startswith("individual"):
        where = extract_postcode(address) or normalize_address(address)
        return f"person:{normalized_name}|{where}"
    return f"corp:{normalized_name}"


def _psc_node_type(kind: str) -> str:
    return "person" if kind.startswith("individual") else "corp"


def _source_label(sources) -> str:
    sources = set(sources)
    if not sources:
        return ""
    return "both" if len(sources) > 1 else sources.pop()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _merge_natures(values) -> str:
    return ";".join(sorted({n for v in values for n in v.split(";") if n}))


def _edges_frame(psc: pd.DataFrame) -> pd.DataFrame:
    """One row per (PSC node, company node); duplicates merged."""
    keys = ["psc_node", "company_node"]
    dup = psc.duplicated(keys, keep=False)
    single = psc.loc[~dup, keys + ["natures_of_control", "address_key", "psc_row_id"]].copy()
    single["psc_row_ids"] = single.pop("psc_row_id").astype(str)
    merged = (
        psc.loc[dup]
        .groupby(keys, sort=False)
        .agg(natures_of_control=("natures_of_control", _merge_natures),
             address_key=("address_key", "first"),
             psc_row_ids=("psc_row_id", lambda s: ";".join(map(str, s))))
        .reset_index()
    )
    edges = pd.concat([single, merged], ignore_index=True)
    return edges.rename(columns={"psc_node": "source", "company_node": "target"})[EDGE_FIELDS]


def _psc_nodes_frame(psc: pd.DataFrame) -> pd.DataFrame:
    first = psc.drop_duplicates("psc_node").set_index("psc_node")
    common_key = (psc.groupby(["psc_node", "address_key"]).size()
                  .reset_index(name="n")
                  .sort_values(["psc_node", "n"], ascending=[True, False])
                  .drop_duplicates("psc_node")
                  .set_index("psc_node")["address_key"])
    nodes = pd.DataFrame({
        "node_id": first.index,
        "node_type": first["kind"].map(_psc_node_type).values,
        "name": first["name"].values,
        "kind": first["kind"].values,
        "postcode": first["address"].map(extract_postcode).values,
        "address_key": common_key.reindex(first.index).values,
        "n_psc_rows": psc["psc_node"].value_counts().reindex(first.index).values,
    })
    nodes.loc[nodes["node_type"] == "corp", "postcode"] = ""
    return nodes


def _map_flags(psc: pd.DataFrame, matches: pd.DataFrame):
    """
    Returns (flags DataFrame indexed by psc node, unmapped strong matches).
    A match is mapped only if its psc_row_id exists and the PSC row there has
    the same company number and name as the match.
    """
    rows = psc.set_index("psc_row_id")[["company_number", "name", "psc_node"]]
    m = matches.join(rows, on="psc_row_id", rsuffix="_psc")
    ok = (m["psc_node"].notna()
          & (m["company_number"] == m["company_number_psc"])
          & (m["psc_name"] == m["name"]))
    unmapped = m.loc[~ok & (m["tier"] == "strong"),
                     ["psc_row_id", "company_number", "psc_name", "list_source", "list_name"]]
    m = m[ok]

    def per_node(tier):
        sub = m[m["tier"] == tier]
        return sub.groupby("psc_node")["list_source"].agg(_source_label)

    flags = pd.DataFrame({"flag_source": per_node("strong"),
                          "weak_flag_source": per_node("weak")}).fillna("")
    flags["flagged"] = flags["flag_source"] != ""
    flags["weak_flagged"] = flags["weak_flag_source"] != ""
    return flags, unmapped.reset_index(drop=True)


def build_graph(psc: pd.DataFrame, matches: pd.DataFrame):
    """
    psc: PSC rows with psc_row_id + PSC_COLS. matches: Phase 1 output.
    Returns (graph, nodes DataFrame, edges DataFrame, unmapped strong matches).
    """
    psc = psc[psc["normalized_name"] != ""].copy()
    psc["psc_node"] = [psc_node_id(k, n, a) for k, n, a in
                       zip(psc["kind"], psc["normalized_name"], psc["address"])]
    psc["company_node"] = psc["company_number"].map(company_node_id)
    psc["address_key"] = psc["address"].map(address_key)

    edges = _edges_frame(psc)

    companies = pd.DataFrame({"node_id": psc["company_node"].unique()})
    companies["node_type"] = "company"
    companies["name"] = companies["node_id"].str.removeprefix("company:")
    nodes = pd.concat([companies, _psc_nodes_frame(psc)], ignore_index=True)

    flags, unmapped = _map_flags(psc, matches)
    nodes = nodes.join(flags, on="node_id")
    for col in ("flag_source", "weak_flag_source", "kind", "postcode", "address_key"):
        nodes[col] = nodes[col].fillna("")
    for col in ("flagged", "weak_flagged"):
        nodes[col] = nodes[col].fillna(False).astype(bool)
    nodes["n_psc_rows"] = nodes["n_psc_rows"].fillna(0).astype(int)
    nodes = nodes[NODE_FIELDS]

    graph = nx.DiGraph()
    graph.add_nodes_from(
        (r["node_id"], {k: r[k] for k in NODE_FIELDS if k != "node_id"})
        for r in nodes.to_dict("records")
    )
    graph.add_edges_from(
        (r["source"], r["target"], {k: r[k] for k in EDGE_FIELDS[2:]})
        for r in edges.to_dict("records")
    )
    return graph, nodes, edges, unmapped


# ---------------------------------------------------------------------------
# Checkpoint 2 summary
# ---------------------------------------------------------------------------

def graph_stats(graph: nx.DiGraph, nodes: pd.DataFrame, n_strong: int,
                unmapped: pd.DataFrame) -> dict:
    sizes = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    sizes_s = pd.Series(sizes)
    psc_nodes = nodes[nodes["node_type"] != "company"]
    people = nodes[nodes["node_type"] == "person"]
    return {
        "nodes_by_type": nodes["node_type"].value_counts().to_dict(),
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": len(sizes),
        "size_min": int(sizes_s.min()),
        "size_median": float(sizes_s.median()),
        "size_max": int(sizes_s.max()),
        "top10_sizes": sizes[:10],
        "size_counts": sizes_s.value_counts().sort_index().head(6).to_dict(),
        "people_multi_company": sum(graph.out_degree(n) >= 2 for n in people["node_id"]),
        "people_without_postcode": int((people["postcode"] == "").sum()),
        "flagged_nodes": int(psc_nodes["flagged"].sum()),
        "flagged_by_source": psc_nodes.loc[psc_nodes["flagged"], "flag_source"].value_counts().to_dict(),
        "weak_flagged_nodes": int(psc_nodes["weak_flagged"].sum()),
        "n_strong_matches": n_strong,
        "unmapped": unmapped,
    }


def write_stats(stats: dict, path: str = STATS_OUTPUT):
    by_type = stats["nodes_by_type"]
    unmapped = stats["unmapped"]
    lines = [
        "# Graph statistics (Phase 2, Checkpoint 2)",
        "",
        f"Generated {date.today().isoformat()} by `src/graph_builder.py` from "
        "`data/interim/psc_clean.csv` and `data/processed/matches.csv`.",
        "",
        "## Size",
        "",
        "| Node type | Count |",
        "|---|---|",
        *[f"| {t} | {by_type.get(t, 0):,} |" for t in ("company", "person", "corp")],
        f"| **total nodes** | **{stats['n_nodes']:,}** |",
        f"| **edges** (PSC -> company) | **{stats['n_edges']:,}** |",
        "",
        f"- People who control 2+ companies (merged by name + postcode): "
        f"{stats['people_multi_company']:,}",
        f"- Person nodes with no postcode (keyed by full address instead): "
        f"{stats['people_without_postcode']:,}",
        "",
        "## Connected components (weakly connected)",
        "",
        f"- Number of components: {stats['n_components']:,}",
        f"- Size (nodes): min {stats['size_min']}, median {stats['size_median']:g}, "
        f"max {stats['size_max']:,}",
        f"- Top 10 sizes: {', '.join(f'{s:,}' for s in stats['top10_sizes'])}",
        "- Components by size (smallest sizes): "
        + ", ".join(f"{k} nodes: {v:,}" for k, v in stats["size_counts"].items()),
        "",
        "## Flags",
        "",
        f"- Strongly flagged PSC nodes: {stats['flagged_nodes']} "
        f"({', '.join(f'{k}: {v}' for k, v in stats['flagged_by_source'].items()) or 'none'})",
        f"- Weakly flagged PSC nodes: {stats['weak_flagged_nodes']:,}",
        "",
        "## Strong-match mapping check",
        "",
    ]
    if unmapped.empty:
        lines.append(f"All {stats['n_strong_matches']} strong matches were mapped to a node.")
    else:
        lines += [
            f"**{len(unmapped)} of {stats['n_strong_matches']} strong matches were NOT "
            "mapped** (psc_row_id missing, or company/name differ from psc_clean.csv -- "
            "rerun Phase 1 if the PSC file changed):",
            "",
            "| psc_row_id | company_number | psc_name | list_source | list_name |",
            "|---|---|---|---|---|",
            *[f"| {r.psc_row_id} | {r.company_number} | {r.psc_name} | {r.list_source} "
              f"| {r.list_name} |" for r in unmapped.itertuples()],
        ]
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_psc(path: str = PSC_INPUT) -> pd.DataFrame:
    psc = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=PSC_COLS)
    psc.insert(0, "psc_row_id", range(len(psc)))  # same row positions as Phase 1
    return psc


def load_matches(path: str = MATCHES_INPUT) -> pd.DataFrame:
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    m["psc_row_id"] = m["psc_row_id"].astype(int)
    return m


def run(psc_path=PSC_INPUT, matches_path=MATCHES_INPUT, graph_path=GRAPH_OUTPUT,
        nodes_path=NODES_OUTPUT, edges_path=EDGES_OUTPUT, stats_path=STATS_OUTPUT):
    psc = load_psc(psc_path)
    matches = load_matches(matches_path)
    graph, nodes, edges, unmapped = build_graph(psc, matches)

    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    stats = graph_stats(graph, nodes, int((matches["tier"] == "strong").sum()), unmapped)
    write_stats(stats, stats_path)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--psc", default=PSC_INPUT)
    ap.add_argument("--matches", default=MATCHES_INPUT)
    args = ap.parse_args()

    result = run(args.psc, args.matches)
    for k, v in result.items():
        if k != "unmapped":
            print(f"  {k}: {v}", file=sys.stderr)
    print(f"  unmapped strong matches: {len(result['unmapped'])}", file=sys.stderr)
    print(f"Graph written to: {GRAPH_OUTPUT}", file=sys.stderr)
    print(f"Stats written to: {STATS_OUTPUT}", file=sys.stderr)
