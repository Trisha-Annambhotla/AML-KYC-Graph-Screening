"""
Phase 3: features and labels (docs/implementation_plan.md).

One row per company, read from the Phase 2 graph (data/processed/graph.pkl).

Labels
    label        1 if any PSC directly connected to the company is strongly
                 flagged.
    label_weak   1 if any connected PSC is strongly OR weakly flagged.
                 REPORTING ONLY -- never a training label: in Mode A most
                 weak matches are different people with the same name.
    label_source pep / sanctions / both, from the strongly flagged PSCs.

Features (the four model inputs, FEATURES below)
    flagged_neighbour_companies
        Number of OTHER companies with label = 1 linked to this company by
        (a) a shared non-flagged PSC, or
        (b) a shared address key: a non-flagged PSC of this company and a
            non-flagged PSC of the other company have the same address key.
        Flagged PSCs are left out on BOTH sides of both links -- the plan's
        "copy of the graph with all flagged PSC nodes removed". This way the
        PSC behind a company's own label can never produce its feature, even
        indirectly (e.g. a flagged person controlling two companies). A
        company linked by both (a) and (b) to the same labelled company counts
        it once. (Changed from the plan at the user's request: the plan only
        used (a).)
    shared_address_count
        Number of OTHER companies with at least one PSC at the same address
        key (all PSCs; empty keys ignored).

    Address cap: any address key shared by more than ADDRESS_KEY_MAX_COMPANIES
    (50) companies is ignored in both features above, and in split_group_id.
    Such keys are virtual offices / formation agents / accountants (e.g.
    "71-75 Shelton Street, WC2H 9JQ" with ~3,000 companies), not real links.
    A key's size is the number of distinct companies with any PSC there.
    degree
        Number of PSCs connected to the company (in-degree, full graph).
    component_size
        Number of nodes in the company's weakly connected component (full
        graph).

Extra columns (not model features)
    company_number, component_id (weakly connected component of the
    ownership graph), split_group_id (component_id groups merged further by
    capped shared address keys -- Phase 4 must use THIS for GroupKFold, so
    address-linked companies never land in different folds),
    flagged_neighbour_via_owner (the plan's original feature 1: link (a)
    only, kept so both versions can be reported), and is_hub_component (the
    component contains a PSC controlling 100+ companies, e.g. a formation
    agent; for analysis only).

Run directly:
    python src/features.py
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from datetime import date

import networkx as nx
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
GRAPH_INPUT = os.path.join(PROCESSED, "graph.pkl")
FEATURES_OUTPUT = os.path.join(PROCESSED, "features.csv")
STATS_OUTPUT = os.path.join(ROOT, "docs", "feature_stats.md")

HUB_MIN_COMPANIES = 100
ADDRESS_KEY_MAX_COMPANIES = 50

FEATURES = ["flagged_neighbour_companies", "shared_address_count", "degree", "component_size"]
OUTPUT_FIELDS = [
    "company_number", "component_id", "split_group_id", "label", "label_weak", "label_source",
    *FEATURES, "flagged_neighbour_via_owner", "is_hub_component",
]


def _source_label(sources) -> str:
    parts = set()
    for s in sources:
        parts |= {"pep", "sanctions"} if s == "both" else ({s} if s else set())
    if not parts:
        return ""
    return "both" if len(parts) > 1 else parts.pop()


def _components(graph: nx.DiGraph):
    """Component id and size per node; ids are stable (ordered by smallest node id)."""
    comps = sorted(nx.weakly_connected_components(graph), key=min)
    comp_id, comp_size, hub_comps = {}, {}, set()
    for cid, comp in enumerate(comps):
        for n in comp:
            comp_id[n] = cid
        comp_size[cid] = len(comp)
        if any(graph.out_degree(n) >= HUB_MIN_COMPANIES
               for n in comp if graph.nodes[n]["node_type"] != "company"):
            hub_comps.add(cid)
    return comp_id, comp_size, hub_comps


def _split_groups(comp_of_company: dict, key_all: dict) -> dict:
    """
    Union-find over component ids: components whose companies share a
    (capped) address key are merged. Returns company -> split_group_id, with
    ids numbered in order of the smallest component id in each group.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for companies in key_all.values():
        comps = {comp_of_company[c] for c in companies}
        if len(comps) > 1:
            first, *rest = comps
            root = find(first)
            for other in rest:
                r = find(other)
                if r != root:
                    if r < root:
                        root, r = r, root
                    parent[r] = root

    roots = {c: find(cid) for c, cid in comp_of_company.items()}
    renumber = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {c: renumber[r] for c, r in roots.items()}


def compute_features(graph: nx.DiGraph):
    """Returns (features DataFrame, info dict about the address cap and split groups)."""
    nodes = graph.nodes
    companies = [n for n, d in nodes(data=True) if d["node_type"] == "company"]
    is_flagged = {n: bool(d.get("flagged")) for n, d in nodes(data=True)}

    # Labels
    label, label_weak, label_source = {}, {}, {}
    for c in companies:
        pscs = list(graph.predecessors(c))
        strong = [p for p in pscs if is_flagged[p]]
        label[c] = int(bool(strong))
        label_weak[c] = int(bool(strong) or any(nodes[p].get("weak_flagged") for p in pscs))
        label_source[c] = _source_label(nodes[p]["flag_source"] for p in strong)
    positives = [c for c in companies if label[c]]

    # Address keys per company: all PSCs, and non-flagged PSCs only
    key_all, key_nf = defaultdict(set), defaultdict(set)
    company_keys_all, company_keys_nf = defaultdict(set), defaultdict(set)
    for p, c, key in graph.edges(data="address_key"):
        if not key:
            continue
        key_all[key].add(c)
        company_keys_all[c].add(key)
        if not is_flagged[p]:
            key_nf[key].add(c)
            company_keys_nf[c].add(key)

    # Address cap: drop keys shared by too many companies, everywhere below
    capped = {k for k, cos in key_all.items() if len(cos) > ADDRESS_KEY_MAX_COMPANIES}
    info = {
        "address_cap": ADDRESS_KEY_MAX_COMPANIES,
        "n_address_keys": len(key_all),
        "n_capped_keys": len(capped),
        "companies_at_capped_keys": len(set().union(*(key_all[k] for k in capped))) if capped else 0,
        "largest_capped_keys": sorted(((len(key_all[k]), k) for k in capped), reverse=True)[:5],
    }
    for k in capped:
        del key_all[k]
        key_nf.pop(k, None)
    for keys in (company_keys_all, company_keys_nf):
        for c in keys:
            keys[c] -= capped

    # Feature 1, both versions: walk out from each labelled company
    via_owner, via_any = defaultdict(set), defaultdict(set)
    for pos in positives:
        for p in graph.predecessors(pos):
            if is_flagged[p]:
                continue
            for other in graph.successors(p):
                if other != pos:
                    via_owner[other].add(pos)
                    via_any[other].add(pos)
        for key in company_keys_nf[pos]:
            for other in key_nf[key]:
                if other != pos:
                    via_any[other].add(pos)

    # Feature 2
    def shared_address_count(c):
        keys = company_keys_all.get(c)
        if not keys:
            return 0
        if len(keys) == 1:
            return len(key_all[next(iter(keys))]) - 1
        return len(set().union(*(key_all[k] for k in keys))) - 1

    comp_id, comp_size, hub_comps = _components(graph)
    split_group = _split_groups({c: comp_id[c] for c in companies}, key_all)

    rows = [{
        "company_number": c.removeprefix("company:"),
        "component_id": comp_id[c],
        "split_group_id": split_group[c],
        "label": label[c],
        "label_weak": label_weak[c],
        "label_source": label_source[c],
        "flagged_neighbour_companies": len(via_any.get(c, ())),
        "shared_address_count": shared_address_count(c),
        "degree": graph.in_degree(c),
        "component_size": comp_size[comp_id[c]],
        "flagged_neighbour_via_owner": len(via_owner.get(c, ())),
        "is_hub_component": comp_id[c] in hub_comps,
    } for c in companies]
    df = pd.DataFrame(rows, columns=OUTPUT_FIELDS).sort_values("company_number", ignore_index=True)
    return df, info


# ---------------------------------------------------------------------------
# Checkpoint 3 summary
# ---------------------------------------------------------------------------

def feature_summary(df: pd.DataFrame, info: dict = None) -> dict:
    n, pos = len(df), int(df["label"].sum())
    cols = FEATURES + ["flagged_neighbour_via_owner"]
    # index: (feature, statistic), columns: label 0 / 1
    by_label = df.groupby("label")[cols].describe(percentiles=[0.5, 0.9, 0.99]).T
    corr = {c: df[c].corr(df["label"]) for c in cols}
    # Spearman = Pearson on ranks (avoids a scipy dependency)
    corr_spearman = {c: df[c].rank().corr(df["label"].rank()) for c in cols}
    nonzero = {
        c: {"companies": int((df[c] > 0).sum()),
            "positives": int(((df[c] > 0) & (df["label"] == 1)).sum())}
        for c in ("flagged_neighbour_companies", "flagged_neighbour_via_owner")
    }
    return {
        "n_companies": n, "n_positive": pos, "pct_positive": 100 * pos / n,
        "n_positive_weak": int(df["label_weak"].sum()),
        "label_source": df.loc[df["label"] == 1, "label_source"].value_counts().to_dict(),
        "by_label": by_label, "corr": corr, "corr_spearman": corr_spearman,
        "nonzero": nonzero,
        "hub_companies": int(df["is_hub_component"].sum()),
        "hub_positives": int((df["is_hub_component"] & (df["label"] == 1)).sum()),
        "n_components": int(df["component_id"].nunique()),
        "n_split_groups": int(df["split_group_id"].nunique()),
        "largest_split_group": int(df["split_group_id"].value_counts().max()),
        "positive_split_groups": int(df.loc[df["label"] == 1, "split_group_id"].nunique()),
        "info": info or {},
    }


def _cap_lines(s: dict) -> list:
    info = s.get("info") or {}
    if not info:
        return ["- (no cap information)"]
    top = "; ".join(f"`{k}` ({n:,})" for n, k in info["largest_capped_keys"])
    return [
        f"- Address keys shared by more than **{info['address_cap']} companies** are "
        "ignored in `flagged_neighbour_companies`, `shared_address_count` and "
        "`split_group_id`. They are virtual offices, formation agents and "
        "accountants, not real links.",
        f"- Capped keys: {info['n_capped_keys']:,} of {info['n_address_keys']:,}, "
        f"covering {info['companies_at_capped_keys']:,} companies",
        f"- Largest capped keys: {top}",
    ]


def write_summary(s: dict, path: str = STATS_OUTPUT):
    def g(x):
        return f"{x:,.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else f"{x:,}"

    lines = [
        "# Feature statistics (Phase 3, Checkpoint 3)",
        "",
        f"Generated {date.today().isoformat()} by `src/features.py` from "
        "`data/processed/graph.pkl`.",
        "",
        "## Companies and labels",
        "",
        f"- Companies: {s['n_companies']:,} in {s['n_components']:,} components",
        f"- Positives (`label = 1`): {s['n_positive']} ({s['pct_positive']:.4f}%)",
        f"- By source: {', '.join(f'{k}: {v}' for k, v in s['label_source'].items())}",
        f"- `label_weak = 1` (strong OR weak flags): {s['n_positive_weak']:,}. "
        "**Reporting only, never a training label**: in Mode A most weak "
        "matches are different people with the same name.",
        f"- Companies in a hub component (PSC with {HUB_MIN_COMPANIES}+ companies): "
        f"{s['hub_companies']:,}, of which positive: {s['hub_positives']}",
        "",
        "## Address-key cap",
        "",
        *_cap_lines(s),
        "",
        "## Train/test split groups",
        "",
        f"- `split_group_id` = ownership-graph components merged by shared "
        f"(capped) address keys: {s['n_split_groups']:,} groups "
        f"(vs {s['n_components']:,} components)",
        f"- Largest group: {s['largest_split_group']:,} companies; the "
        f"{s['n_positive']} positives sit in {s['positive_split_groups']} groups",
        "- Phase 4 must use `split_group_id` (not `component_id`) for "
        "GroupKFold, so address-linked companies never land in different folds.",
        "",
        "## Feature 1: companies with a flagged neighbour > 0",
        "",
        "| Version | Companies > 0 | Positives > 0 |",
        "|---|---|---|",
        f"| `flagged_neighbour_companies` (owner OR address) | "
        f"{s['nonzero']['flagged_neighbour_companies']['companies']:,} | "
        f"{s['nonzero']['flagged_neighbour_companies']['positives']} |",
        f"| `flagged_neighbour_via_owner` (plan's original) | "
        f"{s['nonzero']['flagged_neighbour_via_owner']['companies']:,} | "
        f"{s['nonzero']['flagged_neighbour_via_owner']['positives']} |",
        "",
        "## Feature summary by label",
        "",
    ]
    by = s["by_label"]
    stats = ["mean", "std", "min", "50%", "90%", "99%", "max"]
    lines.append("| Feature | label | " + " | ".join(stats) + " |")
    lines.append("|---|---|" + "---|" * len(stats))
    for feat in by.index.get_level_values(0).unique():
        for lab in (0, 1):
            vals = [g(float(by.loc[(feat, st), lab])) for st in stats]
            lines.append(f"| {feat} | {lab} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Correlation with label (leakage check)",
        "",
        "| Feature | Pearson | Spearman |",
        "|---|---|---|",
        *[f"| {c} | {s['corr'][c]:.4f} | {s['corr_spearman'][c]:.4f} |" for c in s["corr"]],
        "",
        ("No feature is perfectly correlated with the label (all |r| < 0.99)."
         if all(abs(v) < 0.99 for v in list(s["corr"].values()) + list(s["corr_spearman"].values()))
         else "**WARNING: a feature is (almost) perfectly correlated with the label.**"),
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def run(graph_path=GRAPH_INPUT, output_path=FEATURES_OUTPUT, stats_path=STATS_OUTPUT):
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    df, info = compute_features(graph)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    summary = feature_summary(df, info)
    write_summary(summary, stats_path)
    return df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=GRAPH_INPUT)
    ap.add_argument("--output", default=FEATURES_OUTPUT)
    args = ap.parse_args()

    _, summ = run(args.graph, args.output)
    for k in ("n_companies", "n_positive", "pct_positive", "n_positive_weak",
              "label_source", "nonzero", "hub_companies", "hub_positives",
              "n_split_groups", "largest_split_group", "positive_split_groups", "corr"):
        print(f"  {k}: {summ[k]}", file=sys.stderr)
    print(f"Features written to: {args.output}", file=sys.stderr)
    print(f"Stats written to: {STATS_OUTPUT}", file=sys.stderr)
