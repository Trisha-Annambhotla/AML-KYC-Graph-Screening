"""
Phase 5: case studies (docs/implementation_plan.md, updated after Checkpoint 4).

No risky company ranked highly in Phase 4, so the cases explain why the models
failed rather than show successes:

    best_positive   the best-ranked risky company (by Random Forest)
    top_false_pos   the highest-scoring company that is not risky
    typical_pos     the risky company whose rank is closest to the median
                    rank of all risky companies
    caruana_group   the one flagged owner who controls several companies
                    (Peter Caruana) -- where feature 1 fires

For each case: a drawing of the 2-hop neighbourhood (results/case_*.png),
the feature values with their percentiles among all companies, the model
scores and ranks, and a plain-English explanation built from the data. The
key test is the "feature twins" count: how many companies have exactly the
same four feature values, and how many of those are risky. If a risky company
has thousands of non-risky twins, no model using these features can rank it
above them.

Plus a Random Forest vs Logistic Regression comparison
(results/case_model_comparison.png and a section in docs/case_studies.md).

Ranks use the same fixed tie-break order as Phase 4 (train.rank_order), so
they match the Phase 4 metrics exactly.

Run directly:
    python src/case_studies.py
"""

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import train  # noqa: E402
from features import ADDRESS_KEY_MAX_COMPANIES, FEATURES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
GRAPH_INPUT = os.path.join(PROCESSED, "graph.pkl")
FEATURES_INPUT = os.path.join(PROCESSED, "features.csv")
PREDICTIONS_INPUT = os.path.join(PROCESSED, "predictions.csv")
METRICS_INPUT = os.path.join(ROOT, "results", "metrics.csv")
RESULTS = os.path.join(ROOT, "results")
DOC_OUTPUT = os.path.join(ROOT, "docs", "case_studies.md")

RANK_METHODS = ["random_forest", "logistic_regression", "rule_address", "random"]
PRIMARY = "random_forest"
MAX_PER_NODE = 8  # neighbours drawn per node before folding into "+N more"
RECALL_KS = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000]

# Reference palette (dataviz skill), light mode
C_COMPANY, C_PERSON, C_CORP = "#2a78d6", "#eb6834", "#1baf7a"
C_OTHER, C_FLAG = "#8f8e89", "#e34948"
INK, INK_MUTED, GRID, SURFACE = train.INK, train.INK_MUTED, train.GRID, train.SURFACE

CASE_TITLES = {
    "best_positive": "1. The best-ranked risky company",
    "top_false_pos": "2. The highest-scoring false positive",
    "typical_pos": "3. A typical risky company",
    "caruana_group": "4. Peter Caruana's group",
}


# ---------------------------------------------------------------------------
# Data and ranks
# ---------------------------------------------------------------------------

def load_table(features_path=FEATURES_INPUT, predictions_path=PREDICTIONS_INPUT) -> pd.DataFrame:
    feats = train.load_features(features_path)
    preds = pd.read_csv(predictions_path, dtype={"company_number": str},
                        usecols=["company_number", "fold"]
                        + [f"score_{m}" for m in RANK_METHODS])
    df = feats.merge(preds, on="company_number", how="left", validate="one_to_one")
    return add_ranks(df)


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """rank_<method>: 1 = highest score. Same tie-break as Phase 4."""
    tiebreak = np.random.default_rng(train.RANDOM_STATE + 1).random(len(df))
    for m in RANK_METHODS:
        order = train.rank_order(df[f"score_{m}"].to_numpy(float), tiebreak)
        rank = np.empty(len(df), dtype=int)
        rank[order] = np.arange(1, len(df) + 1)
        df[f"rank_{m}"] = rank
    return df


def percentile(df: pd.DataFrame, col: str, value) -> float:
    """Share of companies (in %) with a strictly smaller value."""
    return 100.0 * (df[col] < value).mean()


def twin_mask(df: pd.DataFrame, row: pd.Series) -> np.ndarray:
    same = np.ones(len(df), dtype=bool)
    for f in FEATURES:
        same &= df[f].to_numpy() == row[f]
    return same


def feature_twins(df: pd.DataFrame, row: pd.Series) -> dict:
    """Companies with exactly the same four feature values (including itself).
    train_*: the twins in OTHER folds, i.e. in the training data when this
    company was scored out-of-fold."""
    twins = df[twin_mask(df, row)]
    out = {"n": int(len(twins)), "positives": int(twins["label"].sum()),
           "rate": float(twins["label"].mean())}
    if "fold" in df:
        tr = twins[twins["fold"] != row["fold"]]
        out["train_n"] = int(len(tr))
        out["train_positives"] = int(tr["label"].sum())
    return out


def risky_twin_owners(df: pd.DataFrame, row: pd.Series, graph) -> Counter:
    """Flagged owners of the risky twins (who the model may be memorising)."""
    twins = df[twin_mask(df, row) & (df["label"] == 1).to_numpy()]
    owners = Counter()
    for c in twins["company_number"]:
        for p in graph.predecessors("company:" + c):
            if graph.nodes[p]["flagged"]:
                owners[graph.nodes[p]["name"]] += 1
    return owners


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def find_caruana(graph: nx.DiGraph) -> str:
    """The strongly flagged PSC node that controls the most companies."""
    flagged = [n for n, d in graph.nodes(data=True)
               if d.get("flagged") and d["node_type"] != "company"]
    return max(flagged, key=lambda n: (graph.out_degree(n), n))


def select_cases(df: pd.DataFrame, graph: nx.DiGraph) -> dict:
    r = f"rank_{PRIMARY}"
    pos = df[df["label"] == 1].sort_values(r)
    neg = df[df["label"] == 0].sort_values(r)
    best = pos.iloc[0]
    rest = pos.iloc[1:]
    median_rank = pos[r].median()
    typical = rest.iloc[(rest[r] - median_rank).abs().argmin()]
    return {
        "best_positive": "company:" + best["company_number"],
        "top_false_pos": "company:" + neg.iloc[0]["company_number"],
        "typical_pos": "company:" + typical["company_number"],
        "caruana_group": find_caruana(graph),
    }


# ---------------------------------------------------------------------------
# Feature 1 links (why a company has flagged_neighbour_companies > 0)
# ---------------------------------------------------------------------------

def address_key_sizes(graph: nx.DiGraph) -> dict:
    companies = defaultdict(set)
    for _, c, key in graph.edges(data="address_key"):
        if key:
            companies[key].add(c)
    return {k: len(v) for k, v in companies.items()}


def feature1_links(graph, company, positives: set, key_sizes: dict) -> list:
    """(positive company, how) pairs, mirroring features.compute_features."""
    def nf_pscs(c):
        return [p for p in graph.predecessors(c) if not graph.nodes[p]["flagged"]]

    def nf_keys(c):
        return {graph.edges[p, c]["address_key"] for p in nf_pscs(c)
                if graph.edges[p, c]["address_key"]
                and key_sizes[graph.edges[p, c]["address_key"]] <= ADDRESS_KEY_MAX_COMPANIES}

    how = defaultdict(list)
    my_keys = nf_keys(company)
    for p in nf_pscs(company):
        for other in graph.successors(p):
            if other != company and other in positives:
                how[other].append(f"shared owner {graph.nodes[p]['name']}")
    for other in positives:
        if other != company and my_keys & nf_keys(other):
            how[other].append(f"shared address '{sorted(my_keys & nf_keys(other))[0]}'")
    return [(other, " + ".join(reasons)) for other, reasons in sorted(how.items())]


# ---------------------------------------------------------------------------
# Neighbourhood drawing
# ---------------------------------------------------------------------------

def neighbourhood(graph: nx.DiGraph, center: str, radius: int, important: set,
                  max_per_node: int = MAX_PER_NODE):
    """
    BFS on the undirected graph up to `radius` hops. Each node shows at most
    `max_per_node` new neighbours (important nodes first); the rest fold into
    one "+N more" node. Returns (nodes, edges, more) where more maps a summary
    node id -> (parent, count).
    """
    und = graph.to_undirected(as_view=True)
    nodes, edges, more = {center}, set(), {}
    frontier = [center]
    for _ in range(radius):
        nxt = []
        for n in frontier:
            new = sorted((m for m in und.neighbors(n) if m not in nodes),
                         key=lambda m: (m not in important, m))
            for m in new[:max_per_node]:
                nodes.add(m)
                nxt.append(m)
                edges.add((n, m))
            if len(new) > max_per_node:
                sid = f"more:{n}"
                more[sid] = (n, len(new) - max_per_node)
            for m in und.neighbors(n):  # edges among already-shown nodes
                if m in nodes:
                    edges.add((n, m))
        frontier = nxt
    return nodes, edges, more


def _short(text: str, n: int = 22) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def draw_case(graph, df_by_node, center, radius, title, path, important, extra_labels,
              address_nodes=()):
    """address_nodes: (label, [psc nodes]) pairs drawn as grey squares, dashed."""
    nodes, edges, more = neighbourhood(graph, center, radius, important)
    g = nx.Graph()
    # sorted: set order varies between runs, and spring_layout depends on it
    g.add_nodes_from(sorted(nodes))
    g.add_edges_from(sorted(edges))
    for sid, (parent, _) in more.items():
        g.add_edge(sid, parent)
    for i, (_, pscs) in enumerate(address_nodes):
        for p in pscs:
            if p in g:
                g.add_edge(f"addr:{i}", p)
    pos = nx.spring_layout(g, seed=train.RANDOM_STATE, k=1.4 / np.sqrt(max(len(g), 1)),
                           iterations=200)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    solid = [(u, v) for u, v in g.edges if not (str(u).startswith("addr:") or str(v).startswith("addr:"))]
    dashed = [(u, v) for u, v in g.edges if (u, v) not in set(solid)]
    nx.draw_networkx_edges(g, pos, edgelist=solid, ax=ax, edge_color="#bdbcb6", width=1)
    nx.draw_networkx_edges(g, pos, edgelist=dashed, ax=ax, edge_color=C_OTHER, width=1,
                           style="dashed")

    def kind(n):
        n = str(n)
        if n.startswith(("more:", "addr:")):
            return "other"
        return graph.nodes[n]["node_type"]

    shapes = {"company": ("s", C_COMPANY), "person": ("o", C_PERSON),
              "corp": ("D", C_CORP), "other": ("s", C_OTHER)}
    for k, (marker, color) in shapes.items():
        ns = [n for n in g if kind(n) == k]
        if not ns:
            continue
        risky = [n for n in ns if k != "other" and (graph.nodes[n].get("flagged")
                 or df_by_node.get(n, {}).get("label") == 1)]
        nx.draw_networkx_nodes(g, pos, nodelist=ns, node_shape=marker, node_color=color,
                               node_size=[520 if n == center else 260 for n in ns],
                               edgecolors=[C_FLAG if n in risky else ("#0b0b0b" if n == center
                                           else SURFACE) for n in ns],
                               linewidths=[3 if (n in risky or n == center) else 1 for n in ns],
                               ax=ax)

    labels = {}
    for n in g:
        n_s = str(n)
        if n_s.startswith("more:"):
            labels[n] = f"+{more[n_s][1]:,} more"
        elif n_s.startswith("addr:"):
            labels[n] = address_nodes[int(n_s.split(":")[1])][0]
        elif kind(n) == "company":
            labels[n] = f"Co {n_s.removeprefix('company:')}"
        else:
            labels[n] = _short(graph.nodes[n]["name"])
        if graph.has_node(n) and graph.nodes[n].get("flagged"):
            labels[n] += f"\nFLAGGED ({graph.nodes[n]['flag_source'].upper()})"
        elif df_by_node.get(n, {}).get("label") == 1:
            labels[n] += "\nrisky (label 1)"
        if n in extra_labels:
            labels[n] += "\n" + extra_labels[n]
    for n, (x, y) in pos.items():
        ax.text(x, y - 0.045, labels[n], fontsize=7, ha="center", va="top", color=INK,
                fontweight="bold" if n == center else "normal")

    legend = [
        Line2D([], [], marker="s", ls="", color=C_COMPANY, markersize=9, label="Company"),
        Line2D([], [], marker="o", ls="", color=C_PERSON, markersize=9, label="Person (PSC)"),
        Line2D([], [], marker="D", ls="", color=C_CORP, markersize=8, label="Corporate PSC"),
        Line2D([], [], marker="s", ls="", color=C_OTHER, markersize=9,
               label="Address / '+N more' summary"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=C_FLAG,
               markeredgewidth=3, markersize=10, label="Red ring: flagged owner or risky company"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#0b0b0b",
               markeredgewidth=3, markersize=13,
               label="Large node, bold label: the case itself (black ring if not risky)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=False,
               labelcolor=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return len(g)


# ---------------------------------------------------------------------------
# Plain-English explanations
# ---------------------------------------------------------------------------

def feature_lines(df, row) -> list:
    out = []
    for f in FEATURES:
        out.append(f"| `{f}` | {row[f]:,} | higher than {percentile(df, f, row[f]):.1f}% "
                   f"of companies |")
    return out


def rank_lines(row, n) -> list:
    names = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression",
             "rule_address": "Rule 2 (shared address)", "random": "Random baseline"}
    return [f"| {names[m]} | {row[f'score_{m}']:.4g} | {row[f'rank_{m}']:,} of {n:,} |"
            for m in RANK_METHODS]


def unusual(df, row) -> list:
    return [f for f in FEATURES if percentile(df, f, row[f]) >= 90]


def _owner_text(graph, company) -> str:
    names = [f"{graph.nodes[p]['name']} (controls {graph.out_degree(p)} companies)"
             for p in graph.predecessors(company)]
    return ", ".join(names)


def explain_company(case, df, row, twins, base_rate, graph=None) -> list:
    n = len(df)
    pos_ranks = df.loc[df["label"] == 1, f"rank_{PRIMARY}"]
    odd = unusual(df, row)
    odd_text = (", ".join(f"`{f}` = {row[f]:,}" for f in odd)
                if odd else "none of its four values is unusual")
    lines = []
    if case == "best_positive":
        lines += [
            f"- This is the risky company that Random Forest ranked highest, yet it is only "
            f"at **{row['rank_random_forest']:,}** of {n:,} (Logistic Regression: "
            f"{row['rank_logistic_regression']:,}). No risky company made the top 1,000.",
            f"- Unusual values (top 10%): {odd_text}.",
            f"- **{twins['n']:,} companies have exactly the same four feature values; "
            f"{twins['positives']} of them are risky** ({twins['rate']:.3%}, about "
            f"{twins['rate'] / base_rate:.0f}x the overall rate of {base_rate:.4%}). That "
            "small enrichment is why the forest puts this profile in roughly the top 1-2%, "
            "but it cannot tell this company apart from the other "
            f"{twins['n'] - twins['positives']:,} non-risky twins, so it cannot rank higher.",
        ]
    elif case == "top_false_pos":
        owners = risky_twin_owners(df, row, graph)
        owner_txt = ", ".join(f"{name} ({k})" for name, k in owners.most_common())
        same_group = df[twin_mask(df, row) & (df["label"] == 0).to_numpy()]
        top_ranks = sorted(same_group["rank_random_forest"])[:10]
        lines += [
            "- Random Forest's single highest score belongs to a company that is **not "
            f"risky** (Logistic Regression ranks it {row['rank_logistic_regression']:,}). "
            f"Its owner: {_owner_text(graph, 'company:' + row['company_number'])}.",
            f"- Unusual values (top 10%): {odd_text}. Hub component: "
            f"{'yes' if row['is_hub_component'] else 'no'}.",
            f"- **{twins['n']:,} companies share its exact feature profile and "
            f"{twins['positives']} of them are risky**, all owned by: {owner_txt}. The "
            "non-risky twins take Random Forest ranks "
            f"{', '.join(f'{r:,}' for r in top_ranks)}.",
            f"- When this company was scored, the training folds held {twins['train_n']} "
            f"companies with this exact profile and **{twins['train_positives']} of them were "
            "risky**. So the forest learned 'this exact profile = risky' from one flagged "
            "owner's group of companies, and applied it to an unrelated corporate group that "
            "happens to have the same shape. This is memorising one group, not a general "
            "pattern.",
        ]
    elif case == "typical_pos":
        lines += [
            f"- Its Random Forest rank, {row['rank_random_forest']:,}, is the closest to the "
            f"median rank of all risky companies ({pos_ranks.median():,.0f}).",
            f"- Unusual values (top 10%): {odd_text}.",
            f"- **{twins['n']:,} companies have exactly the same four feature values; "
            f"{twins['positives']} are risky** (rate {twins['rate']:.4%} vs {base_rate:.4%} "
            f"overall). This is the most common profile of all ({twins['n'] / len(df):.0%} of "
            "companies: one owner, a two-node component, no shared address). On these four "
            "features it is indistinguishable from an ordinary company, so it lands mid-table.",
        ]
    return lines


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def model_comparison(df: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    lr, rf = df["score_logistic_regression"], df["score_random_forest"]
    pos = df[df["label"] == 1]
    top_lr = set(df.nsmallest(1000, "rank_logistic_regression")["company_number"])
    top_rf = set(df.nsmallest(1000, "rank_random_forest")["company_number"])
    recall = {
        m: [int((pos[f"rank_{m}"] <= k).sum()) for k in RECALL_KS] for m in RANK_METHODS
    }
    disagree = pos.assign(gap=(pos["rank_random_forest"] - pos["rank_logistic_regression"]).abs()
                          ).sort_values("gap", ascending=False).iloc[0]
    main = metrics[metrics["run"] == "main"].set_index("method")
    return {
        "spearman": float(lr.rank().corr(rf.rank())),
        "top1000_overlap": len(top_lr & top_rf),
        "rf_better": int((pos["rank_random_forest"] < pos["rank_logistic_regression"]).sum()),
        "lr_better": int((pos["rank_logistic_regression"] < pos["rank_random_forest"]).sum()),
        "median_rank": {m: float(pos[f"rank_{m}"].median()) for m in RANK_METHODS},
        "recall": recall, "n_pos": len(pos), "n": len(df),
        "disagree": disagree, "metrics": main,
    }


def plot_model_comparison(df: pd.DataFrame, comp: dict, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    pos = df[df["label"] == 1]
    ax = axes[0]
    train._style(ax)
    ax.scatter(pos["rank_logistic_regression"], pos["rank_random_forest"], s=36,
               color=C_COMPANY, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    lim = (1_000, len(df) * 1.6)
    ax.plot(lim, lim, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(1_600, 2_300, "same rank", color=INK_MUTED, fontsize=8, rotation=35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Logistic Regression rank (1 = most risky)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("Random Forest rank", color=INK_MUTED, fontsize=10)
    ax.set_title(f"Where each model ranked the {comp['n_pos']} risky companies\n"
                 f"(below the line = Random Forest ranked it higher)",
                 loc="left", fontsize=10, color=INK)

    ax = axes[1]
    train._style(ax)
    for m in RANK_METHODS:
        ax.plot(RECALL_KS, comp["recall"][m], marker="o", markersize=5, linewidth=2,
                color=train.COLORS[m], linestyle=train.LINESTYLES[m],
                label=train.METHOD_LABELS[m])
    expected = [comp["n_pos"] * k / comp["n"] for k in RECALL_KS]
    ax.plot(RECALL_KS, expected, color=INK_MUTED, linewidth=1, linestyle=(0, (1, 3)),
            label="Expected by chance")
    ax.set_xscale("log")
    ax.set_xlabel("Companies reviewed (top k)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("Risky companies found", color=INK_MUTED, fontsize=10)
    ax.set_title("Risky companies found in the top k", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def comparison_lines(comp: dict) -> list:
    m = comp["metrics"]
    d = comp["disagree"]
    lines = [
        "## 5. Random Forest vs Logistic Regression",
        "",
        "![Model comparison](../results/case_model_comparison.png)",
        "",
        "| | Logistic Regression | Random Forest |",
        "|---|---|---|",
        f"| PR-AUC (mean ± std over folds) | {m.loc['logistic_regression', 'pr_auc_mean']:.5f} ± "
        f"{m.loc['logistic_regression', 'pr_auc_std']:.5f} | "
        f"{m.loc['random_forest', 'pr_auc_mean']:.5f} ± {m.loc['random_forest', 'pr_auc_std']:.5f} |",
        f"| ROC-AUC | {m.loc['logistic_regression', 'roc_auc_mean']:.3f} | "
        f"{m.loc['random_forest', 'roc_auc_mean']:.3f} |",
        f"| Accuracy at 0.5 (context only) | {m.loc['logistic_regression', 'accuracy_at_0_5']:.3f} | "
        f"{m.loc['random_forest', 'accuracy_at_0_5']:.3f} |",
        f"| Median rank of the risky companies | {comp['median_rank']['logistic_regression']:,.0f} | "
        f"{comp['median_rank']['random_forest']:,.0f} |",
        f"| Risky companies it ranked higher than the other model did | {comp['lr_better']} | "
        f"{comp['rf_better']} |",
        "",
        "Risky companies found in the top k (pooled out-of-fold scores):",
        "",
        "| k | " + " | ".join(f"{k:,}" for k in RECALL_KS) + " |",
        "|---|" + "---|" * len(RECALL_KS),
        *[f"| {train.METHOD_LABELS[mm]} | " + " | ".join(str(v) for v in comp["recall"][mm]) + " |"
          for mm in RANK_METHODS],
        "| Expected by chance | " + " | ".join(
            f"{comp['n_pos'] * k / comp['n']:.1f}" for k in RECALL_KS) + " |",
        "",
        "- **The two models agree only weakly:** Spearman rank correlation of their scores "
        f"across all {comp['n']:,} companies is {comp['spearman']:.2f}, and "
        + (f"{comp['top1000_overlap']} companies are"
           if comp["top1000_overlap"] else "no company is")
        + " in both top-1,000 lists.",
        f"- **Biggest disagreement on a risky company:** {d['company_number']} — Logistic "
        f"Regression rank {d['rank_logistic_regression']:,}, Random Forest rank "
        f"{d['rank_random_forest']:,}.",
        "- **Why they differ:** Logistic Regression can only move a score smoothly up or down "
        "along each feature, so similar companies get similar scores. Random Forest scores "
        "small groups of companies with near-identical values by how many training positives "
        "they held. With ~40 positives per training fold that lets it memorise exact profiles "
        "(cases 2 and 4 are the two sides of one memorised profile). It ranks "
        f"{comp['rf_better']} of the {comp['n_pos']} risky companies higher than Logistic "
        "Regression does, without either being useful.",
        "- **Neither model is useful at a realistic review size:** in the top 1,000 neither "
        "finds a single risky company. Further down, Random Forest finds "
        f"{comp['recall']['random_forest'][4]} by k = 50,000 and Logistic Regression "
        f"{comp['recall']['logistic_regression'][4]}, against "
        f"{comp['n_pos'] * 50_000 / comp['n']:.1f} expected by chance and "
        f"{comp['recall']['rule_address'][4]} for the simple shared-address rule. These are small "
        "counts from 52 positives, so the differences are not reliable, and "
        "neither model beats the baselines (Checkpoint 4).",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(graph_path=GRAPH_INPUT, features_path=FEATURES_INPUT, predictions_path=PREDICTIONS_INPUT,
        metrics_path=METRICS_INPUT, results_dir=RESULTS, doc_path=DOC_OUTPUT):
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    df = load_table(features_path, predictions_path)
    metrics = pd.read_csv(metrics_path)
    n, base_rate = len(df), df["label"].mean()
    by_node = {("company:" + r["company_number"]): r for r in df.to_dict("records")}
    positives = {c for c, r in by_node.items() if r["label"] == 1}
    key_sizes = address_key_sizes(graph)
    cases = select_cases(df, graph)
    os.makedirs(results_dir, exist_ok=True)

    doc = [
        "# Case studies (Phase 5)",
        "",
        f"Generated {date.today().isoformat()} by `src/case_studies.py`. Ranks are out-of-fold "
        "(each company scored by a model that never saw it), 1 = most risky, with the same "
        "tie-break as Phase 4.",
        "",
        "No risky company reached the top 1,000 of any method (Checkpoint 4), so these cases "
        "explain **why the models failed** rather than show successes. The key test is the "
        "**feature twins** count: how many companies have exactly the same four feature values. "
        "A model that only sees these four numbers must give all twins the same score.",
        "",
        "Drawings: squares are companies, circles people, diamonds corporate owners; a red ring "
        "marks a flagged owner or risky company; grey squares are an address or a '+N more' "
        f"summary (at most {MAX_PER_NODE} neighbours drawn per node).",
        "",
    ]
    summary = {}
    for case in ("best_positive", "top_false_pos", "typical_pos"):
        node = cases[case]
        row = pd.Series(by_node[node])
        twins = feature_twins(df, row)
        addr = defaultdict(list)
        for p in graph.predecessors(node):
            key = graph.edges[p, node]["address_key"]
            if key:
                addr[key].append(p)
        address_nodes = [
            (f"Address: {_short(k, 28)}\n{key_sizes[k]:,} "
             f"compan{'y' if key_sizes[k] == 1 else 'ies'}"
             + (" (capped)" if key_sizes[k] > ADDRESS_KEY_MAX_COMPANIES else ""), ps)
            for k, ps in addr.items()]
        img = f"case_{case}.png"
        drawn = draw_case(
            graph, by_node, node, 2,
            f"{CASE_TITLES[case]}: company {row['company_number']} "
            f"(RF rank {row['rank_random_forest']:,}, LR rank {row['rank_logistic_regression']:,})",
            os.path.join(results_dir, img), positives, {}, address_nodes)
        doc += [
            f"## {CASE_TITLES[case]}: company {row['company_number']}",
            "",
            f"![{CASE_TITLES[case]}](../results/{img})",
            "",
            *explain_company(case, df, row, twins, base_rate, graph),
            "",
            "| Feature | Value | Percentile |",
            "|---|---|---|",
            *feature_lines(df, row),
            "",
            "| Method | Score | Rank |",
            "|---|---|---|",
            *rank_lines(row, n),
            "",
        ]
        summary[case] = {"company": row["company_number"], "rf_rank": int(row["rank_random_forest"]),
                         "lr_rank": int(row["rank_logistic_regression"]), "twins": twins,
                         "nodes_drawn": drawn}

    # Case 4: Peter Caruana's group
    car = cases["caruana_group"]
    car_cos = sorted(graph.successors(car))
    f1 = df[df["flagged_neighbour_companies"] > 0]
    f1_links = {("company:" + c): feature1_links(graph, "company:" + c, positives, key_sizes)
                for c in f1["company_number"]}
    in_group = {c for c, links in f1_links.items()
                if any(p in car_cos for p, _ in links)}
    car_row = pd.Series(by_node[car_cos[0]])
    car_twins = feature_twins(df, car_row)
    car_owners = risky_twin_owners(df, car_row, graph)
    other_owners = sorted({graph.nodes[p]["name"] for c in car_cos
                           for p in graph.predecessors(c) if p != car})
    important = set(car_cos) | set(f1_links)
    extra = {c: f"feature 1 = {by_node[c]['flagged_neighbour_companies']}" for c in f1_links}
    drawn = draw_case(graph, by_node, car, 3,
                      f"{CASE_TITLES['caruana_group']}: {graph.nodes[car]['name']} "
                      f"(flagged {graph.nodes[car]['flag_source'].upper()}, "
                      f"{len(car_cos)} companies)",
                      os.path.join(results_dir, "case_caruana_group.png"), important, extra)
    doc += [
        f"## {CASE_TITLES['caruana_group']}",
        "",
        "![Peter Caruana's group](../results/case_caruana_group.png)",
        "",
        f"- {graph.nodes[car]['name']} is the only flagged owner who controls more than one "
        f"company: **{len(car_cos)} companies**, all labelled risky.",
        "- **His own companies all have `flagged_neighbour_companies` = 0**, by design: he is "
        "flagged, so he is removed before counting neighbours (otherwise the feature would "
        "contain the label).",
        ("- His companies have no owner other than him. " if not other_owners else
         f"- His companies' other owners: {', '.join(other_owners)}. ")
        + "Once he is removed, there is nothing left to link through.",
        f"- Feature 1 is non-zero for only **{len(f1)} companies** in the whole dataset, none "
        f"of them risky, and **{len(in_group)} of them** are linked to his group. "
        + ("So feature 1 does NOT fire around Caruana at all; " if not in_group else "")
        + "every non-zero value comes from a different risky company:",
        "",
        "| Company | Feature 1 | Linked to (risky company) | How |",
        "|---|---|---|---|",
        *[f"| {c.removeprefix('company:')} | {by_node[c]['flagged_neighbour_companies']} | "
          f"{p.removeprefix('company:')} | {how} |"
          for c, links in sorted(f1_links.items()) for p, how in links],
        "",
        "| Caruana company | Label | Feature 1 | RF rank | LR rank |",
        "|---|---|---|---|---|",
        *[f"| {c.removeprefix('company:')} | {by_node[c]['label']} | "
          f"{by_node[c]['flagged_neighbour_companies']} | {by_node[c]['rank_random_forest']:,} | "
          f"{by_node[c]['rank_logistic_regression']:,} |" for c in car_cos],
        "",
        "- **Why his companies rank near the bottom** (Random Forest ~382,000, Logistic "
        f"Regression ~405,600 of {n:,}): all 8 are in the same split group, so they always "
        "share a test fold. When they were scored, the training folds held "
        f"{car_twins['train_n']} companies with their exact profile and "
        f"{car_twins['train_positives']} of them were risky (the twins of case 2). The model "
        "had learned 'this profile = not risky', the mirror image of case 2. The grouped "
        "split did its job: his companies could not vouch for each other.",
        "- Their only risky twins are owned by: "
        f"{', '.join(f'{k} ({v})' for k, v in car_owners.items())}, i.e. themselves; nobody "
        "else in the data has this profile and is risky.",
        "- So the feature designed to capture 'risk spreads through the network' can only "
        "fire on *neighbours* of a risky company through a non-flagged owner or a "
        f"not-too-busy address, and in this 1-of-32 sample there are just {len(f1)} such "
        "neighbours, none of them risky themselves.",
        "",
    ]
    summary["caruana_group"] = {"node": car, "companies": len(car_cos),
                                "feature1_companies": len(f1), "feature1_via_group": len(in_group),
                                "nodes_drawn": drawn}

    comp = model_comparison(df, metrics)
    plot_model_comparison(df, comp, os.path.join(results_dir, "case_model_comparison.png"))
    doc += comparison_lines(comp)
    summary["model_comparison"] = {k: comp[k] for k in
                                   ("spearman", "top1000_overlap", "rf_better", "lr_better",
                                    "median_rank", "recall")}

    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    result = run()
    for k, v in result.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"Case studies written to: {DOC_OUTPUT}", file=sys.stderr)
