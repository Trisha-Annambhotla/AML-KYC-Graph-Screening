"""
Interactive investigation dashboard for the AML/KYC graph-screening project.

Run from the project root (after `python scripts/run_pipeline.py`):
    streamlit run app/dashboard.py

Pages
    Overview          key numbers, list matches, the Phase 4 metrics table
    Investigate       search a company (by number or owner name): owners, list
                      matches, features, model scores, "Why this score?",
                      and a 2-hop network view (at most 150 nodes)
    Case studies      the four Phase 5 cases, each opening in Investigate

All data logic lives in src/investigate.py (tested); this file is the UI.
"""

import os
import sys

import pandas as pd
import streamlit as st
from pyvis.network import Network

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import investigate as inv  # noqa: E402
from train import METHOD_LABELS  # noqa: E402

ROOT = inv.ROOT
PAGES = ["Overview", "Investigate", "Case studies"]
CASE_IMAGES = ["case_best_positive.png", "case_top_false_pos.png",
               "case_typical_pos.png", "case_caruana_group.png"]

# Reference palette (same as the report charts)
C_COMPANY, C_PERSON, C_CORP, C_FLAG = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
C_EDGE, C_INK = "#bdbcb6", "#0b0b0b"

EXPERIMENTAL = (
    "**Experimental -- not predictive in this sample.** Neither model beat the simple "
    "baselines, and no method placed a single risky company in its top 1,000 "
    "(see `results/metrics.md`, shown below). Treat these scores as a demonstration "
    "of the method, not as a risk assessment."
)

st.set_page_config(page_title="AML/KYC graph screening", layout="wide")


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the ownership graph (can take ~30 seconds)...")
def get_graph():
    return inv.load_graph()


@st.cache_resource(show_spinner="Loading companies and scores...")
def get_companies():
    return inv.load_companies().set_index("company_number", drop=False)


@st.cache_resource(show_spinner="Indexing owners...")
def get_owners():
    return inv.owner_index(get_graph())


@st.cache_data
def get_matches():
    return inv.load_matches()


@st.cache_data
def get_metrics():
    return inv.load_metrics()


@st.cache_data
def get_metrics_doc():
    with open(inv.METRICS_DOC, encoding="utf-8") as f:
        return f.read()


@st.cache_resource(show_spinner="Selecting case studies...")
def get_cases():
    return inv.pick_cases(get_companies().reset_index(drop=True), get_graph())


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def open_company(company_number: str):
    """Callback: switch to Investigate with this company selected."""
    st.session_state["company"] = company_number
    st.session_state["page"] = "Investigate"


def on_company_number():
    st.session_state["company"] = inv.normalise_company_number(
        st.session_state.get("company_query", ""))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.title("AML/KYC graph screening")
    st.write("Companies House owners (1 of 32 PSC files) matched against the UK FCDO "
             "sanctions list and a UK PEP list, turned into an ownership graph, and "
             "scored by Logistic Regression and Random Forest.")
    companies, matches = get_companies(), get_matches()
    nums = inv.overview_numbers(companies, matches)
    by_src = nums["positives_by_source"]

    c = st.columns(4)
    c[0].metric("Companies", f"{nums['companies']:,}")
    c[1].metric("Risky companies (label = 1)", f"{nums['positives']}",
                help="Directly owned by a strongly matched PEP or sanctioned owner.")
    c[2].metric("of which PEP", f"{by_src.get('pep', 0) + by_src.get('both', 0)}")
    c[3].metric("of which sanctions", f"{by_src.get('sanctions', 0) + by_src.get('both', 0)}")

    st.subheader("List matches by list and tier")
    st.caption("Strong matches become labels; weak matches (similar names only) are "
               "reported but never used as labels.")
    st.dataframe(nums["matches"], width="content")

    st.subheader("Model comparison (Phase 4)")
    st.caption("5-fold cross-validation grouped by split_group_id. PR-AUC is mean ± std "
               "over folds; the @k columns use pooled out-of-fold scores.")
    m = get_metrics()
    view = pd.DataFrame({
        "label run": m["run"].map({"main": "PEP + sanctions", "pep_only": "PEP only"}),
        "method": m["method"].map(METHOD_LABELS),
        "PR-AUC": [f"{a:.5f} ± {b:.5f}" for a, b in zip(m["pr_auc_mean"], m["pr_auc_std"])],
        "ROC-AUC": m["roc_auc_mean"].round(3),
        "hits@100": m["hits@100"], "hits@1000": m["hits@1000"],
        "accuracy@0.5 (context)": m["accuracy_at_0_5"].round(3),
    })
    st.dataframe(view, hide_index=True, width="stretch")
    st.warning("Neither model beats the baselines; no method puts a risky company in its "
               "top 1,000. Full table and verdicts: `results/metrics.md`.")
    with st.expander("results/metrics.md"):
        st.markdown(get_metrics_doc())


# ---------------------------------------------------------------------------
# Investigate
# ---------------------------------------------------------------------------

def search_panel():
    left, right = st.columns(2)
    with left:
        st.text_input("Company number", key="company_query", on_change=on_company_number,
                      placeholder="e.g. 10277215 or SC447141")
    with right:
        q = st.text_input("Owner name (at least 3 letters)", key="owner_query",
                          placeholder="e.g. Caruana")
    if q:
        hits = inv.search_owners(get_owners(), q)
        if hits.empty:
            st.info("No owner found.")
            return
        labels = [f"{r.name}  ({r.n_companies} compan{'y' if r.n_companies == 1 else 'ies'})"
                  + (f"  [FLAGGED {r.flag_source.upper()}]" if r.flagged else "")
                  for r in hits.itertuples()]
        pick = st.selectbox(f"{len(hits)} owner(s) found (max 50 shown)", range(len(hits)),
                            format_func=lambda i: labels[i])
        owner = hits.iloc[pick]["node_id"]
        cos = sorted(c.removeprefix("company:") for c in get_graph().successors(owner))
        col_a, col_b = st.columns([3, 1])
        chosen = col_a.selectbox("Their companies", cos)
        col_b.button("Investigate", on_click=open_company, args=(chosen,),
                     width="stretch")


def legend_html() -> str:
    def chip(color, shape, text, ring=None):
        radius = "50%" if shape == "circle" else "2px"
        rot = "transform:rotate(45deg);" if shape == "diamond" else ""
        border = f"3px solid {ring}" if ring else f"1px solid {color}"
        return (f'<span style="display:inline-flex;align-items:center;margin-right:16px">'
                f'<span style="width:12px;height:12px;background:{color};border:{border};'
                f'border-radius:{radius};{rot}display:inline-block;margin-right:6px"></span>'
                f'{text}</span>')
    return ("<div style='font-size:0.85rem'>"
            + chip(C_COMPANY, "square", "Company")
            + chip(C_PERSON, "circle", "Person (PSC)")
            + chip(C_CORP, "diamond", "Corporate PSC")
            + chip(C_FLAG, "circle", "Flagged owner (PEP / sanctions)")
            + chip(C_COMPANY, "square", "Risky company (red ring)", ring=C_FLAG)
            + chip(C_COMPANY, "square", "Selected company (black ring)", ring=C_INK)
            + "</div>")


def _safe(text: str) -> str:
    """Names come from public data and end up inside pyvis's <script> block;
    dropping angle brackets means no name can close the script tag."""
    return str(text).replace("<", "").replace(">", "")


def network_html(graph, center: str, companies) -> tuple:
    sub, total, truncated = inv.ego_network(graph, center)
    net = Network(height="560px", width="100%", directed=True, bgcolor="#fcfcfb",
                  font_color=C_INK, cdn_resources="in_line")
    for n, d in sub.nodes(data=True):
        t = d["node_type"]
        if t == "company":
            num = _safe(n.removeprefix("company:"))
            risky = num in companies.index and int(companies.at[num, "label"]) == 1
            border = C_INK if n == center else (C_FLAG if risky else C_COMPANY)
            net.add_node(n, label=num, shape="square", size=22 if n == center else 12,
                         color={"background": C_COMPANY, "border": border},
                         borderWidth=4 if (n == center or risky) else 1,
                         title=f"Company {num}" + (" (risky, label 1)" if risky else ""))
        else:
            flagged = d.get("flagged")
            color = C_FLAG if flagged else (C_PERSON if t == "person" else C_CORP)
            name = _safe(d["name"])
            net.add_node(n, label=name if len(name) <= 24 else name[:23] + "…",
                         shape="dot" if t == "person" else "diamond", size=12,
                         color={"background": color, "border": color},
                         title=(f"{name}" + (f" -- FLAGGED ({d['flag_source'].upper()})"
                                             if flagged else "")
                                + f"\ncontrols {graph.out_degree(n)} compan"
                                + ("y" if graph.out_degree(n) == 1 else "ies")))
    for u, v in sub.edges():
        net.add_edge(u, v, color=C_EDGE, arrows="to")
    net.set_options('{"physics": {"stabilization": {"iterations": 200}}, '
                    '"interaction": {"hover": true}}')
    # pyvis's template also loads Bootstrap from a CDN for menus this view does not
    # use; drop it so the dashboard works offline and makes no outside requests.
    html = "\n".join(line for line in net.generate_html().splitlines()
                     if "cdn.jsdelivr.net/npm/bootstrap" not in line)
    return html, len(sub), total, truncated


def page_investigate():
    st.title("Investigate a company")
    search_panel()
    company = st.session_state.get("company")
    if not company:
        st.info("Enter a company number, or search for an owner and pick one of their "
                "companies. The Case studies page has four good starting points.")
        return
    companies, graph = get_companies(), get_graph()
    if company not in companies.index:
        st.error(f"Company {company} is not in this sample (1 of 32 PSC files).")
        return
    row = companies.loc[company]
    label = int(row["label"])
    st.header(f"Company {company}")
    st.markdown(
        f"**Label:** {'RISKY (label 1, ' + row['label_source'] + ')' if label else 'not risky (label 0)'}"
        f" · [Companies House page](https://find-and-update.company-information.service.gov.uk"
        f"/company/{company}/persons-with-significant-control)")

    matches = inv.company_matches(get_matches(), company)
    twin = inv.twins(companies.reset_index(drop=True), row)

    st.subheader("Why this score?")
    st.markdown("\n".join(f"- {fact}" for fact in inv.explain(row, matches, twin)))

    left, right = st.columns(2)
    with left:
        st.subheader("Owners")
        st.dataframe(inv.company_owners(graph, company), hide_index=True, width="stretch")
        st.subheader("List matches")
        if matches.empty:
            st.write("No owner of this company matched either list.")
        else:
            st.dataframe(pd.DataFrame({
                "owner": matches["psc_name"],
                "list": matches["list_source"].map(inv.LIST_NAMES),
                "list entry": matches["list_name"],
                "tier": matches["tier"], "rule used": matches["rule_used"],
                "score": matches["score"],
                "why": [inv.match_reason(r) for r in matches.to_dict("records")],
            }), hide_index=True, width="stretch")
    with right:
        st.subheader("Features")
        st.dataframe(inv.feature_table(row), hide_index=True, width="stretch")
        st.subheader("Model scores (out-of-fold)")
        st.warning(EXPERIMENTAL)
        st.dataframe(inv.score_table(row, len(companies)), hide_index=True, width="stretch")
        with st.expander("results/metrics.md"):
            st.markdown(get_metrics_doc())

    st.subheader(f"Network: up to {inv.MAX_HOPS} hops around {company}")
    html, shown, total, truncated = network_html(graph, "company:" + company, companies)
    if truncated:
        st.info(f"Showing {shown} of {total:,} nodes within {inv.MAX_HOPS} hops: the view "
                f"is capped at {inv.MAX_NODES} nodes (closest and flagged nodes first).")
    else:
        st.caption(f"All {total} nodes within {inv.MAX_HOPS} hops. Address links are not "
                   "drawn (addresses are not nodes in the ownership graph).")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.iframe(html, height=580)


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------

def page_cases():
    st.title("Case studies")
    st.write("The four cases from `docs/case_studies.md`. No risky company ranked highly, "
             "so they explain why the models failed.")
    for (title, company, summary), img in zip(get_cases(), CASE_IMAGES):
        st.subheader(title)
        st.write(summary)
        cols = st.columns([1, 3])
        cols[0].button(f"Open {company} in Investigate", key=f"case_{company}",
                       on_click=open_company, args=(company,))
        path = os.path.join(ROOT, "results", img)
        if os.path.exists(path):
            with cols[1].expander("Static drawing from the report"):
                st.image(path)


# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"
st.sidebar.radio("Page", PAGES, key="page")
st.sidebar.caption("Data: 1 of 32 Companies House PSC files (2026-08-02 snapshot). "
                   "Labels: strong PEP / sanctions matches (Mode A). The manual "
                   "spot-check of matches is still pending.")

{"Overview": page_overview, "Investigate": page_investigate,
 "Case studies": page_cases}[st.session_state["page"]]()
