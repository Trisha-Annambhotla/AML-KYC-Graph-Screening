"""
Phase 4: train and compare the models (docs/implementation_plan.md).

Logistic Regression vs Random Forest on the four Phase 3 features, judged
against three baselines, with 5-fold GroupKFold on split_group_id (so
owner-linked and address-linked companies never sit in both train and test).

Label runs
    main      label (PEP + sanctions combined, 52 positives)
    pep_only  PEP-only label (46 positives). Sanctions-only positives are
              dropped from this run rather than counted as negatives.
    No label_weak run and no sanctions-only run (see the plan).

Methods
    logistic_regression  log1p -> StandardScaler -> LR(class_weight=balanced)
    random_forest        raw features, 300 trees, min_samples_leaf=5,
                         class_weight=balanced_subsample
    random               uniform random scores (baseline)
    rule_neighbour_then_address
                         rank by flagged_neighbour_companies, then
                         shared_address_count (baseline)
    rule_address         rank by shared_address_count alone (baseline)

Metrics
    Per fold: PR-AUC (average precision) and ROC-AUC -> mean +/- std (sample
    std, ddof=1) over folds that contain at least one positive.
    Pooled out-of-fold scores: precision@k, recall@k, lift@k (precision@k /
    positive rate) for k in K_VALUES; accuracy at 0.5 for the two models
    (context only). Ties in a ranking (common for the rules) are broken by
    one fixed random order (seed 42) shared by every method, so no method
    gets a lucky or unlucky tie order.

Interpretation (main run only)
    LR coefficients (on the scaled log1p features, mean +/- std over folds)
    and RF permutation importance (drop in average precision on each held-out
    fold, mean +/- std over folds).

Outputs
    results/metrics.csv, results/metrics.md, results/feature_importance.csv,
    results/pr_curves.png, results/feature_importance.png,
    data/processed/predictions.csv (out-of-fold scores).

Run directly:
    python src/train.py
"""

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

from features import FEATURES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_INPUT = os.path.join(ROOT, "data", "processed", "features.csv")
PREDICTIONS_OUTPUT = os.path.join(ROOT, "data", "processed", "predictions.csv")
RESULTS = os.path.join(ROOT, "results")

RANDOM_STATE = 42
N_FOLDS = 5
K_VALUES = [50, 100, 500, 1000]
GROUP_COL = "split_group_id"

MODELS = ["logistic_regression", "random_forest"]
BASELINES = ["random", "rule_neighbour_then_address", "rule_address"]
METHOD_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "random": "Baseline: random",
    "rule_neighbour_then_address": "Baseline: rule 1 (neighbours, then address)",
    "rule_address": "Baseline: rule 2 (shared address)",
}
RUN_LABELS = {"main": "Main label (PEP + sanctions)", "pep_only": "PEP-only label"}

# Reference palette (dataviz skill), light mode; validated as an adjacent set.
COLORS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "rule_neighbour_then_address": "#1baf7a",
    "rule_address": "#eda100",
    "random": "#8f8e89",
}
LINESTYLES = {
    "logistic_regression": "-", "random_forest": "-",
    "rule_neighbour_then_address": "--", "rule_address": (0, (6, 2, 1, 2)),
    "random": ":",
}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------------------
# Models, baselines, folds
# ---------------------------------------------------------------------------

def make_models(rf_estimators: int = 300) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000,
                                      random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_estimators, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    }


def baseline_scores(df: pd.DataFrame) -> dict:
    sa = df["shared_address_count"].to_numpy(float)
    fn = df["flagged_neighbour_companies"].to_numpy(float)
    rng = np.random.default_rng(RANDOM_STATE)
    return {
        "random": rng.random(len(df)),
        # lexicographic: feature 1 first, shared address breaks its ties
        "rule_neighbour_then_address": fn * (sa.max() + 1) + sa,
        "rule_address": sa,
    }


def assign_folds(groups, n_folds: int = N_FOLDS) -> np.ndarray:
    fold = np.full(len(groups), -1)
    dummy = np.zeros(len(groups))
    for f, (_, test_idx) in enumerate(GroupKFold(n_splits=n_folds).split(dummy, groups=groups)):
        fold[test_idx] = f
    return fold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rank_order(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices sorted by score (high first); ties broken by the fixed tiebreak."""
    return np.lexsort((tiebreak, -scores))


def at_k(y: np.ndarray, scores: np.ndarray, tiebreak: np.ndarray, k: int) -> dict:
    top = rank_order(scores, tiebreak)[:k]
    hits = int(y[top].sum())
    n_pos = int(y.sum())
    precision = hits / len(top)
    rate = n_pos / len(y)
    return {"hits": hits, "precision": precision,
            "recall": hits / n_pos if n_pos else np.nan,
            "lift": precision / rate if rate else np.nan}


def _fold_scores(y, scores, fold):
    pr, roc = [], []
    for f in np.unique(fold):
        m = fold == f
        if y[m].sum() == 0 or y[m].sum() == m.sum():
            continue  # undefined without both classes
        pr.append(average_precision_score(y[m], scores[m]))
        roc.append(roc_auc_score(y[m], scores[m]))
    return pr, roc


def _mean_std(values):
    if not values:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def metrics_row(run, method, y, scores, fold, tiebreak, is_model) -> dict:
    pr, roc = _fold_scores(y, scores, fold)
    pr_m, pr_s = _mean_std(pr)
    roc_m, roc_s = _mean_std(roc)
    row = {
        "run": run, "method": method, "type": "model" if is_model else "baseline",
        "n_companies": len(y), "n_positive": int(y.sum()),
        "positive_rate": y.mean(),
        "pr_auc_mean": pr_m, "pr_auc_std": pr_s,
        "roc_auc_mean": roc_m, "roc_auc_std": roc_s,
        "folds_scored": len(pr),
        "accuracy_at_0_5": float(((scores >= 0.5) == y).mean()) if is_model else np.nan,
    }
    for k in K_VALUES:
        r = at_k(y, scores, tiebreak, k)
        row[f"hits@{k}"] = r["hits"]
        row[f"precision@{k}"] = r["precision"]
        row[f"recall@{k}"] = r["recall"]
        row[f"lift@{k}"] = r["lift"]
    return row


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(df: pd.DataFrame, label_col: str, run: str, models: dict = None,
                   importance: bool = False):
    """
    Returns (metrics rows, out-of-fold scores {method: array}, fold array,
    importance DataFrame or None).
    """
    models = models if models is not None else make_models()
    X = df[FEATURES].to_numpy(float)
    y = df[label_col].to_numpy(int)
    fold = assign_folds(df[GROUP_COL].to_numpy())
    tiebreak = np.random.default_rng(RANDOM_STATE + 1).random(len(df))

    oof = {m: np.full(len(df), np.nan) for m in models}
    coefs, perms = [], []
    for f in range(N_FOLDS):
        train, test = fold != f, fold == f
        for name, model in models.items():
            est = clone(model).fit(X[train], y[train])
            oof[name][test] = est.predict_proba(X[test])[:, 1]
            if not importance:
                continue
            if name == "logistic_regression":
                coefs.append(est.named_steps["lr"].coef_[0])
            elif name == "random_forest" and y[test].sum() > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pi = permutation_importance(est, X[test], y[test],
                                                scoring="average_precision", n_repeats=5,
                                                random_state=RANDOM_STATE)
                perms.append(pi.importances_mean)

    scores = {**oof, **baseline_scores(df)}
    rows = [metrics_row(run, m, y, s, fold, tiebreak, m in models) for m, s in scores.items()]

    imp = None
    if importance:
        imp = pd.DataFrame({
            "feature": FEATURES,
            "lr_coef_mean": np.mean(coefs, axis=0), "lr_coef_std": np.std(coefs, axis=0, ddof=1),
            "rf_perm_mean": np.mean(perms, axis=0), "rf_perm_std": np.std(perms, axis=0, ddof=1),
        })
    return rows, scores, fold, imp


def pep_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    """PEP-only run: drop sanctions-only positives, label = has a PEP flag."""
    keep = df["label_source"] != "sanctions"
    out = df[keep].copy()
    out["label_pep"] = out["label_source"].isin(["pep", "both"]).astype(int)
    return out


# ---------------------------------------------------------------------------
# Verdicts and reports
# ---------------------------------------------------------------------------

def verdict(model: dict, base: dict) -> str:
    diff = model["pr_auc_mean"] - base["pr_auc_mean"]
    noise = max(model["pr_auc_std"], base["pr_auc_std"])
    p_m, p_b = model["precision@100"], base["precision@100"]
    if diff > noise:
        pr_text = "clearly higher PR-AUC"
    elif diff > 0:
        pr_text = "higher PR-AUC, but within fold-to-fold noise"
    else:
        pr_text = "lower or equal PR-AUC"
    at100 = ("more" if p_m > p_b else "the same number of" if p_m == p_b else "fewer")
    beats = diff > noise and p_m >= p_b
    return (f"**{'BEATS' if beats else 'does NOT clearly beat'}** -- {pr_text} "
            f"({model['pr_auc_mean']:.5f} vs {base['pr_auc_mean']:.5f}); "
            f"{at100} positives in the top 100 ({model['hits@100']} vs {base['hits@100']})")


def _fmt(x, digits=5):
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


def write_report(metrics: pd.DataFrame, imp: pd.DataFrame, positives_per_fold: dict, path: str):
    lines = [
        "# Phase 4 results: Logistic Regression vs Random Forest",
        "",
        f"5-fold GroupKFold on `{GROUP_COL}`; `random_state={RANDOM_STATE}`. "
        "PR-AUC and ROC-AUC are mean ± std over folds; @k metrics use the pooled "
        "out-of-fold scores. Lift = precision@k / positive rate.",
        "",
        "**Note:** `flagged_neighbour_companies` is kept as a model feature (it is in the "
        "scope) but is near-constant: non-zero for only 7 of 406,715 companies, none of "
        "them positive. Rule 1 therefore ranks almost exactly like rule 2.",
        "",
    ]
    for run, sub in metrics.groupby("run", sort=False):
        n_pos, rate = int(sub["n_positive"].iloc[0]), sub["positive_rate"].iloc[0]
        lines += [
            f"## {RUN_LABELS[run]}",
            "",
            f"{int(sub['n_companies'].iloc[0]):,} companies, {n_pos} positives "
            f"(rate {rate:.5%}). Positives per fold: {positives_per_fold[run]}. "
            f"There are fewer positives than every k, so the best possible "
            f"precision@k is {', '.join(f'{min(1, n_pos / k):.3f} (k={k})' for k in K_VALUES)}.",
            "",
            "| Method | PR-AUC | ROC-AUC | " + " | ".join(f"P@{k}" for k in K_VALUES)
            + " | " + " | ".join(f"R@{k}" for k in K_VALUES)
            + " | " + " | ".join(f"Lift@{k}" for k in K_VALUES) + " | Acc@0.5 |",
            "|---|---|---|" + "---|" * (3 * len(K_VALUES)) + "---|",
        ]
        for r in sub.to_dict("records"):
            lines.append(
                f"| {METHOD_LABELS[r['method']]} "
                f"| {_fmt(r['pr_auc_mean'])} ± {_fmt(r['pr_auc_std'])} "
                f"| {_fmt(r['roc_auc_mean'], 3)} ± {_fmt(r['roc_auc_std'], 3)} | "
                + " | ".join(f"{r[f'precision@{k}']:.3f}" for k in K_VALUES) + " | "
                + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in K_VALUES) + " | "
                + " | ".join(f"{r[f'lift@{k}']:.1f}" for k in K_VALUES)
                + f" | {_fmt(r['accuracy_at_0_5'], 4)} |"
            )
        lines += ["", "**Does each model beat each baseline?**", ""]
        rows = {r["method"]: r for r in sub.to_dict("records")}
        for m in MODELS:
            for b in BASELINES:
                lines.append(f"- {METHOD_LABELS[m]} vs {METHOD_LABELS[b]}: "
                             f"{verdict(rows[m], rows[b])}")
        lines.append("")
    if imp is not None:
        lines += [
            "## Interpretation (main label)",
            "",
            "| Feature | LR coefficient (scaled log1p) | RF permutation importance (drop in AP) |",
            "|---|---|---|",
            *[f"| {r.feature} | {r.lr_coef_mean:+.3f} ± {r.lr_coef_std:.3f} "
              f"| {r.rf_perm_mean:+.5f} ± {r.rf_perm_std:.5f} |" for r in imp.itertuples()],
            "",
            "A beats-verdict needs a PR-AUC gap larger than the fold-to-fold std AND at "
            "least as many positives in the top 100.",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr_curves(curves: dict, path: str):
    """curves: {run: (y, {method: scores})}"""
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4.8),
                             sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (run, (y, scores)) in zip(axes, curves.items()):
        _style(ax)
        for method in BASELINES[::-1] + MODELS:
            prec, rec, _ = precision_recall_curve(y, scores[method])
            ap = average_precision_score(y, scores[method])
            keep = rec > 0  # drop sklearn's (recall 0, precision 1) endpoint
            ax.plot(rec[keep], prec[keep], color=COLORS[method], linestyle=LINESTYLES[method],
                    linewidth=2, label=f"{METHOD_LABELS[method]} (pooled AP {ap:.5f})")
        ax.set_yscale("log")
        ax.set_ylim(max(y.mean() / 5, 1e-6), 1)
        ax.set_xlim(0, 1)
        ax.set_title(f"{RUN_LABELS[run]}: {int(y.sum())} positives in {len(y):,}",
                     color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Recall", color=INK_MUTED, fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK)
    axes[0].set_ylabel("Precision (log scale)", color=INK_MUTED, fontsize=10)
    fig.suptitle("Precision-recall curves, pooled out-of-fold scores",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    panels = [
        ("lr_coef", "Logistic Regression coefficients\n(scaled log1p features, mean ± std over folds)"),
        ("rf_perm", "Random Forest permutation importance\n(drop in average precision, held-out folds)"),
    ]
    order = imp.iloc[::-1]
    for ax, (col, title) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.barh(order["feature"], order[f"{col}_mean"], xerr=order[f"{col}_std"],
                height=0.5, color=COLORS["logistic_regression"],
                error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 3})
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(axis="y", colors=INK, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_features(path: str = FEATURES_INPUT) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"company_number": str, "label_source": str},
                       keep_default_na=False)


def run(features_path=FEATURES_INPUT, results_dir=RESULTS, predictions_path=PREDICTIONS_OUTPUT,
        models: dict = None):
    df = load_features(features_path)
    pep = pep_only_frame(df)

    rows_main, scores_main, fold_main, imp = cross_validate(df, "label", "main", models,
                                                            importance=True)
    rows_pep, scores_pep, fold_pep, _ = cross_validate(pep, "label_pep", "pep_only", models)

    metrics = pd.DataFrame(rows_main + rows_pep)
    positives_per_fold = {
        "main": [int(df["label"][fold_main == f].sum()) for f in range(N_FOLDS)],
        "pep_only": [int(pep["label_pep"][fold_pep == f].sum()) for f in range(N_FOLDS)],
    }

    os.makedirs(results_dir, exist_ok=True)
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    imp.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
    write_report(metrics, imp, positives_per_fold, os.path.join(results_dir, "metrics.md"))
    plot_pr_curves({"main": (df["label"].to_numpy(), scores_main),
                    "pep_only": (pep["label_pep"].to_numpy(), scores_pep)},
                   os.path.join(results_dir, "pr_curves.png"))
    plot_importance(imp, os.path.join(results_dir, "feature_importance.png"))

    preds = df[["company_number", GROUP_COL, "label", "label_source"]].copy()
    preds["fold"] = fold_main
    for m, s in scores_main.items():
        preds[f"score_{m}"] = s
    pep_scores = pd.DataFrame({f"score_{m}_pep": s for m, s in scores_pep.items()
                               if m in MODELS}, index=pep.index)
    preds = preds.join(pep_scores)
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    preds.to_csv(predictions_path, index=False)
    return metrics, imp, positives_per_fold


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES_INPUT)
    args = ap.parse_args()

    m, _, ppf = run(args.features)
    cols = ["run", "method", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "hits@100", "precision@100"]
    print(m[cols].to_string(index=False), file=sys.stderr)
    print(f"Positives per fold: {ppf}", file=sys.stderr)
    print(f"Results written to: {RESULTS}", file=sys.stderr)
