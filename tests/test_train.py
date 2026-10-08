import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import (  # noqa: E402
    BASELINES, MODELS, assign_folds, at_k, baseline_scores, cross_validate,
    make_models, pep_only_frame, rank_order, run,
)


def _synthetic(n=600, seed=0):
    """Features with real signal: positives have more shared addresses and owners."""
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.08).astype(int)
    df = pd.DataFrame({
        "company_number": [f"{i:08d}" for i in range(n)],
        "component_id": np.arange(n) // 2,
        "split_group_id": np.arange(n) // 3,
        "label": label,
        "label_weak": label,
        "label_source": np.where(label == 1, np.where(rng.random(n) < 0.8, "pep", "sanctions"), ""),
        "flagged_neighbour_companies": (rng.random(n) < 0.01).astype(int),
        "shared_address_count": rng.poisson(1 + 6 * label),
        "degree": 1 + rng.poisson(0.3 + 2 * label),
        "component_size": 2 + rng.poisson(1, n),
        "flagged_neighbour_via_owner": 0,
        "is_hub_component": False,
    })
    return df


def test_at_k_precision_recall_lift():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = at_k(y, scores, np.zeros(10), k=2)
    assert r["hits"] == 1 and r["precision"] == 0.5 and r["recall"] == 0.5
    assert abs(r["lift"] - 0.5 / 0.2) < 1e-12


def test_ties_are_broken_by_the_fixed_order_not_by_position():
    scores = np.array([1.0, 1.0, 1.0, 0.0])
    tiebreak = np.array([0.9, 0.1, 0.5, 0.0])
    assert rank_order(scores, tiebreak).tolist() == [1, 2, 0, 3]


def test_rule_1_ranks_feature_1_before_shared_address():
    df = pd.DataFrame({"flagged_neighbour_companies": [0, 1, 0],
                       "shared_address_count": [50, 0, 3]})
    s = baseline_scores(df)["rule_neighbour_then_address"]
    assert s[1] > s[0] > s[2]
    assert (baseline_scores(df)["rule_address"] == [50, 0, 3]).all()


def test_a_group_never_spans_two_folds():
    groups = np.repeat(np.arange(40), 3)
    fold = assign_folds(groups)
    per_group = pd.Series(fold).groupby(groups).nunique()
    assert (per_group == 1).all()
    assert set(fold) == set(range(5))


def test_pep_only_drops_sanctions_only_positives():
    df = _synthetic()
    pep = pep_only_frame(df)
    assert (pep["label_source"] != "sanctions").all()
    assert pep["label_pep"].sum() == (df["label_source"] == "pep").sum()


def test_cross_validate_scores_every_row_and_beats_random_on_signal():
    df = _synthetic(n=900)
    rows, scores, fold, imp = cross_validate(df, "label", "main", make_models(rf_estimators=30),
                                             importance=True)
    assert {r["method"] for r in rows} == set(MODELS + BASELINES)
    for m in MODELS:
        assert not np.isnan(scores[m]).any()
    by = {r["method"]: r for r in rows}
    assert by["random_forest"]["pr_auc_mean"] > by["random"]["pr_auc_mean"]
    assert list(imp["feature"]) == ["flagged_neighbour_companies", "shared_address_count",
                                    "degree", "component_size"]


def test_run_writes_all_outputs(tmp_path):
    feats = tmp_path / "features.csv"
    _synthetic().to_csv(feats, index=False)
    preds = tmp_path / "predictions.csv"
    run(feats, tmp_path / "results", preds, models=make_models(rf_estimators=20))
    for name in ("metrics.csv", "metrics.md", "feature_importance.csv",
                 "pr_curves.png", "feature_importance.png"):
        assert (tmp_path / "results" / name).exists(), name
    p = pd.read_csv(preds)
    assert {"fold", "score_logistic_regression", "score_random_forest",
            "score_random_forest_pep"} <= set(p.columns)
    md = (tmp_path / "results" / "metrics.md").read_text(encoding="utf-8")
    assert "Does each model beat each baseline?" in md
