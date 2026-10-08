"""
Runs the analysis pipeline (docs/implementation_plan.md, Phases 1-4) on the
cleaned files in data/interim/. Does NOT run preprocessing -- raw data is not
available, and run_preprocessing.py would overwrite data/interim/.

Run from the project root:
    python scripts/run_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import features  # noqa: E402
import graph_builder  # noqa: E402
import matching  # noqa: E402
import train  # noqa: E402


def main():
    print("=== Phase 1: Name matching ===")
    _, match_counts = matching.run()
    for k, v in match_counts.items():
        print(f"  {k}: {v}")
    print(f"  -> {matching.MATCHES_OUTPUT}")

    print("\n=== Phase 2: Ownership graph ===")
    stats = graph_builder.run()
    for k, v in stats.items():
        if k != "unmapped":
            print(f"  {k}: {v}")
    print(f"  unmapped strong matches: {len(stats['unmapped'])}")
    print(f"  -> {graph_builder.GRAPH_OUTPUT}")
    print(f"  -> {graph_builder.STATS_OUTPUT}")

    print("\n=== Phase 3: Features and labels ===")
    _, summary = features.run()
    for k in ("n_companies", "n_positive", "pct_positive", "n_positive_weak",
              "label_source", "nonzero", "hub_companies", "hub_positives",
              "n_split_groups", "largest_split_group", "positive_split_groups"):
        print(f"  {k}: {summary[k]}")
    print(f"  -> {features.FEATURES_OUTPUT}")
    print(f"  -> {features.STATS_OUTPUT}")

    print("\n=== Phase 4: Train and compare the models ===")
    metrics, _, positives_per_fold = train.run()
    cols = ["run", "method", "pr_auc_mean", "pr_auc_std", "hits@100", "precision@100"]
    print(metrics[cols].to_string(index=False))
    print(f"  positives per fold: {positives_per_fold}")
    print(f"  -> {train.RESULTS}")
    print(f"  -> {train.PREDICTIONS_OUTPUT}")


if __name__ == "__main__":
    main()
