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

import graph_builder  # noqa: E402
import matching  # noqa: E402


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


if __name__ == "__main__":
    main()
