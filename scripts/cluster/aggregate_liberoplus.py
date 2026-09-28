#!/usr/bin/env python3
"""Aggregate LIBERO+ evaluator results by official perturbation category."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="TurboVLA scratch root")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LIBERO+ results run name; repeat for suites split across runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    classification_path = (
        root / "simulators/LIBERO-plus/libero/libero/benchmark/task_classification.json"
    )
    classification = json.loads(classification_path.read_text())
    suite_totals = {}
    missing = []

    for suite in SUITES:
        result_paths = [
            root / "results/liberoplus" / run / suite / "results.json"
            for run in args.run
        ]
        result_path = next((path for path in result_paths if path.exists()), None)
        if result_path is None:
            missing.append(suite)
            continue
        results = json.loads(result_path.read_text())
        category_totals = defaultdict(lambda: [0, 0])
        for task in results["tasks"]:
            category = classification[suite][task["task_id"]]["category"]
            category_totals[category][0] += task["successes"]
            category_totals[category][1] += task["episodes"]
        suite_totals[suite] = category_totals

    all_totals = defaultdict(lambda: [0, 0])
    for suite, totals in suite_totals.items():
        print(f"\n## {suite}")
        print("| Category | Successes / episodes | Success rate |")
        print("|---|---:|---:|")
        for category in CATEGORIES:
            category_successes, category_episodes = totals[category]
            all_totals[category][0] += category_successes
            all_totals[category][1] += category_episodes
            rate = 100 * category_successes / category_episodes if category_episodes else 0
            print(
                f"| {category} | {category_successes:,} / {category_episodes:,} | "
                f"{rate:.2f}% |"
            )
        successes = sum(value[0] for value in totals.values())
        episodes = sum(value[1] for value in totals.values())
        rate = 100 * successes / episodes if episodes else 0
        print(f"| **Total** | **{successes:,} / {episodes:,}** | **{rate:.2f}%** |")

    print("\n## All available suites")
    print("| Category | Successes / episodes | Success rate |")
    print("|---|---:|---:|")
    for category in CATEGORIES:
        category_successes, category_episodes = all_totals[category]
        rate = 100 * category_successes / category_episodes if category_episodes else 0
        print(f"| {category} | {category_successes:,} / {category_episodes:,} | {rate:.2f}% |")
    successes = sum(value[0] for value in all_totals.values())
    episodes = sum(value[1] for value in all_totals.values())
    rate = 100 * successes / episodes if episodes else 0
    print(f"| **Total** | **{successes:,} / {episodes:,}** | **{rate:.2f}%** |")
    if missing:
        print(f"\nMissing results.json: {', '.join(missing)}")


if __name__ == "__main__":
    main()
