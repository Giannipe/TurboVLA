#!/usr/bin/env python3
"""Validate and summarize the four released TurboVLA LIBERO evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "libero_spatial": "spatial.pth",
    "libero_object": "object.pth",
    "libero_goal": "goal.pth",
    "libero_10": "long.pth",
}
EXPECTED_PROTOCOL = {
    "num_trials_per_task": 50,
    "num_open_loop_steps": 12,
    "seed": 7,
    "precision": "bf16",
}
PAPER_SUCCESS_RATES = {
    "libero_spatial": 0.992,
    "libero_object": 0.998,
    "libero_goal": 0.974,
    "libero_10": 0.942,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--unified-checkpoint", action="store_true",
                        help="Expect the September turbovla_libero.pth release for all suites")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    total_successes = 0
    total_episodes = 0
    unified_path = None

    for suite, checkpoint_name in EXPECTED.items():
        if args.unified_checkpoint:
            checkpoint_name = "turbovla_libero.pth"
        path = args.result_dir / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing suite result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_suite_name") != suite:
            raise ValueError(f"{path}: unexpected suite {payload.get('task_suite_name')!r}")
        if Path(payload.get("ckpt_path", "")).name != checkpoint_name:
            raise ValueError(f"{path}: expected checkpoint {checkpoint_name}")
        if args.unified_checkpoint:
            current_path = Path(payload["ckpt_path"]).resolve()
            if unified_path is not None and current_path != unified_path:
                raise ValueError("All suites must use the same unified checkpoint path")
            unified_path = current_path
        mismatches = {
            key: (expected, payload.get(key))
            for key, expected in EXPECTED_PROTOCOL.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"{path}: protocol mismatch: {mismatches}")

        tasks = payload.get("tasks", [])
        episodes = int(payload.get("total_episodes", 0))
        successes = int(payload.get("total_successes", 0))
        if len(tasks) != 10 or episodes != 500:
            raise ValueError(f"{path}: expected 10 tasks and 500 episodes, got {len(tasks)} and {episodes}")
        if any(int(task.get("episodes", 0)) != 50 for task in tasks):
            raise ValueError(f"{path}: every task must contain exactly 50 trials")
        if {int(task["task_id"]) for task in tasks} != set(range(10)):
            raise ValueError(f"{path}: expected unique task IDs 0..9")
        if any(not 0 <= int(task["successes"]) <= 50 for task in tasks):
            raise ValueError(f"{path}: invalid task success count")
        if sum(int(task["successes"]) for task in tasks) != successes:
            raise ValueError(f"{path}: aggregate successes disagree with task results")

        rows.append(
            {
                "suite": suite,
                "checkpoint": checkpoint_name,
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes,
                "paper_success_rate": PAPER_SUCCESS_RATES[suite],
                "delta_percentage_points": 100.0 * (successes / episodes - PAPER_SUCCESS_RATES[suite]),
                "source": str(path),
            }
        )
        total_successes += successes
        total_episodes += episodes

    summary = {
        "script": "turbovla_libero_official_summary",
        "protocol": EXPECTED_PROTOCOL,
        "suites": rows,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "average_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
        "paper_average_success_rate": sum(PAPER_SUCCESS_RATES.values()) / len(PAPER_SUCCESS_RATES),
        "pooled_success_rate": total_successes / total_episodes,
    }
    output = args.output or args.result_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"{row['suite']:16s} {row['successes']:3d}/{row['episodes']}  "
            f"observed={100 * row['success_rate']:6.2f}%  paper={100 * row['paper_success_rate']:5.1f}%  "
            f"delta={row['delta_percentage_points']:+5.2f} pp"
        )
    average_delta = 100.0 * (
        summary["average_success_rate"] - summary["paper_average_success_rate"]
    )
    print(
        f"{'average':16s} {'':7s}  observed={100 * summary['average_success_rate']:6.2f}%  "
        f"paper={100 * summary['paper_average_success_rate']:6.2f}%  delta={average_delta:+5.2f} pp"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
