#!/usr/bin/env python3
"""Create the official RoboTwin clean-50 episode manifest.

RoboTwin 2.0 stores each of its 50 task blocks as 550 consecutive episodes:
the first 50 are the clean demonstrations and the following 500 are the
randomized demonstrations.  The LeRobot metadata does not contain a split
column, so this script records that public ordering rule explicitly and keeps
the selected episodes intact (never subsampling frames).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="data/robotwin2.0/robotwin2.0",
        help="RoboTwin LeRobot dataset directory containing meta/info.json.",
    )
    parser.add_argument(
        "--output",
        default="data/robotwin2.0/subsets/clean_50x50.json",
        help="Output manifest path.",
    )
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--episodes-per-task", type=int, default=550)
    parser.add_argument("--clean-per-task", type=int, default=50)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin metadata under {dataset_dir}")
    with info_path.open(encoding="utf-8") as f:
        total = int(json.load(f)["total_episodes"])
    expected = args.tasks * args.episodes_per_task
    if total != expected:
        raise ValueError(
            f"Expected {expected} episodes ({args.tasks} tasks x "
            f"{args.episodes_per_task}), found {total}; refusing an unsafe split."
        )
    with episodes_path.open(encoding="utf-8") as f:
        rows = sum(1 for _ in f)
    if rows != total:
        raise ValueError(f"info.json says {total} episodes but episodes.jsonl has {rows} rows")
    if not 0 < args.clean_per_task <= args.episodes_per_task:
        raise ValueError("clean-per-task must be in (0, episodes-per-task]")

    indices = [
        task * args.episodes_per_task + offset
        for task in range(args.tasks)
        for offset in range(args.clean_per_task)
    ]
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "fastwam_robotwin_split_v1",
        "dataset_dir": str(dataset_dir),
        "total_episodes": total,
        "selected_episodes": len(indices),
        "episode_indices": indices,
        "task_count": args.tasks,
        "episodes_per_task": args.episodes_per_task,
        "clean_per_task": args.clean_per_task,
        "partition_rule": (
            "For task block i (550 consecutive episodes), clean episodes are "
            "[550*i, 550*i+50); randomized episodes are [550*i+50, 550*(i+1))."
        ),
        "note": "The downloaded LeRobot release has no explicit clean/randomized field.",
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(indices)} clean episodes ({args.tasks} tasks x {args.clean_per_task}) to {output}")


if __name__ == "__main__":
    main()
