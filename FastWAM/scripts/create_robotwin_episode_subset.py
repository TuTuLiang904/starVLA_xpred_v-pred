#!/usr/bin/env python3
"""Create a deterministic episode-level RoboTwin subset manifest.

The manifest is consumed by both the training dataset and the T5 cache
precomputation script.  Sampling full episodes preserves each demonstration's
video/action temporal structure; never subsample individual frames here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="data/robotwin2.0/robotwin2.0",
        help="LeRobot RoboTwin dataset directory containing meta/info.json.",
    )
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/robotwin2.0/subsets/half_seed42.json",
        help="Output JSON manifest. Existing files are replaced deterministically.",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    with info_path.open(encoding="utf-8") as f:
        total_episodes = int(json.load(f)["total_episodes"])

    selected_count = round(total_episodes * args.fraction)
    selected_count = min(max(selected_count, 1), total_episodes)
    episode_indices = np.random.default_rng(args.seed).choice(
        total_episodes, size=selected_count, replace=False
    )
    episode_indices.sort()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "fastwam_episode_subset_v1",
        "dataset_dir": str(dataset_dir),
        "total_episodes": total_episodes,
        "selected_episodes": selected_count,
        "fraction": args.fraction,
        "seed": args.seed,
        "episode_indices": episode_indices.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"Wrote {selected_count}/{total_episodes} complete RoboTwin episodes "
        f"({args.fraction:.1%}, seed={args.seed}) to {output_path}"
    )


if __name__ == "__main__":
    main()
