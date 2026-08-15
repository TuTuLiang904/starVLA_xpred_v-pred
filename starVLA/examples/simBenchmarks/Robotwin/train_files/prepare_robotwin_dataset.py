#!/usr/bin/env python3
"""Create zero-copy StarVLA views of the public RoboTwin 2.0 dataset.

The downloaded public LeRobot-v2.1 dataset contains 50 consecutive
550-episode blocks, matching the RoboTwin 2.0 data-scaling protocol:
the first 50 episodes of a block are the clean demonstrations and the next
500 are the randomized demonstrations.  The release itself does not retain a
``split`` column, so this script makes that ordering rule explicit in small,
versioned manifests rather than silently pretending it was stored as metadata.

No parquet or video data is copied.  The generated views only contain metadata
and symlinks back to the source data, so both training modes share the same
149-GB source dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
]

MODALITY = {
    "state": {
        "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
        "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
    },
    "action": {
        "left_joints": {"start": 0, "end": 6, "original_key": "action"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
        "right_joints": {"start": 7, "end": 13, "original_key": "action"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def link(target: Path, path: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.symlink_to(target)


def make_stats(source_stats: Path, destination: Path) -> None:
    """Translate source global statistics to StarVLA's cached stats format."""
    with source_stats.open(encoding="utf-8") as f:
        src = json.load(f)
    statistics = {}
    for field in ("state", "action"):
        values = src[field]["default"]
        statistics["observation.state" if field == "state" else "action"] = {
            "min": values["global_min"],
            "max": values["global_max"],
            "mean": values["global_mean"],
            "std": values["global_std"],
            "q01": values["global_q01"],
            "q99": values["global_q99"],
        }
    atomic_json(destination, {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
    })


def make_view(output_root: Path, source: Path, split: str, episode_ids: list[int], source_stats: Path) -> None:
    view = output_root / split
    meta = view / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    link(source / "data", view / "data")
    link(source / "videos", view / "videos")
    for filename in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        link(source / "meta" / filename, meta / filename)
    atomic_json(meta / "modality.json", MODALITY)
    make_stats(source_stats, meta / "stats_gr00t.json")
    atomic_json(meta / "episode_indices.json", {
        "format": "starvla_robotwin_split_v1",
        "source_dataset": str(source),
        "split": split,
        "episode_count": len(episode_ids),
        "episode_indices": episode_ids,
        "partition_rule": "For task block i (550 consecutive episodes): clean=[550*i,550*i+50), randomized=[550*i+50,550*(i+1)).",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Extracted public RoboTwin directory containing data/meta/videos.")
    parser.add_argument("--source-stats", type=Path, required=True, help="Source dataset_stats.json with global state/action statistics.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination for tiny zero-copy StarVLA dataset views.")
    args = parser.parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    required = [source / "data", source / "videos", source / "meta" / "info.json", source / "meta" / "episodes.jsonl", args.source_stats]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required RoboTwin dataset files:\n" + "\n".join(missing))

    with (source / "meta" / "info.json").open(encoding="utf-8") as f:
        info = json.load(f)
    total = int(info["total_episodes"])
    expected = len(TASKS) * 550
    if total != expected:
        raise ValueError(f"Expected {expected} episodes (50 tasks x 550), found {total}; refusing an unsafe split.")
    with (source / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        episode_count = sum(1 for _ in f)
    if episode_count != total:
        raise ValueError(f"info.json says {total} episodes but episodes.jsonl has {episode_count} rows.")

    clean = [task_i * 550 + offset for task_i in range(len(TASKS)) for offset in range(50)]
    randomized = [task_i * 550 + offset for task_i in range(len(TASKS)) for offset in range(50, 550)]
    full = list(range(total))
    make_view(output_root, source, "clean", clean, args.source_stats.resolve())
    make_view(output_root, source, "randomized", randomized, args.source_stats.resolve())
    make_view(output_root, source, "full", full, args.source_stats.resolve())
    atomic_json(output_root / "split_report.json", {
        "format": "starvla_robotwin_split_report_v1",
        "source_dataset": str(source),
        "source_total_episodes": total,
        "task_count": len(TASKS),
        "episodes_per_task": 550,
        "clean_per_task": 50,
        "randomized_per_task": 500,
        "clean_total": len(clean),
        "randomized_total": len(randomized),
        "full_total": len(full),
        "task_order": TASKS,
        "important_note": "The downloaded LeRobot re-release has no explicit clean/randomized field. This view records the public data-scaling ordering convention as an auditable partition rule.",
    })
    print(f"Prepared zero-copy views in {output_root}")
    print(f"  clean={len(clean)} episodes (50 tasks x 50)")
    print(f"  randomized={len(randomized)} episodes (50 tasks x 500)")
    print(f"  full={len(full)} episodes")


if __name__ == "__main__":
    main()
