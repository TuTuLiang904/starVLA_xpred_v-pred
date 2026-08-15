"""Aggregate the eight LIBERO-plus FastWAM shard result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    overall = {"overall": {"total_count": 0, "success_count": 0}}
    for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
        files = sorted(log_dir.glob(f"{suite}_*.json"))
        if not files:
            raise FileNotFoundError(f"No completed shard result for {suite} in {log_dir}")
        for path in files:
            with path.open(encoding="utf-8") as f:
                shard = json.load(f)
            for category, values in shard.items():
                target = overall.setdefault(category, {"total_count": 0, "success_count": 0})
                target["total_count"] += int(values["total_count"])
                target["success_count"] += int(values["success_count"])
                overall["overall"]["total_count"] += int(values["total_count"])
                overall["overall"]["success_count"] += int(values["success_count"])

    for values in overall.values():
        values["success_rate"] = values["success_count"] / values["total_count"]
    with (log_dir / "overall_results.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    columns = [
        ("Camera", "Camera Viewpoints"),
        ("Robot", "Robot Initial States"),
        ("Language", "Language Instructions"),
        ("Light", "Light Conditions"),
        ("Background", "Background Textures"),
        ("Noise", "Sensor Noise"),
        ("Layout", "Objects Layout"),
        ("Total", "overall"),
    ]
    header = " | ".join(label for label, _ in columns)
    values = " | ".join(f"{overall[key]['success_rate'] * 100:.1f}%" for _, key in columns)
    counts = " | ".join(
        f"{overall[key]['success_count']}/{overall[key]['total_count']}" for _, key in columns
    )
    summary = "LIBERO-Plus zero-shot results (%)\n"
    summary += header + "\n"
    summary += values + "\n"
    summary += counts + "\n"
    (log_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
