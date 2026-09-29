#!/usr/bin/env python3
"""Create a readable per-task summary from RoboTwin result files.

RoboTwin writes one plain result file per task, while the launcher log contains
ANSI terminal output.  This utility intentionally reads only ``_result_*.txt``
files and emits plain UTF-8 TXT/CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def all_tasks(start_eval: Path) -> list[str]:
    text = start_eval.read_text(encoding="utf-8")
    match = re.search(r"ROBOTWIN_ALL_TASKS=\((.*?)\)", text, flags=re.S)
    if match is None:
        raise RuntimeError(f"Could not find ROBOTWIN_ALL_TASKS in {start_eval}")
    return re.findall(r"^\s+([a-z0-9_]+)\s*$", match.group(1), flags=re.M)


def latest_result(results_root: Path, task: str, mode: str, run_id: str) -> Path | None:
    suffix = "clean" if mode == "demo_clean" else "random"
    pattern = (
        f"{task}/model2robotwin_interface/{mode}/{run_id}/"
        f"*/_result_{suffix}.txt"
    )
    candidates = list(results_root.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_rate(path: Path) -> float:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return float(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--start-eval", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["demo_clean", "demo_randomized"], required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    tasks = all_tasks(args.start_eval)
    rows: list[dict[str, str | float | int | None]] = []
    for task in tasks:
        result = latest_result(args.results_root, task, args.mode, args.run_id)
        if result is None:
            rows.append({"task": task, "successes": "", "episodes": "", "success_rate": "", "status": "MISSING", "result_file": ""})
            continue
        rate = read_rate(result)
        episodes = 50
        rows.append({
            "task": task,
            "successes": int(round(rate * episodes)),
            "episodes": episodes,
            "success_rate": rate,
            "status": "OK",
            "result_file": str(result),
        })

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    txt_path = args.output_prefix.with_suffix(".txt")
    fields = ["task", "successes", "episodes", "success_rate", "status", "result_file"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row["status"] == "OK"]
    rates = [float(row["success_rate"]) for row in completed]
    mean_completed = sum(rates) / len(rates) if rates else 0.0
    mean_all = sum(rates) / len(rows) if rows else 0.0
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"run_id: {args.run_id}\nmode: {args.mode}\n")
        handle.write(f"completed_tasks: {len(completed)}/{len(rows)}\n")
        handle.write(f"mean_completed: {mean_completed:.4f} ({mean_completed * 100:.2f}%)\n")
        handle.write(f"mean_all_missing_as_zero: {mean_all:.4f} ({mean_all * 100:.2f}%)\n\n")
        handle.write("task\tsuccesses/episodes\tsuccess_rate\tstatus\n")
        for row in rows:
            if row["status"] == "OK":
                handle.write(f"{row['task']}\t{row['successes']}/{row['episodes']}\t{float(row['success_rate']) * 100:.1f}%\tOK\n")
            else:
                handle.write(f"{row['task']}\t-\t-\tMISSING\n")

    print(f"TXT: {txt_path}")
    print(f"CSV: {csv_path}")
    print(f"completed: {len(completed)}/{len(rows)}")
    print(f"mean_completed: {mean_completed * 100:.2f}%")
    print(f"mean_all_missing_as_zero: {mean_all * 100:.2f}%")


if __name__ == "__main__":
    main()
