#!/usr/bin/env python3
"""Estimate wall-clock completion time of run_local_8gpu_eval.sh from logs.

Usage:
    python estimate_local_8gpu_eta.py --output-dir results/libero_plus_8gpu_v_ft
"""

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path


WORKER_RE = re.compile(r"^(?P<suite>.+)_(?P<start>\d+)_(?P<end>\d+)\.log$")
TIMESTAMP_RE = re.compile(r"(?P<month>\d{2})/(?P<day>\d{2}) \[(?P<time>\d{2}:\d{2}:\d{2})\]")
COMPLETED_RE = re.compile(r"# episodes completed so far: (?P<count>\d+)")


def parse_timestamp(line: str, year: int) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    return datetime.strptime(
        f"{year}-{match.group('month')}-{match.group('day')} {match.group('time')}", "%Y-%m-%d %H:%M:%S"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="8-GPU launcher output directory")
    args = parser.parse_args()

    log_dir = Path(args.output_dir) / "workers"
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Worker log directory not found: {log_dir}")

    estimates: list[timedelta] = []
    print("worker | completed/total | sec/task | estimated remaining")
    print("-" * 92)
    for log_file in sorted(log_dir.glob("*.log")):
        worker_match = WORKER_RE.match(log_file.name)
        if not worker_match:
            continue
        total = int(worker_match.group("end")) - int(worker_match.group("start"))
        first_time = None
        last_completed_time = None
        completed = 0
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            timestamp = parse_timestamp(line, datetime.now().year)
            if timestamp is not None and first_time is None:
                first_time = timestamp
            match = COMPLETED_RE.search(line)
            if match:
                completed = max(completed, int(match.group("count")))
                if timestamp is not None:
                    last_completed_time = timestamp

        if first_time is None or last_completed_time is None or completed == 0:
            print(f"{log_file.stem} | {completed}/{total} | warming up | --")
            continue

        # Both timestamps come from the same worker log, so this duration is
        # immune to a mismatch between the log's local timezone and the
        # launcher machine's timezone.
        elapsed = max((last_completed_time - first_time).total_seconds(), 1.0)
        seconds_per_task = elapsed / completed
        remaining = timedelta(seconds=(total - completed) * seconds_per_task)
        estimates.append(remaining)
        print(
            f"{log_file.stem} | {completed}/{total} | {seconds_per_task:.1f} | "
            f"{str(remaining).split('.')[0]}"
        )

    if estimates:
        remaining = max(estimates)
        print("-" * 92)
        print(f"Estimated all-shard remaining time: {str(remaining).split('.')[0]}.")
        print("Estimate stabilizes after roughly 30-50 completed tasks per worker.")
    else:
        print("No completed task yet; run this again after workers finish their first tasks.")


if __name__ == "__main__":
    main()
