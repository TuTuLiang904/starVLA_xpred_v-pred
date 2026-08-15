import json
import os
from pathlib import Path

log_dir = os.environ.get("LOG_DIR")
if not log_dir:
    raise ValueError("LOG_DIR must point to the LIBERO-Plus run's logs directory.")

log_dir = Path(log_dir)

task_suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
overall_results = {"overall": {"total_count": 0, "success_count": 0}}
for task_suite in task_suites:
    shard_files = sorted(log_dir.glob(f"{task_suite}*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No completed result JSON found for {task_suite} in {log_dir}.")
    for shard_file in shard_files:
        with open(shard_file, encoding="utf-8") as f:
            results = json.load(f)
        for item in results:
            overall_results["overall"]["total_count"] += results[item]["total_count"]
            overall_results["overall"]["success_count"] += results[item]["success_count"]
            if item not in overall_results:
                overall_results[item] = results[item]
            else:
                overall_results[item]["total_count"] += results[item]["total_count"]
                overall_results[item]["success_count"] += results[item]["success_count"]

for category in overall_results:
    overall_results[category]["success_rate"] = float(overall_results[category]["success_count"]) / float(
        overall_results[category]["total_count"]
    )

with open(log_dir / "overall_results.json", "w", encoding="utf-8") as f:
    json.dump(overall_results, f, indent=2)

# Match the column order used by the LIBERO-Plus paper / leaderboard.
paper_columns = [
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
    ("Total", "overall"),
]

header = " | ".join(label for label, _ in paper_columns)
values = " | ".join(f"{overall_results[key]['success_rate'] * 100:.1f}" for _, key in paper_columns)
summary = f"LIBERO-Plus zero-shot results (%)\\n{header}\\n{values}\\n"

with open(log_dir / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(summary, end="")
