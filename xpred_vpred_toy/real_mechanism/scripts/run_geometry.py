#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from real_mechanism.data import load_manifest_actions
from real_mechanism.geometry import GeometryConfig, analyse_local_geometry, write_rows

def main() -> None:
    p = argparse.ArgumentParser(description="Compute task-local x/epsilon/v spectral and tangent diagnostics.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--k-neighbors", type=int, default=16)
    p.add_argument("--tangent-dim", type=int, default=10)
    p.add_argument("--max-anchors-per-task", type=int)
    p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args()
    actions, rows = load_manifest_actions(a.manifest)
    tasks = __import__("numpy").asarray([r["task"] for r in rows])
    cfg = GeometryConfig(a.k_neighbors, a.tangent_dim, a.max_anchors_per_task, a.seed)
    global_rows, local_rows = analyse_local_geometry(actions, tasks, cfg)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(global_rows, a.output_dir / "global_task.csv")
    write_rows(local_rows, a.output_dir / "local_anchor.csv")
    print(f"wrote {len(global_rows)} global and {len(local_rows)} local rows to {a.output_dir}")

if __name__ == "__main__": main()
