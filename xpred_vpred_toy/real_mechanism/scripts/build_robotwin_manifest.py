#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from real_mechanism.data import build_robotwin_manifest, save_action_cache

ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    p = argparse.ArgumentParser(description="Build a padding-free, task-balanced RobotWin action manifest.")
    p.add_argument("--dataset-root", type=Path, default=ROOT.parent / "LiLa-WAM/Datasets")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/robotwin_clean_h32_manifest.csv")
    p.add_argument("--split", default="demo_clean")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--anchors-per-task", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--cache-actions", action="store_true")
    a = p.parse_args()
    manifest = build_robotwin_manifest(a.dataset_root, a.output, split=a.split, horizon=a.horizon,
                                       anchors_per_task=a.anchors_per_task, seed=a.seed)
    print(f"manifest: {manifest}")
    if a.cache_actions:
        cache = save_action_cache(manifest, manifest.with_suffix(".npz"))
        print(f"action cache: {cache}")

if __name__ == "__main__": main()
