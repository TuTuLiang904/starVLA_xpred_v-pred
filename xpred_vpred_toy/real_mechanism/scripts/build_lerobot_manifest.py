#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from real_mechanism.data import build_lerobot_manifest, save_action_cache

TOY = Path(__file__).resolve().parents[2]; FAST = TOY.parent / "FastWAM"
def main():
    p = argparse.ArgumentParser(description="Build native three-view LeRobot RobotWin manifest for StarVLA/FastWAM.")
    p.add_argument("--dataset-root", type=Path, default=FAST / "data/robotwin2.0/robotwin2.0")
    p.add_argument("--subset", type=Path, default=FAST / "data/robotwin2.0/subsets/clean_50x50.json")
    p.add_argument("--output", type=Path, default=TOY / "real_mechanism/outputs/lerobot_clean_h50_manifest.csv")
    p.add_argument("--horizon", type=int, default=50); p.add_argument("--anchors-per-task", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260918); p.add_argument("--cache-actions", action="store_true")
    a = p.parse_args(); path = build_lerobot_manifest(a.dataset_root, a.subset, a.output, horizon=a.horizon, anchors_per_task=a.anchors_per_task, seed=a.seed)
    print(path)
    if a.cache_actions: print(save_action_cache(path, path.with_suffix(".npz")))
if __name__ == "__main__": main()
