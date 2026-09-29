"""RobotWin action-manifest utilities used by all phase-II probes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


MANIFEST_FIELDS = ("task", "episode", "hdf5_path", "anchor", "horizon", "action_dim")


def _episodes(root: Path, split: str) -> Iterable[tuple[str, Path]]:
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data_dir = task_dir / split / "data"
        for path in sorted(data_dir.glob("*.hdf5")):
            yield task_dir.name, path


def build_robotwin_manifest(
    dataset_root: str | Path,
    output_csv: str | Path,
    *,
    split: str = "demo_clean",
    horizon: int = 32,
    anchors_per_task: int = 40,
    seed: int = 20260918,
) -> Path:
    """Create a deterministic, task-balanced manifest with no padded chunks.

    The stored anchor is the first action of a complete future chunk.  Keeping
    padding out is essential: repeated terminal actions create artificial low
    rank and would invalidate the local-manifold measurements.
    """
    root, output = Path(dataset_root), Path(output_csv)
    if not root.is_dir():
        raise FileNotFoundError(f"RobotWin dataset root does not exist: {root}")
    if horizon < 2 or anchors_per_task < 1:
        raise ValueError("horizon must be >=2 and anchors_per_task must be >=1")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    by_task: dict[str, list[tuple[Path, int, int]]] = {}
    for task, path in _episodes(root, split):
        with h5py.File(path, "r") as h5:
            actions = h5["joint_action"]["vector"]
            length, action_dim = int(actions.shape[0]), int(actions.shape[1])
        # reserve a complete chunk, unlike the training loader which pads it.
        if length >= horizon:
            by_task.setdefault(task, []).append((path, length - horizon + 1, action_dim))
    if not by_task:
        raise RuntimeError(f"No complete action chunks in {root} / {split}")

    for task, episodes in sorted(by_task.items()):
        candidates: list[tuple[Path, int, int]] = []
        # Allocate candidates over episodes first, so one long trajectory does
        # not dominate a task's local geometry.
        for path, count, action_dim in episodes:
            take = min(max(1, int(np.ceil(anchors_per_task / len(episodes))) * 3), count)
            anchors = rng.choice(count, size=take, replace=False)
            candidates.extend((path, int(a), action_dim) for a in anchors)
        chosen = rng.choice(len(candidates), size=min(anchors_per_task, len(candidates)), replace=False)
        for index in np.sort(chosen):
            path, anchor, action_dim = candidates[int(index)]
            rows.append({
                "task": task,
                "episode": path.stem,
                "hdf5_path": str(path.resolve()),
                "anchor": anchor,
                "horizon": horizon,
                "action_dim": action_dim,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps({
        "dataset_root": str(root.resolve()), "split": split, "horizon": horizon,
        "anchors_per_task": anchors_per_task, "seed": seed, "num_rows": len(rows),
    }, indent=2) + "\n")
    return output


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"task", "episode", "anchor", "horizon", "action_dim"} - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Manifest {path} is missing fields: {sorted(missing)}")
    return rows


def load_manifest_actions(path: str | Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Return raw action chunks [N,T,D] and their manifest rows."""
    rows = read_manifest(path)
    chunks: list[np.ndarray] = []
    for row in rows:
        start, horizon = int(row["anchor"]), int(row["horizon"])
        if row.get("source", "hdf5") == "lerobot":
            import pyarrow.parquet as pq
            action = np.asarray(pq.read_table(row["parquet_path"], columns=["action"])["action"].to_pylist()[start:start + horizon], dtype=np.float32)
        else:
            with h5py.File(row["hdf5_path"], "r") as h5:
                action = np.asarray(h5["joint_action"]["vector"][start:start + horizon], dtype=np.float32)
        if len(action) != horizon:
            raise RuntimeError(f"Manifest has a padded/incomplete action chunk: {row}")
        chunks.append(action)
    if not chunks:
        raise ValueError(f"Manifest is empty: {path}")
    return np.stack(chunks), rows


def build_lerobot_manifest(
    dataset_root: str | Path, subset_json: str | Path, output_csv: str | Path, *,
    horizon: int, anchors_per_task: int = 40, seed: int = 20260918,
) -> Path:
    """Build a task-balanced manifest from RobotWin's native LeRobot source."""
    import pyarrow.parquet as pq
    root, subset, output = Path(dataset_root), Path(subset_json), Path(output_csv)
    selected = json.loads(subset.read_text())["episode_indices"]
    episodes_meta = {json.loads(line)["episode_index"]: json.loads(line)
                     for line in (root / "meta/episodes.jsonl").read_text().splitlines()}
    rng = np.random.default_rng(seed); grouped: dict[int, list[tuple[int, Path, int, str]]] = {}
    for episode in selected:
        parquet = root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
        meta = episodes_meta[episode]
        # RoboTwin 2.0 stores 550 demonstrations per physical task; task_index
        # instead labels prompt variants and must not define a manifold group.
        physical_task = int(episode) // 550
        grouped.setdefault(physical_task, []).append((episode, parquet, int(meta["length"]), str(meta["tasks"][0])))
    rows = []
    for task_index, group in sorted(grouped.items()):
        candidates = []
        for episode, parquet, length, instruction in group:
            valid = length - horizon + 1
            if valid > 0:
                take = min(valid, max(2, int(np.ceil(anchors_per_task / len(group))) * 3))
                for anchor in rng.choice(valid, take, replace=False):
                    candidates.append((episode, parquet, int(anchor), instruction))
        for idx in rng.choice(len(candidates), min(anchors_per_task, len(candidates)), replace=False):
            episode, parquet, anchor, instruction = candidates[int(idx)]
            rows.append({"source": "lerobot", "task": f"task_{task_index:03d}", "task_index": task_index,
                         "instruction": instruction, "episode": f"episode_{episode:06d}",
                         "parquet_path": str(parquet.resolve()), "anchor": anchor,
                         "horizon": horizon, "action_dim": 14})
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "task", "task_index", "instruction", "episode", "parquet_path", "anchor", "horizon", "action_dim"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return output


def save_action_cache(manifest: str | Path, output: str | Path) -> Path:
    actions, rows = load_manifest_actions(manifest)
    tasks = np.asarray([r["task"] for r in rows])
    np.savez_compressed(output, actions=actions, tasks=tasks, manifest=np.asarray(str(Path(manifest).resolve())))
    return Path(output)
