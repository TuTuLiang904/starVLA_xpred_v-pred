"""Architecture-agnostic statistics for endpoint-recovery and NFE archives."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _summary(error: np.ndarray) -> tuple[float, float]:
    per_anchor = error.reshape(error.shape[0], -1).mean(axis=1)
    sem = 0.0 if len(per_anchor) < 2 else float(per_anchor.std(ddof=1) / np.sqrt(len(per_anchor)))
    return float(per_anchor.mean()), sem


def summarise_recovery(archive: str | Path, output_csv: str | Path) -> None:
    """Summarise a standard endpoint archive written by a backend.

    Required arrays: endpoint [M,S,R,N,T,D], target [N,T,D], sigma [S],
    model [M].  M is normally x/v/x-vloss, S is the common sigma grid, and R
    is the paired initial-noise seed axis.
    """
    data = np.load(archive, allow_pickle=False)
    required = {"endpoint", "target", "sigma", "model"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{archive} lacks arrays {sorted(missing)}")
    endpoint, target = data["endpoint"], data["target"]
    rows = []
    for m, name in enumerate(data["model"].astype(str)):
        for s, sigma in enumerate(data["sigma"]):
            error = (endpoint[m, s] - target[None]) ** 2  # seeds,N,T,D
            mse, sem = _summary(error.mean(axis=0))
            seed_variance = float(endpoint[m, s].var(axis=0).mean())
            rows.append({"model": name, "sigma": float(sigma), "endpoint_mse": mse,
                         "endpoint_rmse": float(np.sqrt(mse)), "task_sem": sem,
                         "endpoint_seed_variance": seed_variance})
    with Path(output_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def summarise_nfe(archive: str | Path, output_csv: str | Path) -> None:
    """Summarise a sample archive: sample [M,K,R,N,T,D], target [N,T,D]."""
    data = np.load(archive, allow_pickle=False)
    required = {"sample", "target", "nfe", "model"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{archive} lacks arrays {sorted(missing)}")
    rows = []
    for m, name in enumerate(data["model"].astype(str)):
        for k, nfe in enumerate(data["nfe"]):
            samples = data["sample"][m, k]
            # Seeds are paired Monte-Carlo draws, not independent tasks.  First
            # average their squared error, then compute the task/anchor SEM.
            mse, sem = _summary(((samples - data["target"]) ** 2).mean(axis=0))
            velocity = np.diff(samples, axis=-2)
            jerk = np.diff(samples, n=2, axis=-2)
            rows.append({"model": name, "nfe": int(nfe), "replay_mse": mse, "task_sem": sem,
                         "seed_variance": float(samples.var(axis=0).mean()),
                         "action_delta_sq": float(np.square(velocity).mean()),
                         "jerk_sq": float(np.square(jerk).mean()) if jerk.size else float("nan")})
    with Path(output_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
