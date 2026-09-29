"""Local action-manifold diagnostics, implemented without training dependencies."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GeometryConfig:
    # The formal manifest uses 40 anchors per task.  Keep k comfortably below
    # that count so every task contributes local neighborhoods by default.
    k_neighbors: int = 16
    tangent_dim: int = 10
    max_anchors_per_task: int | None = None
    seed: int = 20260918


def _svd_metrics(matrix: np.ndarray) -> dict[str, float]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= np.finfo(np.float64).eps:
        return {"stable_rank": 0.0, "participation_ratio": 0.0, "rank90": 0.0, "rank95": 0.0, "rank99": 0.0}
    p = energy / total
    cumulative = np.cumsum(p)
    return {
        "stable_rank": float(total / max(float(energy.max()), np.finfo(np.float64).eps)),
        "participation_ratio": float(1.0 / np.square(p).sum()),
        "rank90": float(np.searchsorted(cumulative, .90) + 1),
        "rank95": float(np.searchsorted(cumulative, .95) + 1),
        "rank99": float(np.searchsorted(cumulative, .99) + 1),
    }


def _neighbors(x: np.ndarray, k: int) -> np.ndarray:
    """Exact kNN.  N is deliberately modest in manifests (about 2k)."""
    n = len(x)
    k = min(k, n - 1)
    x2 = np.einsum("nd,nd->n", x, x)
    distances = x2[:, None] + x2[None, :] - 2.0 * (x @ x.T)
    np.fill_diagonal(distances, np.inf)
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


def _squared_distances(x: np.ndarray) -> np.ndarray:
    x2 = np.einsum("nd,nd->n", x, x)
    distances = x2[:, None] + x2[None, :] - 2.0 * (x @ x.T)
    np.fill_diagonal(distances, np.inf)
    return distances


def _local_basis(points: np.ndarray, dimension: int) -> tuple[np.ndarray, float, dict[str, float]]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    d = min(dimension, vt.shape[0])
    basis = vt[:d].T
    reconstruction = centered @ basis @ basis.T
    residual = float(np.mean(np.square(centered - reconstruction)))
    return basis, residual, _svd_metrics(points)


def _two_nn_id(distances: np.ndarray) -> float:
    # Facco et al. TwoNN estimator; robust median form for a finite sample.
    ordered = np.sqrt(np.maximum(np.partition(distances, kth=1, axis=1)[:, :2], 0.0))
    ratios = np.maximum(ordered[:, 1], 1e-12) / np.maximum(ordered[:, 0], 1e-12)
    logs = np.log(np.maximum(ratios, 1.0 + 1e-12))
    return float(1.0 / max(np.mean(logs), 1e-12))


def analyse_local_geometry(actions: np.ndarray, tasks: np.ndarray, config: GeometryConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Analyse x-neighbourhoods, then evaluate x/eps/v on matching members."""
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [N,T,D], got {actions.shape}")
    n, _, _ = actions.shape
    flat_raw = actions.reshape(n, -1).astype(np.float64)
    # Neutral action standardisation: this is data-only, never a model's own
    # normalizer, so the geometry comparison is not architecture dependent.
    mean, std = flat_raw.mean(0), flat_raw.std(0)
    x = (flat_raw - mean) / np.maximum(std, 1e-6)
    rng = np.random.default_rng(config.seed)
    eps = rng.standard_normal(x.shape)
    v = x - eps
    targets = {"x": x, "epsilon": eps, "v": v}
    global_rows: list[dict[str, object]] = []
    local_rows: list[dict[str, object]] = []
    for task in sorted(set(tasks.tolist())):
        indices = np.flatnonzero(tasks == task)
        if config.max_anchors_per_task is not None and len(indices) > config.max_anchors_per_task:
            indices = np.sort(rng.choice(indices, config.max_anchors_per_task, replace=False))
        if len(indices) <= config.k_neighbors:
            continue
        x_task = x[indices]
        distances = _squared_distances(x_task)
        neighbours = np.argpartition(distances, kth=config.k_neighbors - 1, axis=1)[:, :config.k_neighbors]
        nearest = np.argmin(distances, axis=1)
        for name, values in targets.items():
            metrics = _svd_metrics(values[indices])
            metrics.update({"scope": "global_task", "task": task, "target": name, "anchor": -1,
                            "two_nn_id": _two_nn_id(_squared_distances(values[indices])),
                            "tangent_variation": float("nan"), "linear_residual": float("nan")})
            global_rows.append(metrics)
        bases: dict[str, list[np.ndarray]] = {name: [] for name in targets}
        residuals: dict[str, list[float]] = {name: [] for name in targets}
        metrics_by_target: dict[str, list[dict[str, float]]] = {name: [] for name in targets}
        for local_i, neigh in enumerate(neighbours):
            for name, values in targets.items():
                local_values = values[indices[neigh]]
                basis, residual, metrics = _local_basis(local_values, config.tangent_dim)
                metrics["two_nn_id"] = _two_nn_id(_squared_distances(local_values))
                bases[name].append(basis)
                residuals[name].append(residual)
                metrics_by_target[name].append(metrics)
        for name in targets:
            for local_i, metrics in enumerate(metrics_by_target[name]):
                b0, b1 = bases[name][local_i], bases[name][nearest[local_i]]
                d = min(b0.shape[1], b1.shape[1])
                overlap = np.linalg.norm(b0[:, :d].T @ b1[:, :d], ord="fro") ** 2 / max(d, 1)
                row: dict[str, object] = dict(metrics)
                row.update({"scope": "local", "task": task, "target": name,
                            "anchor": int(indices[local_i]), "two_nn_id": metrics["two_nn_id"],
                            "tangent_variation": float(1.0 - overlap),
                            "linear_residual": residuals[name][local_i]})
                local_rows.append(row)
    return global_rows, local_rows


def write_rows(rows: list[dict[str, object]], output: str | Path) -> None:
    if not rows:
        raise RuntimeError("No geometry rows were produced; increase manifest size or lower k.")
    fieldnames = ["scope", "task", "target", "anchor", "stable_rank", "participation_ratio", "rank90", "rank95", "rank99", "two_nn_id", "tangent_variation", "linear_residual"]
    with Path(output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
