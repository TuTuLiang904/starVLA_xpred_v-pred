"""Strict archive contract for framework-specific checkpoint adapters.

The three large projects remain untouched.  A backend receives a manifest and
writes portable NPZ archives; the same statistics code then applies to every
framework.  `SyntheticBackend` exists only for deterministic CI/smoke tests.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .data import load_manifest_actions

SIGMA_GRID = np.asarray([.01, .05, .10, .20, .40, .60, .80, .95, .99], dtype=np.float32)
NFE_GRID = np.asarray([1, 2, 4, 8, 16], dtype=np.int64)


class SyntheticBackend:
    """Deterministic endpoint/NFE oracle used to validate archive dimensions."""
    models = np.asarray(["v_prediction", "x_prediction", "x_prediction_v_loss"])

    def __init__(self, manifest: str | Path, seeds: int = 3):
        self.target, _ = load_manifest_actions(manifest)
        self.target = self.target.astype(np.float32)
        self.seeds = seeds

    def recovery(self, output: str | Path) -> Path:
        rng = np.random.default_rng(0)
        m, s, n, t, d = len(self.models), len(SIGMA_GRID), *self.target.shape
        endpoint = np.empty((m, s, self.seeds, n, t, d), dtype=np.float32)
        for mi in range(m):
            for si, sigma in enumerate(SIGMA_GRID):
                # Deliberately gives x lower high-noise error, so smoke output
                # is visibly non-degenerate.  It is not an experiment result.
                scale = (1.3 if mi == 0 else 0.75 + .15 * mi) * float(sigma)
                endpoint[mi, si] = self.target + rng.normal(0, scale, (self.seeds, n, t, d))
        np.savez_compressed(output, endpoint=endpoint, target=self.target, sigma=SIGMA_GRID, model=self.models)
        return Path(output)

    def nfe(self, output: str | Path) -> Path:
        rng = np.random.default_rng(1)
        m, k, n, t, d = len(self.models), len(NFE_GRID), *self.target.shape
        sample = np.empty((m, k, self.seeds, n, t, d), dtype=np.float32)
        for mi in range(m):
            for ki, nfe in enumerate(NFE_GRID):
                scale = (1.2 if mi == 0 else .8 + .1 * mi) / np.sqrt(float(nfe))
                sample[mi, ki] = self.target + rng.normal(0, scale, (self.seeds, n, t, d))
        np.savez_compressed(output, sample=sample, target=self.target, nfe=NFE_GRID, model=self.models)
        return Path(output)
