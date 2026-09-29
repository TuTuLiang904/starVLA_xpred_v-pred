"""Aggregation and paired statistics.

Arms are paired by seed: at a given seed every arm sees the same generative model,
the same initial weights, the same batches, the same noises and the same
timesteps.  Differences are therefore reported as paired quantities, with a
bootstrap interval over the seed-level differences and a Wilcoxon signed-rank
test, rather than as two independent means whose intervals happen to overlap.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(outdir: str | Path) -> list[dict]:
    """Load one or more result directories.

    Accepts several directories separated by ``os.pathsep`` (or a comma), which is
    what lets a re-run of the headline experiments sit beside the older sweeps
    without either being copied: one job is one file named
    ``{experiment}__{arm}__s{seed}.json``, so a later directory *replaces*
    same-named records from an earlier one rather than duplicating them.
    """
    dirs = [
        Path(d)
        for part in str(outdir).split(",")
        for d in str(part).split(os.pathsep)
        if d
    ]
    by_name: dict[str, dict] = {}
    for d in dirs:
        for p in sorted(d.glob("*.json")):
            try:
                by_name[p.name] = json.loads(p.read_text())
            except json.JSONDecodeError:
                print(f"skipping unreadable {p}")
    return list(by_name.values())


def table(recs, experiment: str, metric: str, key=("arm",)) -> dict[tuple, dict[int, float]]:
    """{group -> {seed -> value}} for one metric."""
    out: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in recs:
        if r["experiment"] != experiment:
            continue
        v = r["final"].get(metric)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        out[tuple(r[k] for k in key)][r["seed"]] = float(v)
    return dict(out)


def summarise(values: dict[int, float]) -> dict:
    a = np.array(sorted(values.items()))[:, 1] if values else np.array([])
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "sem": float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired(
    a: dict[int, float], b: dict[int, float], *, n_boot: int = 20000, seed: int = 0
) -> dict:
    """Paired comparison of arm ``a`` against arm ``b`` over shared seeds.

    Returns the mean paired difference, a bootstrap 95% interval on it, the
    fraction of seeds in which ``a`` is smaller, the exact two-sided Wilcoxon
    signed-rank p-value, and a paired Cohen's d.
    """
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return {"n": len(seeds)}
    x = np.array([a[s] for s in seeds])
    y = np.array([b[s] for s in seeds])
    d = x - y

    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    ratio = float(np.mean(x) / np.mean(y)) if np.mean(y) != 0 else float("nan")
    return {
        "n": len(seeds),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "rel_change": float(d.mean() / abs(y.mean())) if y.mean() != 0 else float("nan"),
        "ratio": ratio,
        "win_rate": float((d < 0).mean()),
        "cohen_d": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf"),
        "p_wilcoxon": wilcoxon_exact(d),
        "seeds": seeds,
        "diffs": d.tolist(),
    }


def wilcoxon_exact(d: np.ndarray) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value.

    Exact rather than normal-approximated because the seed counts here are small
    enough (n <= 12) that the approximation is not appropriate, and cheap enough
    that enumerating all 2^n sign assignments is instantaneous.
    """
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    if n > 20:
        from scipy.stats import wilcoxon

        return float(wilcoxon(d).pvalue)
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    w_obs = float(ranks[d > 0].sum())
    total = ranks.sum()
    stat = min(w_obs, total - w_obs)

    counts = np.zeros(int(total) + 1)
    counts[0] = 1
    for r in ranks.astype(int):
        counts[r:] += counts[:-r] if r else 0
    dist = counts / counts.sum()
    cdf = dist[: int(stat) + 1].sum()
    return float(min(1.0, 2 * cdf))


def interaction(
    recs, experiment: str, metric: str, cell: dict[str, str], **kw
) -> dict:
    """A 2 x 2 paired interaction: (a1 - a0) - (b1 - b0), bootstrapped.

    ``cell`` maps the four labels ``a0 a1 b0 b1`` to arm names.  Used for
    "does closing the cross-stream channel cost x-pred more than it costs
    v-pred", which is a statement about the interaction and not about either
    main effect.
    """
    tab = table(recs, experiment, metric)
    got = {k: tab.get((v,), {}) for k, v in cell.items()}
    seeds = sorted(set.intersection(*[set(g) for g in got.values()]))
    if len(seeds) < 2:
        return {"n": len(seeds)}
    arr = {k: np.array([got[k][s] for s in seeds]) for k in got}
    inter = (arr["a1"] - arr["a0"]) - (arr["b1"] - arr["b0"])
    rng = np.random.default_rng(kw.get("seed", 0))
    boot = inter[rng.integers(0, len(inter), size=(20000, len(inter)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(seeds),
        "effect_a": float((arr["a1"] - arr["a0"]).mean()),
        "effect_b": float((arr["b1"] - arr["b0"]).mean()),
        "interaction": float(inter.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_wilcoxon": wilcoxon_exact(inter),
        "cells": {k: float(v.mean()) for k, v in arr.items()},
    }


def dynamics_curve(recs, experiment: str, arm: str, metric: str):
    """Mean +- sem of a metric over training steps, across seeds."""
    per_step: dict[int, list[float]] = defaultdict(list)
    for r in recs:
        if r["experiment"] != experiment or r["arm"] != arm:
            continue
        for d in r["dynamics"]:
            v = d.get(metric)
            if v is not None and math.isfinite(v):
                per_step[int(d["step"])].append(float(v))
    steps = sorted(per_step)
    mean = np.array([np.mean(per_step[s]) for s in steps])
    sem = np.array(
        [
            np.std(per_step[s], ddof=1) / math.sqrt(len(per_step[s]))
            if len(per_step[s]) > 1
            else 0.0
            for s in steps
        ]
    )
    return np.array(steps), mean, sem


def t_curve(recs, experiment: str, arm: str, stem: str, t_grid):
    """Mean +- sem of a per-timestep metric named ``{stem}_t{100t}``."""
    xs, mean, sem = [], [], []
    for t in t_grid:
        key = f"{stem}_t{int(round(t * 100))}"
        vals = [
            r["final"][key]
            for r in recs
            if r["experiment"] == experiment
            and r["arm"] == arm
            and key in r["final"]
            and math.isfinite(r["final"][key])
        ]
        if not vals:
            continue
        xs.append(t)
        mean.append(float(np.mean(vals)))
        sem.append(
            float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        )
    return np.array(xs), np.array(mean), np.array(sem)
