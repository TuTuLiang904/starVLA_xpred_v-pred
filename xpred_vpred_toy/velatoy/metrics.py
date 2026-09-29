"""Metrics on generated chunks.

The condition does not determine the coordination mode, so a plain RMSE against
the recorded chunk is the wrong summary: the average of the two legal modes beats
either of them on that metric while being physically wrong.  Everything here is
therefore either mode-agnostic or explicitly mode-aware.

* ``manifold_residual`` -- the component of the generated chunk orthogonal to the
  embodiment's action subspace.  Exact, because the subspace is known.
* ``task_residual`` -- distance from the realised task allocation to the nearer
  of the two legal modes.  Zero for either mode, large for their average.
* ``coord_violation`` -- ``|a_manip + a_aux - 1|``.  Both heads can be internally
  plausible and still fail to realise the task between them.
* ``mode_mismatch`` -- the two heads committed to different coordination modes.
* ``mode_balance`` / ``mode_collapse`` -- whether both modes survive at all.
* ``sliced_wasserstein`` -- distribution distance in the latent coordinates,
  which is the only place where a fair like-for-like comparison of the generated
  and true action distributions can be made.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, n_proj: int = 512) -> float:
    d = a.shape[1]
    dirs = torch.randn(d, n_proj, device=a.device)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)
    pa = (a @ dirs).sort(dim=0).values
    pb = (b @ dirs).sort(dim=0).values
    n = min(pa.shape[0], pb.shape[0])
    if pa.shape[0] != n:
        idx = torch.linspace(0, pa.shape[0] - 1, n, device=a.device).long()
        pa = pa[idx]
    if pb.shape[0] != n:
        idx = torch.linspace(0, pb.shape[0] - 1, n, device=b.device).long()
        pb = pb[idx]
    return float((pa - pb).abs().mean())


@torch.no_grad()
def generation_metrics(manifold, batch, gen_manip, gen_aux) -> dict[str, float]:
    cfg = manifold.cfg
    live = batch.aux_active > 0

    proj_m = manifold.project_manip(batch.embod, gen_manip)
    proj_a = manifold.project_aux(batch.embod, gen_aux)
    valid_m = batch.mask_manip
    valid_a = batch.mask_aux

    def rel_resid(x, p, m):
        num = ((x - p).pow(2) * m).flatten(1).sum(1)
        den = (x.pow(2) * m).flatten(1).sum(1).clamp_min(1e-8)
        return (num / den).sqrt()

    off_m = rel_resid(gen_manip, proj_m, valid_m)
    off_a = rel_resid(gen_aux, proj_a, valid_a)

    # leakage onto channels the embodiment never labels
    leak = (gen_manip.pow(2) * (1 - valid_m)).flatten(1).sum(1) / (
        gen_manip.pow(2).flatten(1).sum(1).clamp_min(1e-8)
    )

    read = manifold.allocation_readout(batch.embod, gen_manip, gen_aux, batch.cond)
    a_m, a_a = read["a_manip"], read["a_aux"]
    lo, hi = cfg.alpha_low, cfg.alpha_high

    task_res_m = torch.minimum((a_m - lo).abs(), (a_m - hi).abs())
    task_res_a = torch.minimum((a_a - (1 - lo)).abs(), (a_a - (1 - hi)).abs())

    mode_m = (a_m > 0.5)
    mode_a = (a_a < 0.5)          # aux realises 1 - alpha
    mismatch = (mode_m != mode_a).float()

    mid = 0.5 * (hi - lo)         # a value is "collapsed" if it sits mid-way
    collapse_m = ((a_m - 0.5).abs() < 0.25 * (hi - lo)).float()
    balance = mode_m.float().mean()

    # latent-space distribution distance against a fresh true batch
    g_gen = torch.cat(
        [
            manifold.encode_manip(batch.embod, gen_manip),
            manifold.encode_aux(batch.embod, gen_aux),
        ],
        dim=1,
    )
    g_true = torch.cat(
        [
            manifold.encode_manip(batch.embod, batch.x_manip),
            manifold.encode_aux(batch.embod, batch.x_aux),
        ],
        dim=1,
    )

    def mean_live(x):
        return float((x * live.float()).sum() / live.float().sum().clamp_min(1.0))

    return {
        "off_manifold_manip": float(off_m.mean()),
        "off_manifold_aux": mean_live(off_a),
        "off_manifold": 0.5 * (float(off_m.mean()) + mean_live(off_a)),
        "invalid_leakage": float(leak.mean()),
        "task_residual_manip": float(task_res_m.mean()),
        "task_residual_aux": mean_live(task_res_a),
        "task_residual": 0.5 * (float(task_res_m.mean()) + mean_live(task_res_a)),
        "coord_violation": mean_live(read["coord_violation"]),
        "mode_mismatch": mean_live(mismatch),
        "mode_collapse": float(collapse_m.mean()),
        "mode_balance": float(balance),
        "mode_balance_gap": float((balance - 0.5).abs()),
        "task_perp": 0.5
        * (float(read["task_perp_manip"].mean()) + mean_live(read["task_perp_aux"])),
        "sliced_wasserstein": sliced_wasserstein(g_gen, g_true),
        "chunk_rms": float(gen_manip.pow(2).mean().sqrt()),
    }


@torch.no_grad()
def trajectory_consistency(trace) -> dict[str, float]:
    """How much the endpoint estimate moves along the sampling trajectory.

    A parameterization whose endpoint estimate is already stable at t = 0 needs
    fewer solver steps; drift is the mechanism behind the NFE curve.
    """
    if len(trace) < 2:
        return {"endpoint_drift": float("nan"), "endpoint_drift_first": float("nan")}
    drifts = []
    for i in range(len(trace) - 1):
        dm = (trace[i + 1][1] - trace[i][1]).pow(2).mean()
        da = (trace[i + 1][2] - trace[i][2]).pow(2).mean()
        drifts.append(float(0.5 * (dm + da)) ** 0.5)
    final = trace[-1]
    first = trace[0]
    d0 = float(
        0.5 * ((final[1] - first[1]).pow(2).mean() + (final[2] - first[2]).pow(2).mean())
    ) ** 0.5
    return {
        "endpoint_drift": sum(drifts) / len(drifts),
        "endpoint_drift_first": d0,
    }
