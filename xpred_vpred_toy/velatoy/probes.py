"""Linear read-outs of the residual stream, with computed reference levels.

Three families of measurement, one per claim under test.

**The capacity tax (claim 1).**  ``ridge_r2`` fits a ridge regression from a
stream's per-token hidden state to that token's own flow noise ``eps``.  A
velocity or noise head has to emit ``-eps/sig`` to cancel the skip connection, so
its residual stream is obliged to carry the noise; a clean-sample head is not.
The same routine, pointed at the task latent instead, says how much of the stream
is spent on the part of the problem that actually matters.  Since both targets
are read out of the *same* fixed-width stream, the two numbers together are a
budget: ``participation_ratio`` reports how many directions that budget spans.

**The consistency default (claim 2).**  ``noise_invariance`` holds a clean chunk
fixed, draws many independent noises, and measures how far the endpoint estimate
moves.  This has a nonzero optimum -- the Bayes-optimal denoiser also moves when
its input moves -- so the closed-form denoiser is evaluated on exactly the same
draws and reported alongside as the floor.

**The dual-head channel (claim 3).**  The toy's mode is shared by the two heads,
so the Bayes-optimal manip prediction depends on the aux stream's observation.
``cross_stream_utilization`` places a method between two computed anchors: the
stream-local Bayes risk, which is the best a head can do with the cross-stream
channel switched off, and the full Bayes risk, which uses it optimally.  The
resulting number is a fraction, not an arbitrary scale.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------- ridge

def ridge_r2(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 1e-3,
    train_frac: float = 0.7,
) -> float:
    """Held-out R^2 of a ridge probe, pooled over all target dimensions.

    Features are standardised and a bias column is appended, so the reported
    number is invariant to the scale of the residual stream.
    """
    x = features.float()
    y = targets.float()
    n = x.shape[0]
    n_tr = int(train_frac * n)
    if n_tr < x.shape[1] // 2 or n - n_tr < 8:
        return float("nan")

    mu, sd = x[:n_tr].mean(0, keepdim=True), x[:n_tr].std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mu) / sd
    x = torch.cat([x, torch.ones(n, 1, device=x.device)], dim=1)
    ymu = y[:n_tr].mean(0, keepdim=True)
    y = y - ymu

    xt, yt = x[:n_tr], y[:n_tr]
    a = xt.T @ xt
    a = a + ridge * torch.trace(a) / a.shape[0] * torch.eye(a.shape[0], device=a.device)
    w = torch.linalg.solve(a, xt.T @ yt)

    pred = x[n_tr:] @ w
    resid = (y[n_tr:] - pred).pow(2).sum()
    total = y[n_tr:].pow(2).sum().clamp_min(1e-12)
    return float(1.0 - resid / total)


def logit_accuracy(
    features: torch.Tensor, labels: torch.Tensor, *, ridge: float = 1e-3,
    train_frac: float = 0.7,
) -> float:
    """Held-out accuracy of a linear (least-squares) classifier on +-1 labels."""
    y = (labels.float() * 2.0 - 1.0)[:, None]
    x = features.float()
    n = x.shape[0]
    n_tr = int(train_frac * n)
    mu, sd = x[:n_tr].mean(0, keepdim=True), x[:n_tr].std(0, keepdim=True).clamp_min(1e-6)
    x = torch.cat([(x - mu) / sd, torch.ones(n, 1, device=x.device)], dim=1)
    xt = x[:n_tr]
    a = xt.T @ xt
    a = a + ridge * torch.trace(a) / a.shape[0] * torch.eye(a.shape[0], device=a.device)
    w = torch.linalg.solve(a, xt.T @ y[:n_tr])
    pred = (x[n_tr:] @ w).sign()
    return float((pred == y[n_tr:]).float().mean())


def participation_ratio(features: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 -- a soft count of active directions."""
    x = features.float()
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    lam = s.pow(2)
    return float(lam.sum().pow(2) / lam.pow(2).sum().clamp_min(1e-12))


def explained_variance_share(
    features: torch.Tensor, groups: dict[str, torch.Tensor], *, ridge: float = 1e-3
) -> dict[str, float]:
    """How much of the stream's variance each target group linearly explains.

    Not a partition -- the groups are only approximately orthogonal -- but the
    shares are directly comparable across arms because they are computed on the
    same standardised stream with the same regulariser.
    """
    out = {}
    for name, tgt in groups.items():
        out[name] = max(0.0, ridge_r2(tgt, features, ridge=ridge))
    return out


# ------------------------------------------------------------- noise invariance

@torch.no_grad()
def noise_invariance(
    endpoint_fn,
    manifold,
    batch,
    t_value: float,
    n_draws: int,
    flow,
    *,
    bayes: bool = True,
) -> dict[str, float]:
    """Spread of the endpoint estimate over independent noises at fixed data.

    Returns the trace of ``Var_eps[x_hat]`` normalised by the variance of the
    clean chunk, for the model and, optionally, for the closed-form optimum
    evaluated on the very same noises.
    """
    dev = batch.x_manip.device
    B = batch.x_manip.shape[0]
    t = torch.full((B,), t_value, device=dev)

    acc = {"model": [], "bayes": []}
    for _ in range(n_draws):
        e_m = torch.randn_like(batch.x_manip)
        e_a = torch.randn_like(batch.x_aux)
        z_m = flow.interpolate(batch.x_manip, e_m, t)
        z_a = flow.interpolate(batch.x_aux, e_a, t)
        xm, xa = endpoint_fn(z_m, z_a, t)
        acc["model"].append((xm, xa))
        if bayes:
            bm, ba, _ = manifold.bayes_denoise(
                z_m, z_a, t, batch.cond, batch.embod, aux_active=batch.aux_active
            )
            acc["bayes"].append((bm, ba))

    out = {}
    denom_m = batch.x_manip.var().clamp_min(1e-8)
    denom_a = batch.x_aux.var().clamp_min(1e-8)
    for key, draws in acc.items():
        if not draws:
            continue
        sm = torch.stack([d[0] for d in draws])
        sa = torch.stack([d[1] for d in draws])
        vm = sm.var(dim=0, unbiased=True).mean() / denom_m
        va = sa.var(dim=0, unbiased=True).mean() / denom_a
        out[f"niv_manip_{key}"] = float(vm)
        out[f"niv_aux_{key}"] = float(va)
        out[f"niv_{key}"] = float(0.5 * (vm + va))
    return out


@torch.no_grad()
def noise_jacobian(
    endpoint_fn, batch, t_value: float, flow, n_probes: int = 4, delta: float = 1e-2
) -> float:
    """Hutchinson estimate of ``||d x_hat / d eps||_F / sqrt(N)`` at fixed data.

    Central differences.  A value of 0 means the endpoint ignores the noise
    entirely; the pass-through default of a velocity head is ``1 - t``.
    """
    dev = batch.x_manip.device
    B = batch.x_manip.shape[0]
    t = torch.full((B,), t_value, device=dev)
    e_m = torch.randn_like(batch.x_manip)
    e_a = torch.randn_like(batch.x_aux)
    z_m = flow.interpolate(batch.x_manip, e_m, t)
    z_a = flow.interpolate(batch.x_aux, e_a, t)
    scale = 1.0 - t_value

    acc, n = 0.0, 0
    for _ in range(n_probes):
        u_m = torch.randn_like(z_m)
        u_a = torch.randn_like(z_a)
        p = endpoint_fn(z_m + delta * scale * u_m, z_a + delta * scale * u_a, t)
        m = endpoint_fn(z_m - delta * scale * u_m, z_a - delta * scale * u_a, t)
        dm = (p[0] - m[0]) / (2 * delta)
        da = (p[1] - m[1]) / (2 * delta)
        acc += float(dm.pow(2).mean() + da.pow(2).mean()) / 2.0
        n += 1
    return (acc / max(n, 1)) ** 0.5


# ------------------------------------------------------- cross-stream anchors

@torch.no_grad()
def bayes_reference_risk(
    manifold, batch, t_value: float, flow, generator=None
) -> dict[str, float]:
    """Denoising risk of the full, stream-local and mode-oracle closed-form optima.

    All are evaluated on the same noise, so the difference between the first two
    is the value of the cross-stream channel and nothing else.

    The third is the anchor an explicitly mode-conditioned model must be read
    against.  Such a model is handed the coordination mode instead of inferring
    it, so it is not solving the same problem as the other arms and its risk is
    not comparable to ``risk_full``; ``risk_oracle`` is the corresponding optimum
    for the easier problem.  Reported for every arm so the two anchors, and the
    size of the inference problem that separates them, are visible throughout.
    """
    dev = batch.x_manip.device
    B = batch.x_manip.shape[0]
    t = torch.full((B,), t_value, device=dev)
    e_m = torch.randn(batch.x_manip.shape, device=dev, generator=generator)
    e_a = torch.randn(batch.x_aux.shape, device=dev, generator=generator)
    z_m = flow.interpolate(batch.x_manip, e_m, t)
    z_a = flow.interpolate(batch.x_aux, e_a, t)

    out = {}
    for tag, partner, forced in (
        ("full", True, None), ("local", False, None), ("oracle", True, batch.mode),
    ):
        bm, ba, p1 = manifold.bayes_denoise(
            z_m, z_a, t, batch.cond, batch.embod,
            use_partner=partner, aux_active=batch.aux_active, force_mode=forced,
        )
        out[f"risk_{tag}"] = _masked_mse(bm, ba, batch)
        out[f"det_risk_{tag}"] = det_block_risk(manifold, bm, ba, batch)
        out[f"mode_acc_{tag}"] = float(
            ((p1 > 0.5).long() == batch.mode).float().mean()
        )
    out["z_manip"] = z_m
    out["z_aux"] = z_a
    out["t"] = t
    return out


@torch.no_grad()
def partner_probe_ceiling(manifold, batch, z_m, z_a, t) -> dict[str, float]:
    """How much of the partner's clean chunk is knowable at all.

    The manip stream's private style latent is independent of the aux stream's,
    so no amount of communication makes the partner fully predictable.  The
    closed-form denoiser conditioned on *both* streams is by definition the best
    possible estimate of the partner's endpoint, so its R^2 is the ceiling that
    the measured cross-head probe should be read against.
    """
    bm, ba, _ = manifold.bayes_denoise(
        z_m, z_a, t, batch.cond, batch.embod, aux_active=batch.aux_active
    )
    out = {}
    for tag, (pred, tgt, m) in {
        "partner_manip": (ba, batch.x_aux, batch.mask_aux),
        "partner_aux": (bm, batch.x_manip, batch.mask_manip),
    }.items():
        resid = ((pred - tgt).pow(2) * m).sum()
        total = ((tgt - (tgt * m).sum() / m.sum().clamp_min(1)).pow(2) * m).sum()
        out[f"ceiling_{tag}"] = float(1.0 - resid / total.clamp_min(1e-9))
    return out


def block_risks(manifold, xm, xa, batch) -> dict[str, float]:
    """Split the risk into the three places the error can live.

    The generative model has three orthogonal pieces, and the parameterizations
    are not equally suited to them, so a single MSE hides the story:

    ``det``
        The task allocation and the mode witness: deterministic given the
        condition and the coordination mode, discrete, and the part that decides
        whether the plan is one of the demonstrated behaviours.
    ``style``
        An isotropic Gaussian latent inside the same subspace.  Its optimal
        estimate is a scalar shrinkage of the observation -- that is, a scaled
        copy of the input -- which a skip parameterization gets almost for free
        and a clean-sample head has to synthesise.
    ``offman``
        Everything outside the embodiment's subspace, where no clean chunk ever
        lies.  Nonzero here is pure hallucination.
    """
    k = manifold.det_dim
    tau = manifold.task_latent(batch.cond)
    live = batch.aux_active
    out = {}

    def reduce(err_m, err_a):
        return 0.5 * (
            float(err_m.mean())
            + float((err_a * live).sum() / live.sum().clamp_min(1.0))
        )

    g_m = manifold.encode_manip(batch.embod, xm)
    g_a = manifold.encode_aux(batch.embod, xa)
    t_m = manifold.det_latent("manip", batch.mode, tau)
    t_a = manifold.det_latent("aux", batch.mode, tau)

    out["det"] = reduce(
        (g_m[:, :k] - t_m).pow(2).mean(1), (g_a[:, :k] - t_a).pow(2).mean(1)
    )
    out["style"] = reduce(
        (g_m[:, k:] - batch.style_manip).pow(2).mean(1),
        (g_a[:, k:] - batch.style_aux).pow(2).mean(1),
    )
    off_m = ((xm - manifold.project_manip(batch.embod, xm)).pow(2) * batch.mask_manip)
    off_a = ((xa - manifold.project_aux(batch.embod, xa)).pow(2) * batch.mask_aux)
    out["offman"] = reduce(
        off_m.flatten(1).sum(1) / batch.mask_manip.flatten(1).sum(1).clamp_min(1),
        off_a.flatten(1).sum(1) / batch.mask_aux.flatten(1).sum(1).clamp_min(1),
    )
    return out


def det_block_risk(manifold, xm, xa, batch) -> float:
    return block_risks(manifold, xm, xa, batch)["det"]


def _masked_mse(xm, xa, batch, weight_aux: bool = True) -> float:
    em = ((xm - batch.x_manip).pow(2) * batch.mask_manip).flatten(1).sum(1)
    dm = batch.mask_manip.flatten(1).sum(1).clamp_min(1.0)
    ea = ((xa - batch.x_aux).pow(2) * batch.mask_aux).flatten(1).sum(1)
    da = batch.mask_aux.flatten(1).sum(1).clamp_min(1.0)
    live = batch.aux_active * (batch.mask_aux.flatten(1).sum(1) > 0).float()
    manip = float((em / dm).mean())
    aux = float(((ea / da) * live).sum() / live.sum().clamp_min(1.0))
    return 0.5 * (manip + aux) if weight_aux else manip


@torch.no_grad()
def net_jacobian_spectrum(
    net_fn, batch, t_value: float, flow, tok_dim: int, *, n_batch: int = 48,
    delta: float = 1e-2,
) -> dict:
    """Singular spectrum of the network's own input-output Jacobian.

    This is the most direct reading of the redundancy claim available, because it
    counts dimensions instead of inferring them.

    The optimal *endpoint* map is low-rank under either convention: clean chunks
    live in a low-dimensional subspace, so ``d x_hat / d z`` has the rank of that
    subspace and no more.  But the network does not emit the endpoint under a skip
    convention.  With ``x_hat = z_t + sig * net``, the network's own map has to
    satisfy ``d net/d z = (d x_hat/d z - I)/sig``: an identity, which is full rank,
    plus something low-rank.  A clean-sample head emits the endpoint directly, so
    its own Jacobian *is* the low-rank one.

    The prediction is therefore sharp and quantitative.  The number of directions
    the network's map must span is the manifold dimension for a clean-sample head
    and the full per-token action width for a skip head, whatever the loss says.
    Measured on the leading action token's diagonal block by central differences.
    """
    dev = batch.x_manip.device
    b = _slice_batch(batch, n_batch)
    t = torch.full((n_batch,), t_value, device=dev)
    e_m = torch.randn_like(b.x_manip)
    e_a = torch.randn_like(b.x_aux)
    z_m = flow.interpolate(b.x_manip, e_m, t)
    z_a = flow.interpolate(b.x_aux, e_a, t)

    group = tok_dim // b.x_manip.shape[-1]
    D = b.x_manip.shape[-1]

    cols = []
    for k in range(tok_dim):
        step, chan = divmod(k, D)
        pert = torch.zeros_like(z_m)
        pert[:, step, chan] = delta
        plus = net_fn(b, z_m + pert, z_a, t)
        minus = net_fn(b, z_m - pert, z_a, t)
        # response of the same token's own output block
        d = (plus - minus)[:, :group].reshape(n_batch, -1) / (2 * delta)
        cols.append(d)
    J = torch.stack(cols, dim=2)                      # (B, tok_dim, tok_dim)

    s = torch.linalg.svdvals(J.float())               # (B, tok_dim)
    lam = s.pow(2)
    pr = (lam.sum(1).pow(2) / lam.pow(2).sum(1).clamp_min(1e-20)).mean()
    # how many directions carry 99% of the map's energy
    frac = lam.cumsum(1) / lam.sum(1, keepdim=True).clamp_min(1e-20)
    rank99 = (frac < 0.99).sum(1).float().mean() + 1
    mean_spectrum = (s / s[:, :1].clamp_min(1e-12)).mean(0)
    return {
        "jac_participation_ratio": float(pr),
        "jac_rank99": float(rank99),
        "jac_spectrum": mean_spectrum.cpu().tolist(),
        "jac_tok_dim": tok_dim,
    }


def _slice_batch(batch, n):
    from .data import Batch

    return Batch(**{k: v[:n] for k, v in batch.__dict__.items()})


@torch.no_grad()
def mode_decision(manifold, x_hat, stream: str, batch) -> torch.Tensor:
    """Which coordination mode a single head's prediction has committed to.

    The optimal linear discriminant between the two modes in that stream's own
    latent coordinates: project the predicted deterministic block onto the
    difference of the two mode means, relative to their midpoint.  Reading only
    one stream is the point -- it asks what *this* head believes, so the answer
    can be compared against what this head could possibly know on its own.
    """
    k = manifold.det_dim
    tau = manifold.task_latent(batch.cond)
    zeros = torch.zeros_like(batch.embod)
    d0 = manifold.det_latent(stream, zeros, tau)
    d1 = manifold.det_latent(stream, torch.ones_like(zeros), tau)
    enc = manifold.encode_manip if stream == "manip" else manifold.encode_aux
    got = enc(batch.embod, x_hat)[:, :k]
    diff = d1 - d0
    score = ((got - 0.5 * (d0 + d1)) * diff).sum(-1)
    return (score > 0).long()


def cross_stream_utilization(risk_model: float, risk_local: float, risk_full: float) -> float:
    """Where a method sits between no communication and optimal communication.

    1.0 means the head extracts everything the partner stream has to offer, 0.0
    means it behaves as if the channel were closed.
    """
    span = risk_local - risk_full
    if span <= 1e-9:
        return float("nan")
    return float((risk_local - risk_model) / span)
