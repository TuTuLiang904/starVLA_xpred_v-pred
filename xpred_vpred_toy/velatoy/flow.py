"""Rectified flow, three output parameterizations, one loss function.

Path convention, matching ``vela/models/flow.py``::

    z_t = (1 - t) * eps + t * x        t = 0 is noise, t = 1 is the clean chunk
    v   = x - eps

Every arm of the study is written as a map from the network's raw output to an
**endpoint estimate** ``x_hat``, with ``sig(t) = max(1 - t, sigma_min)``::

    x-pred    x_hat = net
    v-pred    x_hat = z_t + sig * net
    eps-pred  x_hat = (z_t - sig * net) / max(t, sigma_min)

and a single loss::

    L = w(t) * || x_hat - x ||^2 ,    w(t) = 1 / sig(t)^2   ("v"-space)
                                     w(t) = 1               ("x"-space)

Two facts follow, and they are the reason the comparison is clean.

*The v-space loss under the v-pred map is exactly velocity regression.*  Since
``(x - z_t) / (1 - t) = x - eps = v``, we get ``L = ||net - v||^2``.  So
"x-pred with a velocity-weighted sample-space loss" and "textbook flow matching"
are **the same objective function**; they differ only in what the head emits.

*The v-pred and eps-pred maps are skip connections from the noisy input to the
prediction.*  ``x_hat = z_t + sig * net`` routes ``eps`` straight to the output
with gain ``1 - t``.  Noise pass-through is their default behaviour and
noise-invariance has to be purchased by emitting an exact ``-eps/sig``.  Under
x-pred there is no such path: noise-invariance is the default and the head only
has to locate the data manifold.  Everything measured in this study is a
consequence of that asymmetry.
"""

from __future__ import annotations

import torch

from .config import FlowConfig


class RectifiedFlow:
    def __init__(self, cfg: FlowConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ basics

    def sample_time(self, batch: int, device, generator=None) -> torch.Tensor:
        a, b = self.cfg.time_beta
        # Beta(a, b) via two Gammas, so a torch.Generator can be honoured.
        u = torch._standard_gamma(
            torch.full((batch,), float(a), device=device)
        )
        v = torch._standard_gamma(
            torch.full((batch,), float(b), device=device)
        )
        return (u / (u + v).clamp_min(1e-12)).clamp(max=1.0 - 1e-3)

    def sig(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t).clamp_min(self.cfg.sigma_min)

    def interpolate(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        tt = t.view(-1, *([1] * (x.dim() - 1)))
        return (1.0 - tt) * eps + tt * x

    def loss_weight(self, t: torch.Tensor, loss_space: str) -> torch.Tensor:
        if loss_space != "v":
            return torch.ones_like(t)
        w = 1.0 / self.sig(t) ** 2
        if self.cfg.normalize_weight:
            w = w / w.mean().clamp_min(1e-6)
        return w

    # -------------------------------------------------------- parameterization

    def to_endpoint(
        self,
        net_out: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        parameterization: str,
    ) -> torch.Tensor:
        """Network output -> endpoint estimate ``x_hat``."""
        shape = (-1, *([1] * (z_t.dim() - 1)))
        tt = t.view(*shape)
        sg = self.sig(t).view(*shape)
        if parameterization == "x":
            return net_out
        if parameterization == "v":
            return z_t + sg * net_out
        if parameterization == "eps":
            return (z_t - sg * net_out) / tt.clamp_min(self.cfg.sigma_min)
        raise ValueError(f"unknown parameterization {parameterization!r}")

    def to_velocity(self, x_hat: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor):
        shape = (-1, *([1] * (z_t.dim() - 1)))
        return (x_hat - z_t) / self.sig(t).view(*shape)

    # ------------------------------------------------------------- integration

    @torch.no_grad()
    def sample(
        self,
        endpoint_fn,
        shapes: tuple[tuple[int, ...], tuple[int, ...]],
        device,
        *,
        nfe: int | None = None,
        generator=None,
        record: bool = False,
    ):
        """Euler integration from noise to a clean chunk, both streams at once.

        ``endpoint_fn(z_manip, z_aux, t) -> (x_hat_manip, x_hat_aux)``.
        """
        steps = nfe or self.cfg.num_inference_steps
        s_m, s_a = shapes
        z_m = torch.randn(s_m, device=device, generator=generator)
        z_a = torch.randn(s_a, device=device, generator=generator)
        dt = 1.0 / steps
        trace = []
        for i in range(steps):
            t = torch.full((s_m[0],), i * dt, device=device)
            xm, xa = endpoint_fn(z_m, z_a, t)
            if record:
                trace.append((float(i * dt), xm.clone(), xa.clone()))
            z_m = z_m + dt * self.to_velocity(xm, z_m, t)
            z_a = z_a + dt * self.to_velocity(xa, z_a, t)
        return (z_m, z_a, trace) if record else (z_m, z_a)


def masked_endpoint_loss(
    x_hat_manip: torch.Tensor,
    x_hat_aux: torch.Tensor,
    x_manip: torch.Tensor,
    x_aux: torch.Tensor,
    mask_manip: torch.Tensor,
    mask_aux: torch.Tensor,
    aux_active: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-head masked squared error, averaged over the heads that are active.

    Averaging over active heads rather than over all supervised scalars keeps the
    58-wide manip head from swamping the 22-wide aux head, matching
    ``vela/models/heads.py``.
    """
    w = weight.view(-1, 1, 1)
    per_head = {}
    total = 0.0
    n_active = 0.0
    for name, (xh, x, m, act) in {
        "manip": (x_hat_manip, x_manip, mask_manip, torch.ones_like(aux_active)),
        "aux": (x_hat_aux, x_aux, mask_aux, aux_active),
    }.items():
        err = (xh - x).pow(2) * m
        num = (err * w).flatten(1).sum(1)
        den = m.flatten(1).sum(1).clamp_min(1.0)
        per_sample = num / den
        live = act * (m.flatten(1).sum(1) > 0).float()
        head_loss = (per_sample * live).sum() / live.sum().clamp_min(1.0)
        per_head[name] = head_loss
        total = total + head_loss
        n_active += 1.0
    return total / n_active, per_head
