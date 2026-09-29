"""An analytically solvable cross-embodiment action-chunk manifold.

The construction is deliberately simple enough that the Bayes-optimal denoiser
``E[x | z_t, c, e]`` is available in closed form, and rich enough to contain the
three structures that the study is about:

1. **A low-dimensional manifold in a high-dimensional ambient space.**  A chunk
   is ``T x 80`` numbers, but every clean chunk lies in a per-embodiment linear
   subspace of dimension ``task_dim + style_dim`` per stream.  The subspace is
   smooth in time and supported only on the channels that the embodiment
   actually labels, so the orthogonal projector onto it is exact and the
   off-manifold residual of a generated chunk is measurable without any fitting.

2. **A coordination constraint that couples the two heads.**  A single task
   latent ``tau`` is split between the manipulation stream and the whole-body
   stream by an allocation ``alpha``: the manip stream realises ``alpha * tau``
   and the aux stream realises ``(1 - alpha) * tau``.  The two shares must add
   up to ``tau``, which is what "the base moves in so the arm does not have to
   reach as far" looks like after stripping the kinematics.

3. **Genuine multimodality at fixed condition.**  ``alpha`` is drawn from
   ``{alpha_low, alpha_high}`` independently of the condition, so the same
   observation admits an arm-dominant and a base-dominant solution.  Both are
   correct; the average of the two is not.  A model must commit to one mode, and
   *both heads must commit to the same one*.  The two values are close together
   (0.35 / 0.65 by default) so that resolving the mode is a real inference
   problem rather than a difference visible at a glance.

The third point is what makes the Bayes-optimal denoiser interesting: because
the mode is shared, the optimal manip prediction depends on the *aux* stream's
noisy observation.  Cross-stream communication is not a nice-to-have in this
toy, it is part of the optimal solution, and the gap between the full Bayes risk
and the stream-local Bayes risk gives an absolute scale against which a model's
actual cross-stream information use can be reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import AUX_SLICE, CANONICAL_DIM, MANIP_SLICE, DataConfig


@dataclass
class Batch:
    """One batch of the toy.  Flattened chunks, one row per sample."""

    cond: torch.Tensor          # (B, cond_dim)     condition / VLM readout
    embod: torch.Tensor         # (B,)              embodiment index
    tau: torch.Tensor           # (B, task_dim)     shared task latent
    alpha: torch.Tensor         # (B,)              coordination allocation
    mode: torch.Tensor          # (B,) long         0 = base-dominant, 1 = arm
    style_manip: torch.Tensor   # (B, style_dim)
    style_aux: torch.Tensor     # (B, style_dim)
    x_manip: torch.Tensor       # (B, T, 58)
    x_aux: torch.Tensor         # (B, T, 22)
    mask_manip: torch.Tensor    # (B, T, 58)  1 where supervised
    mask_aux: torch.Tensor      # (B, T, 22)
    aux_active: torch.Tensor    # (B,) float  1 if the aux stream exists at all

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})


def _smooth_temporal_basis(chunk: int, rank: int, device=None) -> torch.Tensor:
    """A smooth, orthonormal temporal basis: the low-frequency cosine block."""
    t = torch.linspace(0.0, 1.0, chunk, device=device)
    cols = [torch.ones_like(t)]
    for k in range(1, rank):
        cols.append(torch.cos(math.pi * k * t))
    phi = torch.stack(cols, dim=1)
    q, _ = torch.linalg.qr(phi)
    return q[:, :rank]


class VelaToyManifold:
    """Fixed generative model.  Constructed once, shared by every arm."""

    def __init__(self, cfg: DataConfig, seed: int = 1234, device="cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(seed)

        self.manip_dim = MANIP_SLICE[1] - MANIP_SLICE[0]     # 58
        self.aux_dim = AUX_SLICE[1] - AUX_SLICE[0]           # 22
        # latent = [ alpha * tau | mode witness | private style ].  The first two
        # blocks are deterministic given (condition, mode); the last is free.
        self.det_dim = cfg.task_dim + 1
        self.latent_dim = self.det_dim + cfg.style_dim
        self.n_manip = cfg.chunk * self.manip_dim
        self.n_aux = cfg.chunk * self.aux_dim
        self.witness = {"manip": cfg.witness_manip, "aux": cfg.witness_aux}

        # ---- condition -> task latent.  Nonlinear, deterministic, unit-scaled.
        self.W_tau = torch.randn(cfg.cond_dim, cfg.task_dim, generator=g) / math.sqrt(
            cfg.cond_dim
        )

        phi = _smooth_temporal_basis(cfg.chunk, cfg.temporal_rank)

        # ---- per-embodiment validity.  Declared, never inferred from values.
        #      Channel groups follow canonical_80: arm joints, eef pos, eef rot,
        #      gripper, hand for each side; base / torso / head / discrete /
        #      reserved on the aux side.
        manip_valid = torch.ones(cfg.n_embodiments, self.manip_dim)
        aux_valid = torch.ones(cfg.n_embodiments, self.aux_dim)
        for e in range(cfg.n_embodiments):
            # reserved aux channels are never supervised, exactly as in VELA-0
            aux_valid[e, 15:] = 0.0
            if e % 2 == 0:                      # joint-controlled: drop eef block
                manip_valid[e, 7:16] = 0.0
                manip_valid[e, 36:45] = 0.0
            else:                               # eef-controlled: drop arm joints
                manip_valid[e, 0:7] = 0.0
                manip_valid[e, 29:36] = 0.0
            if e % 3 != 0:                      # no dexterous hand
                manip_valid[e, 17:29] = 0.0
                manip_valid[e, 46:58] = 0.0
        # the last `n_aux_inactive` embodiments have no whole-body labels at all
        self.aux_inactive = torch.zeros(cfg.n_embodiments, dtype=torch.bool)
        if cfg.n_aux_inactive > 0:
            self.aux_inactive[-cfg.n_aux_inactive :] = True
            aux_valid[self.aux_inactive] = 0.0

        self.manip_valid = manip_valid
        self.aux_valid = aux_valid

        # ---- per-embodiment action-manifold bases.
        # Each latent direction is a rank-one (smooth-in-time) x (channel
        # pattern) field, zeroed on unlabelled channels.  QR then gives an
        # orthonormal basis of the same subspace, still smooth and still
        # supported on the labelled channels only.
        self.Q_manip = self._make_basis(phi, manip_valid, self.manip_dim, g)
        self.Q_aux = self._make_basis(phi, aux_valid, self.aux_dim, g)

        # ---- scale so the clean chunk has unit per-channel variance on the
        #      labelled channels, matching the unit-variance flow noise.  Without
        #      this the comparison silently favours whichever target is smaller.
        self.s_manip = self._unit_variance_scale(self.Q_manip, manip_valid, cfg)
        self.s_aux = self._unit_variance_scale(self.Q_aux, aux_valid, cfg)

        self.to(device)

    # ------------------------------------------------------------------ setup

    def _make_basis(
        self,
        phi: torch.Tensor,
        valid: torch.Tensor,
        width: int,
        g: torch.Generator,
    ) -> torch.Tensor:
        cfg = self.cfg
        out = []
        for e in range(cfg.n_embodiments):
            cols = []
            for _ in range(self.latent_dim):
                coeff = torch.randn(cfg.temporal_rank, generator=g)
                coeff = coeff / coeff.norm().clamp_min(1e-8)
                pattern = torch.randn(width, generator=g) * valid[e]
                field = (phi @ coeff)[:, None] * pattern[None, :]
                cols.append(field.reshape(-1))
            raw = torch.stack(cols, dim=1)
            if valid[e].sum() == 0:
                out.append(torch.zeros(raw.shape[0], self.latent_dim))
                continue
            q, _ = torch.linalg.qr(raw)
            out.append(q[:, : self.latent_dim].contiguous())
        return torch.stack(out)

    def _unit_variance_scale(
        self, Q: torch.Tensor, valid: torch.Tensor, cfg: DataConfig
    ) -> torch.Tensor:
        """Per-embodiment scale making E[x_i^2] = 1 on labelled channels."""
        if not cfg.variance_normalized:
            return torch.ones(cfg.n_embodiments)
        # E||g||^2 with g = [alpha tau; witness; sigma]; ||tau||^2 == task_dim.
        mean_a2 = 0.5 * (cfg.alpha_low**2 + cfg.alpha_high**2)
        mean_w2 = 0.5 * (cfg.witness_manip**2 + cfg.witness_aux**2)
        e_g2 = mean_a2 * cfg.task_dim + mean_w2 + cfg.style_dim
        scales = []
        for e in range(cfg.n_embodiments):
            n_valid = float(valid[e].sum()) * cfg.chunk
            if n_valid == 0 or e_g2 == 0:
                scales.append(torch.tensor(1.0))
                continue
            # ||x||^2 = s^2 ||g||^2 spread over n_valid live entries.
            scales.append(torch.sqrt(torch.tensor(n_valid / e_g2)))
        return torch.stack(scales)

    def to(self, device) -> "VelaToyManifold":
        self.device = torch.device(device)
        for name in (
            "W_tau",
            "manip_valid",
            "aux_valid",
            "Q_manip",
            "Q_aux",
            "s_manip",
            "s_aux",
            "aux_inactive",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    # ----------------------------------------------------------------- sample

    def task_latent(self, cond: torch.Tensor) -> torch.Tensor:
        """Nonlinear condition -> task latent, rescaled to a fixed norm.

        Fixing ||tau|| = sqrt(task_dim) removes a nuisance scale from every
        downstream allocation estimate without removing the nonlinearity.
        """
        u = torch.tanh(cond @ self.W_tau)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return u * math.sqrt(self.cfg.task_dim)

    def det_latent(
        self, stream: str, mode: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """The part of a stream's latent that is fixed by (condition, mode).

        ``[ share * tau , witness * (+-1) ]``, where the manip share is ``alpha``
        and the aux share is ``1 - alpha``.  The witness amplitude differs by
        stream, which is what makes the whole-body head the better-informed one
        about the coordination mode.
        """
        cfg = self.cfg
        lo, hi = cfg.alpha_low, cfg.alpha_high
        alpha = torch.where(
            mode.bool(), torch.full_like(tau[:, 0], hi), torch.full_like(tau[:, 0], lo)
        )
        share = alpha if stream == "manip" else 1.0 - alpha
        sign = mode.float() * 2.0 - 1.0
        return torch.cat(
            [share[:, None] * tau, (self.witness[stream] * sign)[:, None]], dim=1
        )

    def sample(self, batch: int, generator: torch.Generator | None = None) -> Batch:
        cfg = self.cfg
        dev = self.device

        def randn(*shape):
            return torch.randn(*shape, device=dev, generator=generator)

        cond = randn(batch, cfg.cond_dim)
        embod = torch.randint(
            0, cfg.n_embodiments, (batch,), device=dev, generator=generator
        )
        tau = self.task_latent(cond)

        if cfg.mode_stochastic:
            mode = torch.randint(0, 2, (batch,), device=dev, generator=generator)
        else:
            score = cond[:, 0] + 0.6 * cond[:, 1] * cond[:, 2] - 0.25 * cond[:, 3]
            mode = (score > 0).long()
        alpha = torch.where(
            mode.bool(),
            torch.full_like(tau[:, 0], cfg.alpha_high),
            torch.full_like(tau[:, 0], cfg.alpha_low),
        )

        style_m = randn(batch, cfg.style_dim)
        style_a = randn(batch, cfg.style_dim)

        g_m = torch.cat([self.det_latent("manip", mode, tau), style_m], dim=1)
        g_a = torch.cat([self.det_latent("aux", mode, tau), style_a], dim=1)

        x_m = self.decode_manip(embod, g_m)
        x_a = self.decode_aux(embod, g_a)

        mask_m = self.manip_valid[embod][:, None, :].expand(-1, cfg.chunk, -1)
        mask_a = self.aux_valid[embod][:, None, :].expand(-1, cfg.chunk, -1)
        aux_active = (~self.aux_inactive[embod]).float()

        return Batch(
            cond=cond,
            embod=embod,
            tau=tau,
            alpha=alpha,
            mode=mode,
            style_manip=style_m,
            style_aux=style_a,
            x_manip=x_m,
            x_aux=x_a,
            mask_manip=mask_m.contiguous(),
            mask_aux=mask_a.contiguous(),
            aux_active=aux_active,
        )

    # ------------------------------------------------------- encode / decode

    def decode_manip(self, embod: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        Q = self.Q_manip[embod]
        s = self.s_manip[embod][:, None]
        flat = s * torch.bmm(Q, g[:, :, None]).squeeze(-1)
        return flat.view(-1, self.cfg.chunk, self.manip_dim)

    def decode_aux(self, embod: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        Q = self.Q_aux[embod]
        s = self.s_aux[embod][:, None]
        flat = s * torch.bmm(Q, g[:, :, None]).squeeze(-1)
        return flat.view(-1, self.cfg.chunk, self.aux_dim)

    def encode_manip(self, embod: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Least-squares latent of a (possibly off-manifold) manip chunk."""
        Q = self.Q_manip[embod]
        s = self.s_manip[embod][:, None]
        flat = x.reshape(x.shape[0], -1)
        return torch.bmm(Q.transpose(1, 2), flat[:, :, None]).squeeze(-1) / s

    def encode_aux(self, embod: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Q = self.Q_aux[embod]
        s = self.s_aux[embod][:, None]
        flat = x.reshape(x.shape[0], -1)
        return torch.bmm(Q.transpose(1, 2), flat[:, :, None]).squeeze(-1) / s

    def project_manip(self, embod: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.decode_manip(embod, self.encode_manip(embod, x))

    def project_aux(self, embod: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.decode_aux(embod, self.encode_aux(embod, x))

    # ----------------------------------------------------- Bayes-optimal head

    def bayes_denoise(
        self,
        z_manip: torch.Tensor,
        z_aux: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        embod: torch.Tensor,
        *,
        use_partner: bool = True,
        aux_active: torch.Tensor | None = None,
        force_mode: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-form ``E[x | z_t, c, e]`` for both streams.

        Because ``Q`` is orthonormal, everything the noisy chunk says about the
        clean chunk is contained in ``y = Q^T z_t = t s g + (1 - t) n`` with
        ``n ~ N(0, I)``.  The latent splits into a mode-dependent task block
        ``alpha tau`` and an independent Gaussian style block, so the posterior
        factorises: a two-point posterior over the mode, and a scalar Wiener
        shrinkage on the style.

        The mode is *shared* by the two streams, so the mode posterior uses the
        evidence of both.  ``use_partner=False`` restricts each stream to its own
        evidence and yields the stream-local optimum, i.e. the best a head can do
        with no cross-stream communication.  The gap between the two is the value
        of the dual-head channel, in the units of the metric being reported.

        ``force_mode`` replaces the inferred posterior with a one-hot at the given
        mode, giving the *mode-oracle* optimum: the best achievable denoising when
        the coordination mode is handed over rather than inferred.  This is the
        correct anchor for a model that takes the mode as an explicit discrete
        latent, since such a model is not solving the inference problem at all.

        Returns ``(x_manip_hat, x_aux_hat, p_mode1)``.
        """
        cfg = self.cfg
        tau = self.task_latent(cond)
        B = cond.shape[0]
        if aux_active is None:
            aux_active = (~self.aux_inactive[embod]).float()

        t = t.view(B)
        tt = t[:, None]
        sig = (1.0 - t).clamp_min(1e-6)[:, None]

        y_m = torch.bmm(
            self.Q_manip[embod].transpose(1, 2),
            z_manip.reshape(B, -1)[:, :, None],
        ).squeeze(-1)
        y_a = torch.bmm(
            self.Q_aux[embod].transpose(1, 2),
            z_aux.reshape(B, -1)[:, :, None],
        ).squeeze(-1)

        s_m = self.s_manip[embod][:, None]
        s_a = self.s_aux[embod][:, None]
        k = self.det_dim

        # --- two-point posterior over the shared coordination mode.
        # log p(m | y) = -||y_det - t s d_m||^2 / 2(1-t)^2 + const, summed over
        # whichever streams are allowed to contribute evidence.
        zeros = torch.zeros_like(embod)
        dets = {}
        for m_val in (0, 1):
            m = torch.full_like(zeros, m_val)
            dets[m_val] = (
                self.det_latent("manip", m, tau),
                self.det_latent("aux", m, tau),
            )

        logits = []
        for m_val in (0, 1):
            d_m, d_a = dets[m_val]
            ll = -(y_m[:, :k] - tt * s_m * d_m).pow(2).sum(-1) / (2.0 * sig[:, 0] ** 2)
            if use_partner:
                ll = ll - aux_active * (y_a[:, :k] - tt * s_a * d_a).pow(2).sum(-1) / (
                    2.0 * sig[:, 0] ** 2
                )
            logits.append(ll)
        p = torch.softmax(torch.stack(logits, dim=1), dim=1)         # (B, 2)
        if force_mode is not None:
            p = torch.nn.functional.one_hot(force_mode.view(B).long(), 2).to(p.dtype)

        det_m = p[:, :1] * dets[0][0] + p[:, 1:] * dets[1][0]
        det_a = p[:, :1] * dets[0][1] + p[:, 1:] * dets[1][1]

        # --- style block: scalar Wiener shrinkage, independent per stream
        def style_mean(y, s):
            return (tt * s) * y[:, k:] / ((tt * s) ** 2 + sig**2)

        g_m = torch.cat([det_m, style_mean(y_m, s_m)], dim=1)
        g_a = torch.cat([det_a, style_mean(y_a, s_a)], dim=1)
        return self.decode_manip(embod, g_m), self.decode_aux(embod, g_a), p[:, 1]

    def mode_posterior_accuracy(
        self, z_manip, z_aux, t, cond, embod, mode, *, use_partner: bool = True,
        aux_active=None,
    ) -> torch.Tensor:
        """Per-sample correctness of the closed-form mode decision."""
        _, _, p1 = self.bayes_denoise(
            z_manip, z_aux, t, cond, embod,
            use_partner=use_partner, aux_active=aux_active,
        )
        return ((p1 > 0.5).long() == mode).float()

    # ------------------------------------------------------------ diagnostics

    def allocation_readout(
        self, embod: torch.Tensor, x_manip: torch.Tensor, x_aux: torch.Tensor,
        cond: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Recover each stream's claimed share of the task latent.

        ``a_manip`` should be ``alpha`` and ``a_aux`` should be ``1 - alpha``, so
        a coordinated pair lands on the line ``a_manip + a_aux = 1``.  The
        residual off that line is the coordination violation: both heads believe
        something self-consistent, but together they do not realise the task.
        """
        tau = self.task_latent(cond)
        k = self.cfg.task_dim
        tau_sq = tau.pow(2).sum(-1).clamp_min(1e-6)
        g_m = self.encode_manip(embod, x_manip)
        g_a = self.encode_aux(embod, x_aux)
        u_m, u_a = g_m[:, :k], g_a[:, :k]
        a_m = (u_m * tau).sum(-1) / tau_sq
        a_a = (u_a * tau).sum(-1) / tau_sq
        w_m = g_m[:, k] / max(self.cfg.witness_manip, 1e-6)
        w_a = g_a[:, k] / max(self.cfg.witness_aux, 1e-6)
        # component of the realised task latent that is not parallel to tau
        perp_m = (u_m - a_m[:, None] * tau).norm(dim=-1) / tau.norm(dim=-1)
        perp_a = (u_a - a_a[:, None] * tau).norm(dim=-1) / tau.norm(dim=-1)
        return {
            "a_manip": a_m,
            "a_aux": a_a,
            "witness_manip": w_m,
            "witness_aux": w_a,
            "coord_violation": (a_m + a_a - 1.0).abs(),
            "witness_mismatch": (w_m.sign() != w_a.sign()).float(),
            "task_perp_manip": perp_m,
            "task_perp_aux": perp_a,
        }
