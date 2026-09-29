"""One run: train a single arm and instrument it while it trains.

Pairing is exact.  All arms at a given seed share the generative model, the model
initialisation, the batch sequence, the noise sequence and the timestep sequence,
because every one of those is drawn from a generator seeded only by the seed.
The arms differ in the interpretation of the output head, the loss weighting, the
stream visibility, and nothing else.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch

from .config import RunConfig
from .data import VelaToyManifold
from .flow import RectifiedFlow, masked_endpoint_loss
from .metrics import generation_metrics, trajectory_consistency
from .model import ToyJointExpert
from .probes import (
    bayes_reference_risk,
    cross_stream_utilization,
    block_risks,
    det_block_risk,
    logit_accuracy,
    mode_decision,
    net_jacobian_spectrum,
    noise_invariance,
    noise_jacobian,
    participation_ratio,
    partner_probe_ceiling,
    ridge_r2,
    _masked_mse,
)


PROBE_T = 0.2
"""Where the single-number probes are taken: high noise, mode still open."""

JAC_T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
"""Noise levels for the Jacobian-rank sweep.  Coarser than ``EvalConfig.t_grid``
because each point costs ``2 * tok_dim`` forward passes."""


@dataclass
class RunResult:
    config: dict
    dynamics: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)


def build(cfg: RunConfig):
    dev = torch.device(cfg.device)
    manifold = VelaToyManifold(cfg.data, seed=1234, device=dev)
    width = cfg.arm.width or cfg.model.width
    mcfg = copy.deepcopy(cfg.model)
    mcfg.width = width
    mcfg.joint_attention = cfg.arm.joint_attention
    mcfg.mode_conditioning = cfg.arm.mode_conditioning
    mcfg.mode_classifier = cfg.arm.mode_agree_weight > 0
    torch.manual_seed(10_000 + cfg.train.seed)
    model = ToyJointExpert(
        mcfg,
        chunk=cfg.data.chunk,
        near=cfg.data.near,
        manip_dim=manifold.manip_dim,
        aux_dim=manifold.aux_dim,
        cond_dim=cfg.data.cond_dim,
        n_embodiments=cfg.data.n_embodiments,
    ).to(dev)
    return manifold, model, RectifiedFlow(cfg.flow)


def sample_mode_prior(batch, generator=None):
    """A coordination mode drawn from its prior, for use at inference.

    The mode is independent of the condition by construction, so its prior is
    exactly uniform and no part of the recorded chunk leaks in.  Training and the
    denoising probes use the observed mode -- it is a label -- and generation
    samples it, which is the ordinary treatment of a discrete latent.
    """
    return torch.randint(
        0, 2, batch.mode.shape, device=batch.mode.device, generator=generator
    )


def endpoint_closure(model, flow, cfg, batch, *, return_hidden=False, mode=None):
    """``(z_manip, z_aux, t) -> (x_hat_manip, x_hat_aux)`` for the current arm."""
    if mode is None and cfg.arm.mode_conditioning:
        mode = batch.mode

    def fn(z_m, z_a, t):
        out = model(
            z_m, z_a, t, batch.cond, batch.embod,
            batch.mask_manip, batch.mask_aux, batch.aux_active,
            mode=mode, return_hidden=return_hidden,
        )
        p = cfg.arm.parameterization
        xm = flow.to_endpoint(out.manip, z_m, t, p)
        xa = flow.to_endpoint(out.aux, z_a, t, p)
        return (xm, xa, out) if return_hidden else (xm, xa)

    return fn


# --------------------------------------------------------------------- probing

@torch.no_grad()
def probe_suite(cfg, manifold, model, flow, batch, *, full: bool = False) -> dict:
    model.eval()
    dev = batch.x_manip.device
    ev = cfg.eval
    out: dict[str, float] = {}

    # ---- denoising risk and mode decisions against the two computed anchors.
    # t = 0.2 is where the second step of a five-step Euler schedule lands, and
    # is inside the window where the coordination mode is still undecided; the
    # full sweep covers the rest of the schedule.
    t_probe = ev.t_grid if full else (0.2,)
    for t_val in t_probe:
        ref = bayes_reference_risk(manifold, batch, t_val, flow)
        z_m, z_a, t = ref["z_manip"], ref["z_aux"], ref["t"]
        fn = endpoint_closure(model, flow, cfg, batch)
        xm, xa = fn(z_m, z_a, t)
        r_model = _masked_mse(xm, xa, batch)
        blocks = block_risks(manifold, xm, xa, batch)
        d_model = blocks["det"]
        acc_m = float((mode_decision(manifold, xm, "manip", batch) == batch.mode)
                      .float().mean())
        live = batch.aux_active > 0
        acc_a = float((mode_decision(manifold, xa, "aux", batch) == batch.mode)
                      .float()[live].mean())
        agree = float(
            (mode_decision(manifold, xm, "manip", batch)
             == mode_decision(manifold, xa, "aux", batch)).float()[live].mean()
        )
        tag = "" if not full else f"_t{int(t_val * 100)}"
        out[f"risk_model{tag}"] = r_model
        out[f"risk_bayes_full{tag}"] = ref["risk_full"]
        out[f"risk_bayes_local{tag}"] = ref["risk_local"]
        out[f"risk_bayes_oracle{tag}"] = ref["risk_oracle"]
        out[f"excess_risk{tag}"] = r_model - ref["risk_full"]
        # An arm handed the mode is not solving the inference problem, so its
        # excess over the full optimum is not the like-for-like number; this is.
        out[f"excess_risk_oracle{tag}"] = r_model - ref["risk_oracle"]
        out[f"det_risk_model{tag}"] = d_model
        for blk, val in blocks.items():
            out[f"block_{blk}{tag}"] = val
        out[f"det_risk_full{tag}"] = ref["det_risk_full"]
        out[f"det_risk_local{tag}"] = ref["det_risk_local"]
        out[f"mode_acc_model{tag}"] = acc_m
        out[f"mode_acc_model_aux{tag}"] = acc_a
        out[f"mode_agree_model{tag}"] = agree
        out[f"mode_acc_bayes_full{tag}"] = ref["mode_acc_full"]
        out[f"mode_acc_bayes_local{tag}"] = ref["mode_acc_local"]
        out[f"xstream_utilization{tag}"] = cross_stream_utilization(
            r_model, ref["risk_local"], ref["risk_full"]
        )
        out[f"xstream_utilization_det{tag}"] = cross_stream_utilization(
            d_model, ref["det_risk_local"], ref["det_risk_full"]
        )
        out[f"xstream_utilization_mode{tag}"] = cross_stream_utilization(
            1.0 - acc_m, 1.0 - ref["mode_acc_local"], 1.0 - ref["mode_acc_full"]
        )
    if full:
        for key in (
            "risk_model", "risk_bayes_full", "risk_bayes_local", "excess_risk",
            "risk_bayes_oracle", "excess_risk_oracle",
            "det_risk_model", "det_risk_full", "det_risk_local",
            "mode_acc_model", "mode_acc_model_aux", "mode_agree_model",
            "mode_acc_bayes_full", "mode_acc_bayes_local",
            "block_det", "block_style", "block_offman",
            "xstream_utilization", "xstream_utilization_det",
            "xstream_utilization_mode",
        ):
            out[key] = out[f"{key}_t20"]

    # ---- residual-stream read-outs, in the same high-noise regime
    t = torch.full((batch.x_manip.shape[0],), PROBE_T, device=dev)
    e_m = torch.randn_like(batch.x_manip)
    e_a = torch.randn_like(batch.x_aux)
    z_m = flow.interpolate(batch.x_manip, e_m, t)
    z_a = flow.interpolate(batch.x_aux, e_a, t)
    fn = endpoint_closure(model, flow, cfg, batch, return_hidden=True)
    _, _, netout = fn(z_m, z_a, t)

    layers = [len(netout.hidden["manip"]) - 1] if not full else list(
        range(len(netout.hidden["manip"]))
    )
    for li in layers:
        h_m = netout.hidden["manip"][li]           # (B, 1+T, d)
        h_a = netout.hidden["aux"][li]
        tok_m = h_m[:, 1:].reshape(-1, h_m.shape[-1])
        tok_a = h_a[:, 1:].reshape(-1, h_a.shape[-1])
        pool_m = h_m[:, 1:].mean(1)
        pool_a = h_a[:, 1:].mean(1)
        sfx = f"_L{li}"

        # fold the targets exactly the way the model folds its input, so a probe
        # on one token sees the chunk steps that token is responsible for
        def fold(x):
            return x.reshape(x.shape[0] * model.n_tokens, -1)

        n_tok = min(tok_m.shape[0], 8192)
        sel = torch.randperm(tok_m.shape[0], device=dev)[:n_tok]

        # claim 1: is the stream obliged to carry its own flow noise?
        out[f"probe_eps_manip{sfx}"] = ridge_r2(
            tok_m[sel], fold(e_m)[sel], ridge=ev.probe_ridge
        )
        out[f"probe_eps_aux{sfx}"] = ridge_r2(
            tok_a[sel], fold(e_a)[sel], ridge=ev.probe_ridge
        )
        # ... and how much of the useful latent does it hold?
        out[f"probe_tau_manip{sfx}"] = ridge_r2(pool_m, batch.tau, ridge=ev.probe_ridge)
        out[f"probe_mode_manip{sfx}"] = logit_accuracy(
            pool_m, batch.mode, ridge=ev.probe_ridge
        )
        # claim 3: does the manip stream know the partner's endpoint?
        out[f"probe_partner_manip{sfx}"] = ridge_r2(
            tok_m[sel], fold(batch.x_aux)[sel], ridge=ev.probe_ridge,
        )
        out[f"probe_partner_aux{sfx}"] = ridge_r2(
            tok_a[sel], fold(batch.x_manip)[sel], ridge=ev.probe_ridge,
        )
        # negative control for the two probes above.  A ridge read-out with this
        # many features can score above zero by fitting the target's marginal
        # rather than the pairing, which would make the cross-head numbers
        # meaningless.  Re-running with the pairing destroyed -- same features,
        # same target distribution, rows permuted -- separates the two.
        perm = sel[torch.randperm(sel.shape[0], device=dev)]
        out[f"probe_partner_manip_shuf{sfx}"] = ridge_r2(
            tok_m[sel], fold(batch.x_aux)[perm], ridge=ev.probe_ridge,
        )
        out[f"probe_eps_manip_shuf{sfx}"] = ridge_r2(
            tok_m[sel], fold(e_m)[perm], ridge=ev.probe_ridge,
        )
        out[f"pr_manip{sfx}"] = participation_ratio(pool_m)
    out.update(partner_probe_ceiling(manifold, batch, z_m, z_a, t))

    last = f"_L{len(netout.hidden['manip']) - 1}"
    for key in (
        "probe_eps_manip", "probe_tau_manip", "probe_mode_manip",
        "probe_partner_manip", "probe_partner_aux", "pr_manip",
        "probe_partner_manip_shuf", "probe_eps_manip_shuf",
    ):
        out[key] = out[f"{key}{last}"]

    # ---- claim 2: endpoint stability under a change of noise
    inv = noise_invariance(
        endpoint_closure(model, flow, cfg, batch), manifold, batch,
        PROBE_T, 8 if not full else ev.n_noise_draws, flow,
    )
    out.update(inv)
    out["noise_jacobian"] = noise_jacobian(
        endpoint_closure(model, flow, cfg, batch), batch, PROBE_T, flow
    )

    if full:
        # the network's own input-output Jacobian: a direct count of the
        # dimensions its map has to span
        def raw_net(bb, zm, za, tt):
            return model(
                zm, za, tt, bb.cond, bb.embod, bb.mask_manip, bb.mask_aux,
                bb.aux_active,
                mode=bb.mode if cfg.arm.mode_conditioning else None,
            ).manip

        tok_dim = manifold.manip_dim * model.group
        out.update(net_jacobian_spectrum(raw_net, batch, PROBE_T, flow, tok_dim=tok_dim))
        # The rank claim is a statement about the map, so it must hold across the
        # noise level rather than at one convenient t.  Swept on a smaller batch
        # because the measurement costs 2*tok_dim forward passes per t.
        for t_val in JAC_T_GRID:
            spec = net_jacobian_spectrum(
                raw_net, batch, t_val, flow, tok_dim=tok_dim, n_batch=24
            )
            out[f"jac_pr_t{int(t_val * 100)}"] = spec["jac_participation_ratio"]
            out[f"jac_rank99_t{int(t_val * 100)}"] = spec["jac_rank99"]

        for t_val in ev.t_grid:
            inv = noise_invariance(
                endpoint_closure(model, flow, cfg, batch), manifold, batch,
                t_val, ev.n_noise_draws, flow,
            )
            out[f"niv_model_t{int(t_val * 100)}"] = inv["niv_model"]
            out[f"niv_bayes_t{int(t_val * 100)}"] = inv["niv_bayes"]
            out[f"jac_t{int(t_val * 100)}"] = noise_jacobian(
                endpoint_closure(model, flow, cfg, batch), batch, t_val, flow
            )

    # ---- generation, at the deployment NFE
    gen = sample_and_score(cfg, manifold, model, flow, batch, nfe=cfg.eval.nfe_main)
    out.update({f"gen_{k}": v for k, v in gen.items()})
    model.train()
    return out


@torch.no_grad()
def sample_and_score(cfg, manifold, model, flow, batch, nfe: int, record=False):
    gen_mode = None
    if cfg.arm.mode_conditioning:
        # drawn from the prior, not read off the batch: generation must not see
        # the mode of the recorded chunk
        g = torch.Generator(device=batch.mode.device).manual_seed(4242)
        gen_mode = sample_mode_prior(batch, generator=g)
    fn = endpoint_closure(model, flow, cfg, batch, mode=gen_mode)
    res = flow.sample(
        fn,
        (tuple(batch.x_manip.shape), tuple(batch.x_aux.shape)),
        batch.x_manip.device,
        nfe=nfe,
        record=record,
    )
    if record:
        gm, ga, trace = res
    else:
        gm, ga = res
        trace = []
    m = generation_metrics(manifold, batch, gm, ga)
    if record:
        m.update(trajectory_consistency(trace))
    return (m, gm, ga) if record else m


# ----------------------------------------------------------------------- train

def run(cfg: RunConfig, verbose: bool = False) -> RunResult:
    dev = torch.device(cfg.device)
    manifold, model, flow = build(cfg)
    tr = cfg.train
    arm = cfg.arm

    opt = torch.optim.AdamW(
        model.parameters(), lr=tr.lr, weight_decay=tr.weight_decay, betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(tr.warmup, 1))
        * (0.5 * (1 + math.cos(math.pi * min(1.0, s / tr.steps)))) ** 0.5,
    )
    ema = EMA(model, tr.ema_decay)

    gen = torch.Generator(device=dev).manual_seed(50_000 + tr.seed)
    probe_gen = torch.Generator(device=dev).manual_seed(777 + tr.seed)
    probe_batch = manifold.sample(tr.probe_batch, generator=probe_gen)

    result = RunResult(config=cfg.to_dict())
    t0 = time.time()

    for step in range(tr.steps + 1):
        if step % tr.probe_every == 0:
            snap = copy.deepcopy(model)
            ema.copy_to(snap)
            m = probe_suite(cfg, manifold, snap, flow, probe_batch)
            m["step"] = step
            m["wall"] = time.time() - t0
            result.dynamics.append(m)
            del snap
            if verbose:
                print(
                    f"[{arm.name} s{tr.seed}] step {step:5d} "
                    f"excess={m['excess_risk']:.4f} "
                    f"niv={m['niv_model']:.4f}(bayes {m['niv_bayes']:.4f}) "
                    f"eps_probe={m['probe_eps_manip']:.3f} "
                    f"partner={m['probe_partner_manip']:.3f} "
                    f"coord={m['gen_coord_violation']:.3f}",
                    flush=True,
                )
        if step == tr.steps:
            break

        batch = manifold.sample(tr.batch, generator=gen)
        eps_m = torch.randn(batch.x_manip.shape, device=dev, generator=gen)
        eps_a = torch.randn(batch.x_aux.shape, device=dev, generator=gen)
        t = flow.sample_time(tr.batch, dev)
        z_m = flow.interpolate(batch.x_manip, eps_m, t)
        z_a = flow.interpolate(batch.x_aux, eps_a, t)

        gt_mode = batch.mode if arm.mode_conditioning else None
        out = model(
            z_m, z_a, t, batch.cond, batch.embod,
            batch.mask_manip, batch.mask_aux, batch.aux_active, mode=gt_mode,
        )
        xm = flow.to_endpoint(out.manip, z_m, t, arm.parameterization)
        xa = flow.to_endpoint(out.aux, z_a, t, arm.parameterization)
        w = flow.loss_weight(t, arm.loss_space)
        loss, per_head = masked_endpoint_loss(
            xm, xa, batch.x_manip, batch.x_aux,
            batch.mask_manip, batch.mask_aux, batch.aux_active, w,
        )

        if arm.mode_agree_weight > 0 and out.mode_logits is not None:
            # Both streams predict the same shared label, so the penalty pushes
            # them to represent the coordination mode and to agree about it.  It
            # gives the sampler nothing to commit to, which is the point of
            # separating it from mode conditioning.
            ce = torch.nn.functional.cross_entropy(
                out.mode_logits["manip"], batch.mode
            )
            per = torch.nn.functional.cross_entropy(
                out.mode_logits["aux"], batch.mode, reduction="none"
            )
            live = batch.aux_active
            ce = ce + (per * live).sum() / live.sum().clamp_min(1.0)
            loss = loss + arm.mode_agree_weight * 0.5 * ce

        if arm.consistency_weight > 0:
            # the explicit version of what x-pred gets for free: a second,
            # independent noise draw at the same (x, t), penalised for
            # disagreeing about the endpoint
            eps2_m = torch.randn(batch.x_manip.shape, device=dev, generator=gen)
            eps2_a = torch.randn(batch.x_aux.shape, device=dev, generator=gen)
            z2_m = flow.interpolate(batch.x_manip, eps2_m, t)
            z2_a = flow.interpolate(batch.x_aux, eps2_a, t)
            out2 = model(
                z2_m, z2_a, t, batch.cond, batch.embod,
                batch.mask_manip, batch.mask_aux, batch.aux_active, mode=gt_mode,
            )
            xm2 = flow.to_endpoint(out2.manip, z2_m, t, arm.parameterization)
            xa2 = flow.to_endpoint(out2.aux, z2_a, t, arm.parameterization)
            cons, _ = masked_endpoint_loss(
                xm2, xa2, xm.detach(), xa.detach(),
                batch.mask_manip, batch.mask_aux, batch.aux_active, w,
            )
            loss2, _ = masked_endpoint_loss(
                xm2, xa2, batch.x_manip, batch.x_aux,
                batch.mask_manip, batch.mask_aux, batch.aux_active, w,
            )
            loss = 0.5 * (loss + loss2) + arm.consistency_weight * cons

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)

        if step % tr.probe_every == 0 and result.dynamics:
            result.dynamics[-1]["train_loss"] = float(loss.detach())
            result.dynamics[-1]["grad_norm"] = float(gnorm)
            result.dynamics[-1]["loss_manip"] = float(per_head["manip"])
            result.dynamics[-1]["loss_aux"] = float(per_head["aux"])

    # -------------------------------------------------------------- final eval
    ema.copy_to(model)
    model.eval()
    eval_gen = torch.Generator(device=dev).manual_seed(999)
    big = manifold.sample(cfg.eval.n_samples, generator=eval_gen)
    final = probe_suite(cfg, manifold, model, flow, big, full=True)

    for nfe in cfg.eval.nfe_list:
        m, gm, ga = sample_and_score(cfg, manifold, model, flow, big, nfe, record=True)
        for k, v in m.items():
            final[f"nfe{nfe}_{k}"] = v
        if nfe == cfg.eval.nfe_main:
            read = manifold.allocation_readout(big.embod, gm, ga, big.cond)
            result.artifacts["a_manip"] = read["a_manip"].cpu().numpy()
            result.artifacts["a_aux"] = read["a_aux"].cpu().numpy()
            result.artifacts["alpha_gt"] = big.alpha.cpu().numpy()
            result.artifacts["aux_active"] = big.aux_active.cpu().numpy()
            result.artifacts["embod"] = big.embod.cpu().numpy()
            off = (
                (gm - manifold.project_manip(big.embod, gm)).pow(2)
                * big.mask_manip
            ).flatten(1).sum(1) / (
                (gm.pow(2) * big.mask_manip).flatten(1).sum(1).clamp_min(1e-8)
            )
            result.artifacts["off_manifold"] = off.sqrt().cpu().numpy()
            g_gen = torch.cat(
                [
                    manifold.encode_manip(big.embod, gm),
                    manifold.encode_aux(big.embod, ga),
                ],
                dim=1,
            )
            result.artifacts["latent"] = g_gen.cpu().numpy()
            result.artifacts["chunk_example"] = gm[:8].cpu().numpy()

    final["gradient_noise"] = gradient_noise_scale(cfg, manifold, model, flow)
    final["n_params"] = sum(p.numel() for p in model.parameters())
    final["wall"] = time.time() - t0
    result.final = final
    return result


def gradient_noise_scale(cfg, manifold, model, flow, n_draws: int = 8, batch: int = 128):
    """Variance of the gradient across noise draws at fixed data, normalised.

    The clean-sample target does not depend on the flow noise; the velocity
    target does.  Whatever survives in this number is the extra stochasticity
    the parameterization injects into the update, holding the data fixed.
    """
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(31337)
    b = manifold.sample(batch, generator=g)
    t = flow.sample_time(batch, dev)
    grads = []
    was_training = model.training
    model.train()
    for _ in range(n_draws):
        e_m = torch.randn(b.x_manip.shape, device=dev, generator=g)
        e_a = torch.randn(b.x_aux.shape, device=dev, generator=g)
        z_m = flow.interpolate(b.x_manip, e_m, t)
        z_a = flow.interpolate(b.x_aux, e_a, t)
        out = model(
            z_m, z_a, t, b.cond, b.embod, b.mask_manip, b.mask_aux, b.aux_active,
            mode=b.mode if cfg.arm.mode_conditioning else None,
        )
        xm = flow.to_endpoint(out.manip, z_m, t, cfg.arm.parameterization)
        xa = flow.to_endpoint(out.aux, z_a, t, cfg.arm.parameterization)
        loss, _ = masked_endpoint_loss(
            xm, xa, b.x_manip, b.x_aux, b.mask_manip, b.mask_aux, b.aux_active,
            flow.loss_weight(t, cfg.arm.loss_space),
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(
            torch.cat([p.grad.detach().flatten() for p in model.parameters()
                       if p.grad is not None])
        )
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    G = torch.stack(grads)
    mean = G.mean(0)
    var = G.var(0, unbiased=True).sum()
    return float(var / mean.pow(2).sum().clamp_min(1e-12))
