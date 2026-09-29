"""Figure A -- the mechanism, in four panels and as few words as possible.

The argument is a chain, and each panel is one link:

(a) The clean action chunk is low-dimensional and the flow targets are not.
    A property of the data, established before any model is trained.
(b) So the network's own input-output map has to span far fewer directions under
    a clean-sample head.  Swept over the noise level, with the untrained network
    as the control that rules out initialisation.
(c) The freed capacity is visible in the residual stream: the clean-sample head
    builds a representation of its own flow noise and then discards it.
(d) And it is spent on the partner stream.  Layer-wise, with two negative
    controls, because a ridge read-out with this many features needs them.

Deliberately no long captions inside the axes: each panel gets a short title, a
labelled axis, and at most one annotation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figures.style import (
    C, GRID, INK, MUTED, PAPER, SHORT,
    band, blob_legend, dopamine_header, fs, grid, logscale, panel_tag, save,
    use_style,
)
from velatoy.stats import dynamics_curve, load, summarise, table

HEAD = ("x-pred/v", "v-pred/v", "eps-pred/v")
JAC_T = (0.1, 0.3, 0.5, 0.7, 0.9)
EXP = "main_hd"          # 232 numbers per token, where the counts are cleanest


# ------------------------------------------------------------------ panel (a)

def geometry_panel(ax):
    """Singular spectra of the three targets on the real toy manifold."""
    import torch

    from velatoy.config import DataConfig
    from velatoy.data import VelaToyManifold

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = VelaToyManifold(DataConfig(), seed=1234, device=dev)
    torch.manual_seed(0)
    b = man.sample(4096)
    keep = b.embod == 1                      # one embodiment, one definite subspace
    x = b.x_manip[keep].flatten(1)
    eps = torch.randn_like(x)
    v = x - eps

    floor = 1e-5
    for tag, mat, col, lw in (
        (r"noise $\epsilon$", eps, C["eps"], 2.4),
        (r"velocity $v$", v, C["v"], 2.8),
        (r"clean chunk $x$", x, C["x"], 3.4),
    ):
        m = mat - mat.mean(0, keepdim=True)
        s = torch.linalg.svdvals(m.float()).cpu().numpy()
        s = np.maximum(s / s[0], floor)
        ax.plot(np.arange(1, len(s) + 1), s, color=col, lw=lw, label=tag)

    ax.axhspan(floor, 3e-5, color=INK, alpha=0.07, lw=0, zorder=0)
    ax.axvline(man.latent_dim, color=INK, ls=(0, (4, 2)), lw=1.6, zorder=2)
    ax.annotate(
        f"rank {man.latent_dim}", xy=(man.latent_dim, 4e-3),
        xytext=(man.latent_dim * 3.2, 3e-2), fontsize=fs(10.5), color=INK,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.4),
    )
    ax.set_xscale("log")
    logscale(ax, "y")
    ax.set_xlim(1, 928)
    ax.set_ylim(floor, 2.4)
    ax.set_xlabel("singular value index  (of 928)")
    ax.set_ylabel("normalised singular value")
    ax.set_title("the clean chunk is low-dimensional", loc="left")
    ax.legend(loc="lower left", fontsize=fs(9.6))
    grid(ax)


# ------------------------------------------------------------------ panel (b)

def jac_rank_panel(ax, recs):
    """Participation ratio of d(net)/dz, swept over the noise level.

    Reported as a curve rather than a number because the algebra that forces the
    skip heads to be full rank holds at every ``t``, so a single ``t`` invites the
    reading that the value was chosen.  The untrained control is the important
    line: the convention is a property of training, not of the random network, so
    all three arms must start together.
    """
    tok_dim = None
    for arm in HEAD:
        xs, ys, es = [], [], []
        for t in JAC_T:
            s = summarise(table(recs, EXP, f"jac_pr_t{int(t * 100)}").get((arm,), {}))
            if s["n"]:
                xs.append(t)
                ys.append(s["mean"])
                es.append(s["sem"])
        d = summarise(table(recs, EXP, "jac_tok_dim").get((arm,), {}))
        if d["n"]:
            tok_dim = int(d["mean"])
        if xs:
            band(ax, xs, ys, es, color=C[arm.split("-")[0]], lw=3.0, marker="o",
                 ms=7, zorder=6, label=SHORT[arm])

    if tok_dim:
        ax.axhline(tok_dim, color=MUTED, lw=1.6, ls=(0, (1, 1.5)), zorder=2)
        ax.text(0.5, tok_dim, f"per-token width {tok_dim}", fontsize=fs(9.2),
                color=MUTED, va="bottom", ha="center", fontweight="bold")
    ctrl = _untrained_pr(tok_dim or 232)
    if ctrl:
        ax.axhline(ctrl, color=INK, lw=2.0, ls=(0, (5, 2)), zorder=5)
        ax.text(0.9, ctrl, "untrained (all three)", fontsize=fs(9.4), color=INK,
                va="bottom", ha="right", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none", alpha=0.9))
    ax.axhline(13, color=C["gt"], lw=1.8, ls=(0, (3, 2)), zorder=4)
    ax.text(0.1, 13, "manifold dim 13", fontsize=fs(9.4), color=C["gt"],
            va="bottom", ha="left", fontweight="bold")

    logscale(ax, "y")
    ax.set_xlabel("flow time  $t$")
    ax.set_ylabel(r"effective rank of $\partial\,\mathrm{net}/\partial z_t$")
    ax.set_title("...so its map needs far fewer directions", loc="left")
    ax.legend(loc="center right", fontsize=fs(9.6))
    grid(ax)


_PR_CACHE: dict[int, float] = {}


def _untrained_pr(tok_dim: int):
    """The same measurement on a randomly initialised network of the same shape."""
    if tok_dim in _PR_CACHE:
        return _PR_CACHE[tok_dim]
    try:
        import torch

        from velatoy.arms import main_arms
        from velatoy.config import ModelConfig, RunConfig, TrainConfig
        from velatoy.probes import net_jacobian_spectrum
        from velatoy.train import build

        cfg = RunConfig(
            arm=main_arms()[0],
            model=ModelConfig(token_group=max(1, tok_dim // 58)),
            train=TrainConfig(steps=0),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        man, model, flow = build(cfg)
        b = man.sample(256)

        def raw(bb, zm, za, tt):
            return model(zm, za, tt, bb.cond, bb.embod, bb.mask_manip, bb.mask_aux,
                         bb.aux_active).manip

        val = net_jacobian_spectrum(raw, b, 0.5, flow, tok_dim=tok_dim,
                                    n_batch=24)["jac_participation_ratio"]
    except Exception as exc:  # noqa: BLE001
        print(f"untrained control unavailable: {exc}")
        val = None
    _PR_CACHE[tok_dim] = val
    return val


# ------------------------------------------------------------------ panel (c)

def shedding_panel(ax, recs):
    """Own-noise decodability over training, against its own shuffled control."""
    for arm in HEAD:
        x, m, s = dynamics_curve(recs, EXP, arm, "probe_eps_manip")
        if len(x):
            band(ax, x, m, s, color=C[arm.split("-")[0]], lw=3.0, zorder=6,
                 label=SHORT[arm])
    sh = summarise(table(recs, EXP, "probe_eps_manip_shuf").get(("x-pred/v",), {}))
    if sh["n"]:
        y = max(sh["mean"], 0.0)
        ax.axhline(y, color=INK, lw=1.8, ls=(0, (5, 2)), zorder=4)
        ax.text(11800, y - 0.015, "control: pairing destroyed", fontsize=fs(9.4),
                color=INK, ha="right", va="top", fontweight="bold")
    ax.annotate(
        "learns the noise,\nthen discards it", xy=(1300, 0.44), xytext=(3400, 0.28),
        fontsize=fs(10.4), color=C["x"], fontweight="bold", linespacing=1.3,
        arrowprops=dict(arrowstyle="->", color=C["x"], lw=1.7,
                        connectionstyle="arc3,rad=-0.3"),
    )
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$R^2$:  stream $\rightarrow$ own $\epsilon$")
    ax.set_title("the capacity it stops spending", loc="left")
    ax.set_ylim(-0.15, 1.10)
    grid(ax)
    ax.legend(loc="center right", fontsize=fs(9.6), bbox_to_anchor=(1.0, 0.62))


# ------------------------------------------------------------------ panel (d)

def routing_panel(ax, recs, n_layers=4):
    """Layer-wise partner-stream decodability, with both controls.

    ``x``-pred and ``v``-pred are the comparison; the channel-closed arm and the
    shuffled pairing are the controls.  Without them a rising curve could be the
    probe fitting the target's marginal rather than information that actually
    crossed between the streams.
    """
    xs = np.arange(n_layers + 1)
    curves = (
        ("x-pred/v", C["x"], 3.4, "-", "o"),
        ("v-pred/v", C["v"], 3.0, "-", "s"),
        ("x-pred/v-nojoint", C["x_nj"], 2.4, (0, (2, 1.6)), "^"),
    )
    for arm, col, lw, ls, mk in curves:
        ys, es = [0.0], [0.0]        # input embedding carries no partner info yet
        ok = False
        for li in range(n_layers):
            s = summarise(
                table(recs, EXP, f"probe_partner_manip_L{li}").get((arm,), {})
            )
            if s["n"]:
                ok = True
            ys.append(s["mean"])
            es.append(s["sem"])
        if ok:
            band(ax, xs, ys, es, color=col, lw=lw, ls=ls, marker=mk, ms=7.5,
                 zorder=6, label=SHORT[arm])

    sh = summarise(table(recs, EXP, "probe_partner_manip_shuf").get(("x-pred/v",), {}))
    if sh["n"]:
        y = max(sh["mean"], 0.0)
        ax.axhline(y, color=INK, lw=1.8, ls=(0, (5, 2)), zorder=4)
        ax.text(n_layers, y - 0.012, "control: pairing destroyed ",
                fontsize=fs(9.4), color=INK, ha="right", va="top",
                fontweight="bold")
    ceil = summarise(table(recs, EXP, "ceiling_partner_manip").get(("x-pred/v",), {}))
    if ceil["n"]:
        ax.axhline(ceil["mean"], color=C["gt"], lw=2.0, ls=(0, (4, 2)), zorder=4)
        ax.text(0.98, ceil["mean"] - 0.012, "the most that is knowable",
                transform=ax.get_yaxis_transform(), fontsize=fs(9.6),
                color=C["gt"], ha="right", va="top", fontweight="bold")
        ax.set_ylim(-0.10, ceil["mean"] + 0.09)

    ax.set_xticks(xs)
    ax.set_xticklabels(["in"] + [f"B{i + 1}" for i in range(n_layers)])
    ax.set_xlabel("residual stream, by block")
    ax.set_ylabel(r"$R^2$:  arm stream $\rightarrow$ base's chunk")
    ax.set_title("...and where it spends it instead", loc="left")
    grid(ax)
    ax.legend(loc="upper left", fontsize=fs(9.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/out/figA_mechanism")
    args = ap.parse_args()

    use_style(scale=1.5)
    recs = load(args.results)
    print(f"loaded {len(recs)} runs")

    fig = plt.figure(figsize=(15.0, 11.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.26,
                  left=0.065, right=0.985, top=0.862, bottom=0.052)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    geometry_panel(axes[0])
    jac_rank_panel(axes[1], recs)
    shedding_panel(axes[2], recs)
    routing_panel(axes[3], recs)

    for ax, letter, col in zip(axes, "abcd", [C["x"], C["v"], C["x"], C["gt"]]):
        panel_tag(ax, letter, col, dx=-0.13, dy=1.13)

    dopamine_header(
        fig,
        "Clean-chunk prediction changes what the action expert has to represent",
        "232 action numbers per token. One architecture, one loss, one data stream, "
        "one initialisation; only the output head differs. Mean $\\pm$ s.e.m., 8 paired seeds.",
        title_chars=90,
    )
    save(fig, args.out)


if __name__ == "__main__":
    main()
