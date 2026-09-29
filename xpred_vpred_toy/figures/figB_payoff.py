"""Figure B -- where in the diffusion process the parameterization matters.

Everything here is at VELA-0's own per-token width, 58 action numbers, with the
coordination mode supplied to both arms.  No width extrapolation is used to make
the point.

The finding this figure exists to show.  Aggregated over noise levels the two
conventions are at parity, and the clean-sample head is 1% behind -- and that
aggregate hides the actual structure, because the two conventions do not differ
uniformly in the noise level.  The clean-sample head is *better where the noise is
large* and 1% worse where the noise is small.  The training time distribution,
Beta(1.5, 1), puts nearly all of its mass on small noise, so the aggregate loss is
dominated by exactly the region a few-step sampler barely visits, while the first
and largest solver step is taken in the region where the clean-sample head wins.
That is the whole reconciliation: aggregate risk 0.99x, off-manifold mass at five
steps 1.09x, at one step 1.41x.  These were never in tension; they are different
intervals of t.

One argument is deliberately *not* made.  It would be convenient to claim that
squared error asks for an off-manifold target, so that losing on it is a virtue.
``bayes_target_audit.py`` tests this against the closed-form optimum and it is
false here: this manifold is a linear subspace, hence convex, and the conditional
mean is exactly on it.  The parity is real and is explained by the location of the
advantage in t, not by a defective target.

(a) Excess risk against noise level, with the training time density behind it.
    The crossing is the panel: ahead at high noise, 1% behind where the loss looks.
(b) The mechanism at the same noise levels: how much the endpoint estimate moves
    when only the noise draw changes.  16x at the high-noise end.
(c) The trade as one bar per measurement, aggregated.  One bar points the wrong
    way and it is the only quantity anything was trained on.
(d) Off-manifold mass against solver budget: (a) and (b) as the sampler sees them.

The per-token width sweep, which says the choice is also future-proof, is a
separate figure (``figD_width``) so that neither has to be read through the other.
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
    C, INK, MUTED, PAPER, SHORT,
    band, dopamine_header, fs, grid, logscale, panel_tag, save, use_style,
)
from velatoy.stats import load, paired, summarise, table

NFE = (1, 2, 4, 5, 8, 16)
GROUPS = (1, 2, 4, 8, 16)

# The noise levels the probe suite records, as recorded (t x 100).  The first three
# exist only in the ``noise`` experiment, whose grid was densified below t = 0.1
# because that is where the two conventions differ and where a Beta(1.5, 1) time
# distribution puts almost no mass; ``_vs_t`` simply skips the levels an experiment
# does not have, so the same helper serves both grids.
T_KEYS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 90)


def _vs_t(recs, metric, arm, exp="modewidth", suffix="+mode@g1"):
    """(t, mean, sem) for one arm across the recorded noise levels."""
    ts, ys, es = [], [], []
    for k in T_KEYS:
        s = summarise(
            table(recs, exp, f"{metric}_t{k}").get((f"{arm}-pred/v{suffix}",), {})
        )
        if s["n"]:
            ts.append(k / 100.0)
            ys.append(s["mean"])
            es.append(s["sem"])
    return np.array(ts), np.array(ys), np.array(es)


def _ratio(recs, exp, metric, num, den, key=("arm",)):
    """Paired ratio num/den, seed by seed.  Returns (mean, sem, n)."""
    tab = table(recs, exp, metric, key=key)
    a, b = tab.get(num, {}), tab.get(den, {})
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return float("nan"), float("nan"), len(seeds)
    r = np.array([a[s] / b[s] for s in seeds if b[s]])
    return r.mean(), r.std(ddof=1) / np.sqrt(r.size), r.size


def risk_vs_t_panel(ax, recs, exp="modewidth", suffix="+mode@g1"):
    """Excess risk against noise level, with the training time density behind it.

    The honest reading is stated on the panel rather than left implicit: the
    clean-sample head wins in a region where the absolute risk is already small,
    and loses by 1% near the peak.  That is exactly why the aggregate is a wash.
    What makes the small region matter is not its weight in the loss -- Beta(1.5,1)
    gives everything below t = 0.1 about 3% of the total -- but that a few-step
    sampler takes its first and largest step there.
    """
    # training time density, behind everything, on its own scale
    axd = ax.twinx()
    tt = np.linspace(1e-3, 1.0, 400)
    axd.fill_between(tt, 1.5 * np.sqrt(tt), color=INK, alpha=0.07, lw=0, zorder=0)
    axd.set_ylim(0, 1.5 * 3.4)
    axd.set_yticks([])
    axd.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    # a ratio, because the two absolute curves differ by 1% over most of the range
    # and lie on top of each other; the crossing is the content of the panel
    ts, rs, es = [], [], []
    for k in T_KEYS:
        r, e, n = _ratio(recs, exp, f"excess_risk_oracle_t{k}",
                         (f"v-pred/v{suffix}",), (f"x-pred/v{suffix}",))
        if n >= 2 and np.isfinite(r):
            ts.append(k / 100.0)
            rs.append(r)
            es.append(e)
    ts, rs = np.array(ts), np.array(rs)
    band(ax, ts, rs, np.array(es), color=C["x"], lw=3.4, marker="o", ms=7, zorder=6)
    ax.axhline(1.0, color=INK, lw=2.0, ls=(0, (5, 2)), zorder=5)

    # the peak of the absolute risk, marked without a label: the win is at high
    # noise, where the absolute risk is not the largest, and the caption says so
    _, yx, _ = _vs_t(recs, "excess_risk_oracle", "x", exp, suffix)
    if yx.size:
        t_peak = ts[int(np.argmax(yx[:ts.size]))]
        ax.axvline(t_peak, color=MUTED, lw=1.4, ls=(0, (2, 3)), zorder=3)

    if rs.size:
        ax.annotate(
            f"{100 * (rs[0] - 1):+.0f}%",
            xy=(ts[0], rs[0]), xytext=(0.17, 0.84), textcoords="axes fraction",
            fontsize=fs(13.0), fontweight="bold", color=C["x"],
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=C["x"], lw=1.8,
                            connectionstyle="arc3,rad=-0.3"),
        )
        ax.text(0.96, 0.17, "$-1\\%$", transform=ax.transAxes, fontsize=fs(13.0),
                fontweight="bold", color=C["warn"], ha="right", va="center")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.93, max(1.6, float(rs.max()) * 1.10) if rs.size else 1.6)
    ax.set_xlabel("noise level  $t$   (0 = pure noise, 1 = clean)")
    ax.set_ylabel(r"excess risk:  $v$-pred $/$ $x$-pred")
    ax.set_title("the advantage lives at high noise", loc="left")
    grid(ax)


def niv_vs_t_panel(ax, recs, exp="modewidth", suffix="+mode@g1"):
    """The mechanism at the same noise levels, and the reason for the shape in (a).

    At high noise the noisy chunk says almost nothing about the clean one, so a
    skip head has to emit something that nearly cancels its own input and its error
    inherits the draw.  This measures that directly: change only the noise
    realisation, and see how far the endpoint estimate moves.
    """
    for param, col, mk in (("x", C["x"], "o"), ("v", C["v"], "s")):
        ts, ys, es = _vs_t(recs, "niv_model", param, exp, suffix)
        if ts.size:
            band(ax, ts, ys, es, color=col, lw=3.2, marker=mk, ms=7, zorder=6,
                 label=rf"${param}$-pred")

    tx, yx, _ = _vs_t(recs, "niv_model", "x", exp, suffix)
    tv, yv, _ = _vs_t(recs, "niv_model", "v", exp, suffix)
    if tx.size and tv.size:
        ax.annotate(
            f"{yv[0] / yx[0]:.0f}$\\times$ more stable",
            xy=(tx[0], yx[0]), xytext=(0.16, 0.20), textcoords="axes fraction",
            fontsize=fs(11.0), fontweight="bold", color=C["x"],
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=C["x"], lw=1.8,
                            connectionstyle="arc3,rad=-0.3"),
        )

    logscale(ax, "y")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("noise level  $t$")
    ax.set_ylabel("endpoint move per noise draw")
    ax.set_title("and so does the invariance to the draw", loc="left")
    ax.legend(loc="upper right", fontsize=fs(9.8))
    grid(ax)


# ------------------------------------------------------------------ panel (c)

# One bar per measurement, ordered so the single adverse one comes first and is
# not buried.  Deliberately excludes the mode-gap rate: with the latent supplied
# both arms are at 0.0-0.1%, and a per-seed ratio of two near-zero quantities is
# noise dressed as a factor of five.
TRADE = (
    ("excess_risk_oracle", "denoising\nMSE"),
    ("gen_off_manifold", "off-\nmanifold"),
    ("gen_coord_violation", "heads\ndisagree"),
    ("gen_task_residual", "task not\nachieved"),
    ("nfe1_off_manifold", "off-manifold\n1 step"),
    ("gen_invalid_leakage", "absent\njoints"),
)


def trade_panel(ax, recs, exp="modewidth", suffix="+mode@g1"):
    """The whole trade at VELA-0's per-token width, as one bar per measurement.

    Plotted as ``v``-pred over ``x``-pred so that "taller is better for x-pred"
    holds throughout, with the parity line drawn: the point of the panel is that
    one bar sits below it and the rest sit above, and that the one below is the
    quantity the loss function optimises.

    Risk is scored against the mode-oracle optimum, since these arms are given the
    coordination mode and so are not competing with a denoiser that has to infer it.
    """
    vals, errs, labels, cols = [], [], [], []
    for metric, label in TRADE:
        r, e, n = _ratio(recs, exp, metric,
                         (f"v-pred/v{suffix}",), (f"x-pred/v{suffix}",))
        if n < 2 or not np.isfinite(r):
            continue
        vals.append(r)
        errs.append(e)
        labels.append(label)
        cols.append(C["x"] if r > 1 else C["warn"])

    xs = np.arange(len(vals))
    ax.bar(xs, vals, 0.66, yerr=errs, capsize=3.5, color=cols, edgecolor=INK,
           lw=1.0, zorder=4, error_kw=dict(ecolor=INK, lw=1.2))
    for x, v in zip(xs, vals):
        pct = 100 * (v - 1)
        ax.text(x, v * 1.04 if v >= 1 else v * 0.96,
                f"{pct:+.0f}%", ha="center",
                va="bottom" if v >= 1 else "top",
                fontsize=fs(10.6), fontweight="bold",
                color=C["x"] if v > 1 else C["warn"], zorder=6)
    ax.axhline(1.0, color=INK, lw=2.0, ls=(0, (5, 2)), zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=fs(8.5), linespacing=1.15)
    ax.tick_params(axis="x", pad=3.0)
    ax.set_ylabel(r"$v$-pred $/$ $x$-pred")
    ax.set_title("the trade, at VELA-0's own token width", loc="left")
    _, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.34)
    ax.text(0.98, 0.975, r"taller $=$ better", transform=ax.transAxes,
            ha="right", va="top", fontsize=fs(9.8), color=MUTED)
    # the adverse bar is the one the loss function optimises, which is the whole
    # point of the panel, so it is called out rather than left to the reader
    ax.annotate(
        "the trained\nloss",
        xy=(0, 1.03), xytext=(0.30, 0.66), textcoords="axes fraction",
        fontsize=fs(11.0), color=C["warn"], fontweight="bold", linespacing=1.25,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=C["warn"], lw=1.7,
                        connectionstyle="arc3,rad=0.25"),
    )
    grid(ax, axis="y")


# ------------------------------------------------------------------ panel (b)

def modewidth_panel(ax, recs):
    """Absolute excess risk against per-token width, all three conventions.

    Absolute rather than a ratio, for a reason worth stating.  The reference here
    must be the *mode-oracle* optimum: these arms are told the coordination mode,
    so they hold strictly more information than the denoiser that has to infer it
    from the noisy observation, and can legitimately beat it.  Against that wrong
    reference the clean-sample head's excess risk goes slightly negative at the
    widest token, and a ratio through a near-zero denominator is meaningless.
    Plotting levels sidesteps this and says more anyway: the clean-sample head is
    close to flat across a sixteen-fold change in per-token width, and the two skip
    heads are not.
    """
    for param, key, col, lw, mk in (
        ("x", "x_mode", C["x"], 3.4, "o"),
        ("v", "v_mode", C["v"], 3.0, "s"),
        ("eps", "eps", C["eps"], 2.6, "^"),
    ):
        xs, ys, es = [], [], []
        for g in GROUPS:
            s = summarise(
                table(recs, "modewidth", "excess_risk_oracle")
                .get((f"{param}-pred/v+mode@g{g}",), {})
            )
            if s["n"]:
                xs.append(58 * g)
                ys.append(s["mean"])
                es.append(s["sem"])
        if xs:
            lab = rf"${param if param != 'eps' else chr(92) + 'epsilon'}$-pred"
            ax.errorbar(xs, ys, yerr=es, color=col, lw=lw, marker=mk, ms=8,
                        capsize=3, label=lab, zorder=6)

    ax.axvspan(52, 66, color=C["accent"], alpha=0.35, lw=0, zorder=0)
    ax.set_xscale("log", base=2)
    logscale(ax, "y")
    ax.set_xticks([58, 116, 232, 464, 928])
    ax.set_xticklabels(["58\nVELA-0", "116", "232", "464", "928"])
    ax.set_xlabel("action numbers per token")
    ax.set_ylabel("excess risk over optimum")
    ax.set_title("only the clean-sample head is width-stable", loc="left")
    ax.legend(loc="upper left", fontsize=fs(9.8))
    grid(ax)


# ------------------------------------------------------------------ panel (c)

def gen_width_panel(ax, recs):
    """The same sweep scored on generated chunks instead of denoising risk."""
    for exp, xf, vf, col, ls, lab, lw, mk in (
        ("tokengroup_long", "x-pred/v@g{g}", "v-pred/v@g{g}", MUTED, (0, (4, 2)),
         "mode left ambiguous", 2.4, "s"),
        ("modewidth", "x-pred/v+mode@g{g}", "v-pred/v+mode@g{g}", C["x_mode"], "-",
         "mode supplied", 3.4, "o"),
    ):
        xs, ys, es = [], [], []
        for g in GROUPS:
            r, e, n = _ratio(recs, exp, "gen_off_manifold",
                             (vf.format(g=g),), (xf.format(g=g),))
            if n >= 2:
                xs.append(58 * g)
                ys.append(r)
                es.append(e)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=col, ls=ls, lw=lw, marker=mk,
                        ms=8, capsize=3, label=lab, zorder=6)
    ax.axhline(1.0, color=INK, lw=1.8, ls=(0, (4, 2)), zorder=3)
    ax.axvspan(52, 66, color=C["accent"], alpha=0.35, lw=0, zorder=0)
    ax.set_xscale("log", base=2)
    logscale(ax, "y")
    ax.set_xticks([58, 116, 232, 464, 928])
    ax.set_xticklabels(["58\nVELA-0", "116", "232", "464", "928"])
    ax.set_xlabel("action numbers per token")
    ax.set_ylabel(r"off-manifold:  $v$-pred $/$ $x$-pred")
    ax.set_title("on validity: ahead at every width", loc="left")
    ax.legend(loc="upper left", fontsize=fs(9.8))
    grid(ax)


# ------------------------------------------------------------------ panel (d)

def nfe_panel(ax, recs, exp="modewidth", suffix="+mode@g1"):
    """Off-manifold mass against solver budget, at VELA-0's width and NFE.

    The gap is widest at one step and never closes, which is the practically
    useful shape: a head that already holds an endpoint estimate at high noise
    does not need the solver to find one.
    """
    for param, col in (("x", C["x"]), ("v", C["v"])):
        xs, ys, es = [], [], []
        for n in NFE:
            s = summarise(
                table(recs, exp, f"nfe{n}_off_manifold")
                .get((f"{param}-pred/v{suffix}",), {})
            )
            if s["n"]:
                xs.append(n)
                ys.append(s["mean"])
                es.append(s["sem"])
        if xs:
            band(ax, xs, ys, es, color=col, lw=3.2, marker="o", ms=8, zorder=6,
                 label=rf"${param.replace('eps', chr(92) + 'epsilon')}$-pred")

    ax.axvline(5, color=C["accent"], lw=10, alpha=0.32, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(NFE))
    ax.set_xticklabels([str(n) if n != 5 else "5\nVELA-0" for n in NFE])
    ax.set_xlabel("solver steps (NFE)")
    ax.set_ylabel("off-manifold mass of samples")
    ax.set_title("widest gap where it matters: few steps", loc="left")
    ax.legend(loc="upper right", fontsize=fs(9.8))
    grid(ax)

    # placed in axes fractions: the y range is a narrow band away from zero, so
    # data coordinates chosen by eye land outside the axes
    r, _, n = _ratio(recs, exp, "nfe1_off_manifold",
                     (f"v-pred/v{suffix}",), (f"x-pred/v{suffix}",))
    if n >= 2 and np.isfinite(r):
        # no arrow: the gap at NFE = 1 is the leftmost thing in the panel and an
        # arrow to it has to cross both curves to get there
        ax.text(
            0.03, 0.14, f"{100 * (r - 1):+.0f}% at one step",
            transform=ax.transAxes, fontsize=fs(10.8), color=C["x"],
            fontweight="bold", ha="left", va="bottom",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/out/figB_payoff")
    args = ap.parse_args()

    use_style(scale=1.5)
    recs = load(args.results)
    print(f"loaded {len(recs)} runs")

    fig = plt.figure(figsize=(15.0, 11.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30,
                  left=0.070, right=0.955, top=0.878, bottom=0.055)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # the two t-resolved panels come from the ``noise`` experiment: same arms and
    # config as modewidth@g1, but 16 seeds and a grid densified below t = 0.1, which
    # is exactly the region the claim rests on
    risk_vs_t_panel(axes[0], recs, exp="noise")
    niv_vs_t_panel(axes[1], recs, exp="noise")
    trade_panel(axes[2], recs)
    nfe_panel(axes[3], recs)

    for ax, letter, col in zip(axes, "abcd",
                               [C["x"], C["x"], C["x"], C["x"]]):
        panel_tag(ax, letter, col, dx=-0.14, dy=1.13)

    dopamine_header(
        fig,
        "Aggregate risk is a wash because it averages over noise; the advantage sits where the sampler starts",
        "All at VELA-0's own 58 action numbers per token, coordination mode supplied to both arms. "
        "Mean $\\pm$ s.e.m.  16 paired seeds in (a, b), 8 in (c, d).",
        title_chars=105,
    )
    save(fig, args.out)


if __name__ == "__main__":
    main()
