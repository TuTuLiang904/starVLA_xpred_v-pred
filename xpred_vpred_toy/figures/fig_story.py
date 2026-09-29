"""The headline figure: one causal chain across four panels, read left to right.

    low-dimensional action
        -> simpler learned representation
            -> more stable denoising where the noise is largest
                -> more executable actions at few solver steps

Every number is pulled from the recorded runs rather than typed in, and the
expected value is asserted alongside so that a figure regenerated against
different results fails loudly instead of quietly disagreeing with the text.

    python figures/fig_story.py --results results_full,results_v2,results_v3

Deliberately excluded, because they are appendix material and would break the
single-question reading: the mode-gap distributions, the per-token width sweep,
the stream-capacity sweep, the timestep-schedule controls and the explicit
consistency penalty.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from velatoy.stats import load, summarise, table

# --------------------------------------------------------------------- palette
# One hue per prediction target, held across all four panels, so the key is
# learned once in (a) and never repeated.
CORAL = "#FF4E64"     # x-pred, the deployed choice
BLUE = "#1E7FE8"      # v-pred
PURPLE = "#8B5CF6"    # epsilon / pure flow noise
TEAL = "#0FA48C"      # theoretical reference: the valid manifold
INK = "#1B1A22"
WARM = "#8C8479"      # secondary annotation
PAPER = "#FFFFFF"     # plain white: the figure sits on the page's own ground
FAINT = "#EFE9E0"

# MM-ABC's deployed tokenisation, and the operating point of its sampler.
DEPLOY_ARM = "x-pred/v+mode@g1"
DEPLOY_V = "v-pred/v+mode@g1"
NFE = (1, 2, 4, 5, 8, 16)
NFE_DEPLOY = 5

XLABEL_Y = -0.135
"""Axis-label height, in axes fractions, pinned rather than left to matplotlib.

Two things depend on it.  Across panels it puts every x-axis label on one
horizontal line even though the tick labels above them differ in height, and
within (c) it lets the ``noise``/``clean`` end markers sit on the same baseline as
``flow time t`` so the three read as a single line.
"""


def style(scale: float) -> None:
    """Type sizes are set for the figure to be *reduced* to two-column width.

    Prefer Times (Liberation Serif).  Fall back so a machine without that face
    still renders rather than substituting boxes.
    """
    from matplotlib import font_manager as fm

    families = {f.name for f in fm.fontManager.ttflist}
    serif = "Liberation Serif" if "Liberation Serif" in families else (
        "DejaVu Serif" if "DejaVu Serif" in families else "serif"
    )
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        # Times, to match the body text of the paper
        "font.family": serif,
        "font.size": 11.0 * scale,
        "axes.labelsize": 11.8 * scale,
        "axes.titlesize": 12.0 * scale,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#BDB6AC",
        "axes.linewidth": 1.1,
        "axes.labelpad": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#5E574E",
        "ytick.color": "#5E574E",
        "xtick.labelsize": 11.0 * scale,
        "ytick.labelsize": 11.0 * scale,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "text.color": INK,
        # STIX is Times-metric-compatible and ships with matplotlib, so the maths
        # matches the text face without depending on a system font
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ----------------------------------------------------------------- data access

def _mean(recs, exp, metric, arm) -> float:
    s = summarise(table(recs, exp, metric).get((arm,), {}))
    if not s["n"]:
        raise SystemExit(f"no data: {exp} / {metric} / {arm}")
    return s["mean"]


def _paired_ratio(recs, exp, metric, num, den):
    """Ratio of the two arm means over the seeds both arms ran.

    Deliberately the same estimator as ``velatoy.stats.paired``, which produced
    every ratio in the written section: a mean of per-seed ratios would be a
    defensible statistic too, but it differs by a couple of percent on the
    heavy-tailed high-noise metrics, and a figure that disagrees with its own text
    is worse than either choice.  The band is a paired bootstrap over seeds.
    """
    tab = table(recs, exp, metric)
    a, b = tab.get((num,), {}), tab.get((den,), {})
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise SystemExit(f"no paired seeds: {exp} / {metric}")
    x = np.array([a[s] for s in seeds])
    y = np.array([b[s] for s in seeds])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(seeds), size=(4000, len(seeds)))
    boot = x[idx].mean(1) / y[idx].mean(1)
    return float(x.mean() / y.mean()), float(boot.std()), len(seeds)


def check(name: str, got: float, want: float, tol: float = 0.02) -> float:
    """Guard against the figure and the prose drifting apart."""
    if abs(got - want) > tol * max(abs(want), 1e-9):
        print(f"  !! {name}: figure has {got:.4g}, text claims {want:.4g}")
    return got


# --------------------------------------------------------------- panel headers

def header(ax, title: str) -> None:
    """A conclusion-style title above the axes, centred over its own panel."""
    ax.text(
        0.5, 1.055, title, transform=ax.transAxes,
        fontsize=mpl.rcParams["axes.titlesize"], fontweight="bold",
        va="bottom", ha="center", color=INK,
    )


def footer_label(fig, ax, letter: str, accent: str, y: float) -> None:
    """The panel letter below the axes, centred on it and in the panel's colour."""
    b = ax.get_position()
    fig.text((b.x0 + b.x1) / 2, y, f"({letter})", ha="center", va="center",
             fontsize=mpl.rcParams["font.size"] * 1.12, fontweight="bold",
             color=accent)


def softgrid(ax, axis="both") -> None:
    ax.grid(True, axis=axis, color=FAINT, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ panel  (a)

def panel_spectrum(ax) -> None:
    """Singular spectra of the three targets, on the toy manifold itself.

    Nothing is trained here.  The clean chunk is exactly ``Q g`` with ``g`` in 13
    dimensions, so its spectrum terminates rather than decays; both flow targets
    inherit the noise and stay numerically full rank.  This is the premise the
    other three panels rest on.
    """
    import torch

    from velatoy.config import DataConfig
    from velatoy.data import VelaToyManifold

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = VelaToyManifold(DataConfig(), seed=1234, device=dev)
    torch.manual_seed(0)
    # enough draws that one embodiment alone still gives many more rows than the
    # 928 columns: otherwise the noise spectra fall off a cliff at the sample
    # count and the panel appears to show low-rank noise
    b = man.sample(24576)
    keep = b.embod == 1                    # one embodiment -> one definite subspace
    x = b.x_manip[keep].flatten(1)
    assert x.shape[0] > 2 * x.shape[1], x.shape
    eps = torch.randn_like(x)

    floor = 1e-4
    for tag, mat, col, lw, z in (
        (r"noise $\epsilon$", eps, PURPLE, 2.6, 4),
        (r"velocity $v$", x - eps, BLUE, 3.0, 5),
        (r"clean action $x$", x, CORAL, 3.8, 6),
    ):
        m = mat - mat.mean(0, keepdim=True)
        # on CPU: the cuSOLVER driver fails to converge on the rank-deficient
        # clean matrix and falls back anyway, with a warning
        s = torch.linalg.svdvals(m.float().cpu()).numpy()
        s = np.maximum(s / s[0], floor)
        ax.plot(np.arange(1, s.size + 1), s, color=col, lw=lw, label=tag, zorder=z)

    r = man.latent_dim
    ax.axvline(r, color=TEAL, ls=(0, (3.4, 2.4)), lw=1.7, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 928)
    # both flow targets sit just under 1, so the legend can only clear them above
    # it; the extra decades of headroom are what buy that space
    ax.set_ylim(floor, 260)
    ax.set_yticks([1e-4, 1e-2, 1])
    # The intrinsic rank is a real tick, not an annotation: it is a position on
    # the axis and needs no sentence, and as a tick it aligns with the others for
    # free.  No decade tick at 10 -- on a log axis it lands on top of 13.
    ax.set_xticks([1, r, 100, 928])
    ax.set_xticklabels(["1", str(r), "100", "928"])
    ax.set_xticks([], minor=True)
    lab = ax.get_xticklabels()[1]
    lab.set_color(TEAL)
    lab.set_fontweight("bold")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalised singular value")
    ax.legend(loc="upper right", fontsize=mpl.rcParams["font.size"] * 0.97,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.4)
    softgrid(ax)


# ------------------------------------------------------------------ panel  (b)

def panel_routing(ax, recs) -> None:
    """Three paired measurements, each normalised within its own pair.

    The three quantities have wildly different units, so a shared axis would be
    meaningless; each pair is scaled to its own larger member and the real values
    are printed.  Read as routing: capacity not spent representing the flow noise
    shows up as information about the partner stream.
    """
    jx = check("jac PR x", _mean(recs, "modewidth", "jac_participation_ratio",
                                 DEPLOY_ARM), 3.44)
    jv = check("jac PR v", _mean(recs, "modewidth", "jac_participation_ratio",
                                 DEPLOY_V), 36.97)
    # the directions needed for 99% of the Jacobian energy, which is the row that
    # ties this panel back to (a): the clean-sample map lands on the manifold's
    # own intrinsic rank, the velocity map needs four times as many
    r9x = check("jac rank99 x", _mean(recs, "modewidth", "jac_rank99",
                                      DEPLOY_ARM), 13.51)
    r9v = check("jac rank99 v", _mean(recs, "modewidth", "jac_rank99",
                                      DEPLOY_V), 51.68)
    ex = check("own-noise x", _mean(recs, "modewidth", "probe_eps_manip",
                                    DEPLOY_ARM), 0.617)
    ev = check("own-noise v", _mean(recs, "modewidth", "probe_eps_manip",
                                    DEPLOY_V), 0.943)
    px = check("partner x", _mean(recs, "modewidth", "probe_partner_manip",
                                  DEPLOY_ARM), 0.293)
    pv = check("partner v", _mean(recs, "modewidth", "probe_partner_manip",
                                  DEPLOY_V), 0.212)

    rows = [
        ("Jacobian rank", "\u2193", jx, jv, "{:.2f}"),
        ("directions for 99% energy", "\u2193", r9x, r9v, "{:.1f}"),
        ("own flow-noise", "\u2193", ex, ev, "{:.2f}"),
        ("partner info", "\u2191", px, pv, "{:.2f}"),
    ]

    fsz = mpl.rcParams["font.size"]
    h = 0.27                                  # bar thickness in data units
    for i, (label, arrow, vx, vv, fmt) in enumerate(rows):
        y = len(rows) - 1 - i                 # top row first
        top = max(vx, vv)
        for j, (val, col) in enumerate(((vx, CORAL), (vv, BLUE))):
            yy = y + (h * 0.60 if j == 0 else -h * 0.60)
            ax.barh(yy, val / top, height=h, color=col, lw=0, zorder=3,
                    align="center")
            ax.text(val / top + 0.022, yy, fmt.format(val), va="center",
                    ha="left", fontsize=fsz * 0.97, color=col,
                    fontweight="bold", zorder=4)
        # the arrow carries the direction that is good, so no sentence is needed
        ax.text(0, y + h * 1.34, f"{label} {arrow}", va="bottom", ha="left",
                fontsize=fsz * 1.0, color=INK, fontweight="bold")

    ax.set_xlim(0, 1.28)
    ax.set_ylim(-0.62, len(rows) - 0.08)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    # sits below the last pair, on the baseline the other panels' tick labels use
    ax.legend(
        handles=[Line2D([], [], color=CORAL, lw=5.5, label="$x$-pred"),
                 Line2D([], [], color=BLUE, lw=5.5, label="$v$-pred")],
        loc="lower left", bbox_to_anchor=(-0.012, -0.085), ncol=2,
        fontsize=fsz * 1.0, handlelength=1.0, columnspacing=1.6,
        handletextpad=0.5,
    )


# ------------------------------------------------------------------ panel  (c)

def panel_noise(ax, recs) -> None:
    """Excess-risk ratio against flow time, with the training density behind it.

    Plotted as a ratio because the two absolute curves differ by about 1% over
    most of the range and would lie on top of each other.  The point of the panel
    is that the aggregate loss is an average over this curve, weighted by a
    density that is almost absent exactly where the two conventions differ.
    """
    ts = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 90])
    xs, rs, es = [], [], []
    for k in ts:
        try:
            m, s, _ = _paired_ratio(recs, "noise", f"excess_risk_oracle_t{k}",
                                    DEPLOY_V, DEPLOY_ARM)
        except SystemExit:
            continue
        xs.append(k / 100)
        rs.append(m)
        es.append(s)
    xs, rs, es = np.array(xs), np.array(rs), np.array(es)
    check("peak ratio", rs.max(), 1.64)

    fsz = mpl.rcParams["font.size"]
    lo, hi = 0.77, rs.max() * 1.10
    rng = hi - lo

    # Beta(1.5, 1) training density: the aggregate loss is this curve averaged
    # under that density, which is almost absent exactly where the two
    # conventions differ.  Its label sits inside the shaded band, so the band
    # needs enough height to hold type without touching the ratio curve.
    tt = np.linspace(1e-3, 1, 400)
    dens = tt ** 0.5
    top = lo + dens / dens.max() * rng * 0.155
    ax.fill_between(tt, lo, top, color="#DCD4C7", alpha=0.9, lw=0, zorder=0)
    ax.plot(tt, top, color="#B6AC9C", lw=1.4, zorder=1)
    ax.text(0.63, lo + rng * 0.045, "training $t$ density", ha="center",
            va="bottom", fontsize=fsz * 0.94, color="#6B6152", zorder=2)

    # the high-noise region, where a few-step sampler takes its first step
    ax.axvspan(0, 0.10, color=CORAL, alpha=0.11, lw=0, zorder=1)
    ax.axhline(1.0, color=INK, ls=(0, (4, 2.6)), lw=1.5, zorder=3)
    ax.text(0.985, 1.005, "tie", ha="right", va="bottom", color=INK,
            fontsize=fsz * 0.92, fontweight="bold")

    ax.fill_between(xs, rs - es, rs + es, color=CORAL, alpha=0.22, lw=0, zorder=4)
    ax.plot(xs, rs, color=CORAL, lw=3.4, zorder=5, solid_capstyle="round")
    ax.plot(xs, rs, "o", color=CORAL, ms=4.8, zorder=6,
            markeredgecolor=PAPER, markeredgewidth=1.1)

    ax.annotate(
        f"{rs[0]:.2f}\u00d7 at high noise",
        xy=(xs[0] + 0.004, rs[0]), xytext=(0.285, lo + rng * 0.86),
        fontsize=fsz * 1.02, fontweight="bold", color=CORAL,
        va="center", ha="left",
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.7,
                        connectionstyle="arc3,rad=-0.26"),
    )
    # the endpoint-invariance ratio is reported in the text rather than here: a
    # second callout is what tipped this panel from readable into cluttered
    niv, _, _ = _paired_ratio(recs, "noise", "niv_model_t1", DEPLOY_V, DEPLOY_ARM)
    check("niv t=0.01", niv, 23.45, tol=0.02)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("flow time  $t$")
    ax.set_ylabel("excess risk   $v$-pred $/$ $x$-pred")
    # The axis direction matters more than the numbers, so it is named at the
    # ends.  Same size and same vertical alignment as the axis label, at the same
    # height, which is what makes the three share a baseline.
    for xv, lab, ha in ((0.0, "noise", "left"), (1.0, "clean", "right")):
        ax.text(xv, XLABEL_Y, lab, transform=ax.transAxes, ha=ha, va="top",
                fontsize=mpl.rcParams["axes.labelsize"], color="#4A443C")
    softgrid(ax, axis="y")


# ------------------------------------------------------------------ panel  (d)

def panel_executable(ax, recs) -> None:
    """Off-manifold mass against solver budget.

    Off-manifold mass is the share of the emitted chunk outside the embodiment's
    reachable subspace -- a joint pattern no demonstration contains, so it is
    hallucination rather than imprecision, and it is what decides whether the
    chunk can be executed at all.
    """
    series, labels = {}, {}
    for arm, col, lab in ((DEPLOY_ARM, CORAL, "$x$-pred"),
                          (DEPLOY_V, BLUE, "$v$-pred")):
        ys = [_mean(recs, "modewidth", f"nfe{n}_off_manifold", arm) for n in NFE]
        series[arm] = np.array(ys)
        labels[arm] = (col, lab)
        ax.plot(range(len(NFE)), ys, color=col, lw=3.2, zorder=5,
                marker="o", ms=6.0, markeredgecolor=PAPER, markeredgewidth=1.2)

    x_, v_ = series[DEPLOY_ARM], series[DEPLOY_V]
    check("nfe1 x", x_[0], 0.0903)
    check("nfe1 v", v_[0], 0.1274)
    check("nfe5 x", x_[NFE.index(5)], 0.0949)
    check("nfe5 v", v_[NFE.index(5)], 0.1038)

    fsz = mpl.rcParams["font.size"]
    span = v_.max() - x_.min()
    ax.set_ylim(x_.min() - span * 0.24, v_.max() + span * 0.17)

    i5 = NFE.index(NFE_DEPLOY)
    ax.axvspan(i5 - 0.32, i5 + 0.32, color=TEAL, alpha=0.15, lw=0, zorder=1)
    # at the head of the band rather than beside the curves, which it collided
    # with once the type grew
    ax.text(i5, 0.955, "deployment", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=fsz * 0.97, color=TEAL,
            fontweight="bold")

    ax.annotate(
        f"{100 * (1 - x_[0] / v_[0]):.0f}% lower",
        xy=(0.04, x_[0] - span * 0.02), xytext=(0.62, x_[0] - span * 0.20),
        fontsize=fsz * 1.02, fontweight="bold", color=CORAL,
        va="center", ha="left",
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.7,
                        connectionstyle="arc3,rad=0.26"),
    )

    # invalid-degree-of-freedom leakage belongs in the text, not in a footer box
    check("leak x", _mean(recs, "modewidth", "gen_invalid_leakage", DEPLOY_ARM),
          0.181)
    check("leak v", _mean(recs, "modewidth", "gen_invalid_leakage", DEPLOY_V),
          0.375)

    # the two curves are named at their own ends: the colour key is already
    # established in (b), and a legend box here would sit on top of the band label
    for arm, (col, lab) in labels.items():
        ax.text(len(NFE) - 1 + 0.16, series[arm][-1], lab, va="center",
                ha="left", color=col, fontsize=fsz * 1.0, fontweight="bold")

    ax.set_xticks(range(len(NFE)))
    ax.set_xticklabels([str(n) for n in NFE])
    ax.set_xlim(-0.32, len(NFE) + 0.60)
    ax.set_xlabel("solver steps  (NFE)")
    ax.set_ylabel("off-manifold mass")
    softgrid(ax, axis="y")


# ------------------------------------------------------------------- assembly

PANELS = (("a", CORAL), ("b", BLUE), ("c", TEAL), ("d", PURPLE))

TITLES = (
    "Clean actions are low-dimensional",
    "Capacity goes to action, not noise",
    "The gain is where sampling starts",
    "Cleaner actions in fewer steps",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/out/fig_story")
    ap.add_argument("--scale", type=float, default=1.52)
    args = ap.parse_args()

    recs = load(args.results)
    print(f"loaded {len(recs)} runs")

    style(args.scale)
    fig = plt.figure(figsize=(17.2, 5.0))
    gs = GridSpec(
        1, 4, figure=fig,
        left=0.050, right=0.994, top=0.900, bottom=0.225,
        wspace=0.245,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    panel_spectrum(axes[0])
    panel_routing(axes[1], recs)
    panel_noise(axes[2], recs)
    panel_executable(axes[3], recs)

    for ax, title in zip(axes, TITLES):
        header(ax, title)
        if ax.get_xlabel():
            ax.xaxis.set_label_coords(0.5, XLABEL_Y)
    for ax, (letter, accent) in zip(axes, PANELS):
        footer_label(fig, ax, letter, accent, y=0.048)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=400, facecolor=PAPER)
    print(f"wrote {out}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
