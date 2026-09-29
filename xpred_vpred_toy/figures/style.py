"""Shared plotting style: saturated "dopamine" palette on a warm neutral ground.

The colour choices are load-bearing, not decorative.  One hue per output
parameterization is used in every panel of every figure, the two computed
reference levels are always achromatic (black for the optimum, grey for the
no-communication anchor), and the "channel closed" ablations are the warm
variants of their parent hue.  A reader who learns the key once can read any
panel without a legend.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

INK = "#14121F"
PAPER = "#FFFCF7"
GRID = "#E4DCD0"
MUTED = "#6E6A7A"

C = {
    "x": "#FF2E93",
    "v": "#00C2FF",
    "eps": "#8B5CF6",
    "x_nj": "#FF8A00",
    "v_nj": "#00E5A0",
    "v_cons": "#0B4FA8",
    "x_mode": "#B0006B",
    "v_mode": "#00778F",
    "x_agree": "#FFB300",
    "bayes": INK,
    "bayes_local": "#9A94A6",
    "bayes_oracle": "#4A4552",
    "gt": "#00A878",
    "accent": "#FFD400",
    "warn": "#FF3B30",
}

ARM_STYLE = {
    "x-pred/v": dict(color=C["x"], label=r"$x$-pred  (VELA-0)", lw=2.6, zorder=6),
    "v-pred/v": dict(color=C["v"], label=r"$v$-pred  (flow matching)", lw=2.6, zorder=5),
    "eps-pred/v": dict(color=C["eps"], label=r"$\epsilon$-pred", lw=2.2, zorder=4),
    "x-pred/x": dict(color=C["x"], label=r"$x$-pred, unweighted", lw=2.0, ls="--", zorder=3),
    "v-pred/x": dict(color=C["v"], label=r"$v$-pred, unweighted", lw=2.0, ls="--", zorder=3),
    "eps-pred/x": dict(color=C["eps"], label=r"$\epsilon$-pred, unweighted", lw=1.8, ls="--", zorder=2),
    "v-pred/v+cons": dict(color=C["v_cons"], label=r"$v$-pred + explicit consistency", lw=2.2, ls="-.", zorder=4),
    "x-pred/v-nojoint": dict(color=C["x_nj"], label=r"$x$-pred, channel closed", lw=2.0, ls=":", zorder=3),
    "v-pred/v-nojoint": dict(color=C["v_nj"], label=r"$v$-pred, channel closed", lw=2.0, ls=":", zorder=3),
    "x-pred/v+mode": dict(color=C["x_mode"], label=r"$x$-pred $+$ mode latent", lw=2.6, zorder=6),
    "v-pred/v+mode": dict(color=C["v_mode"], label=r"$v$-pred $+$ mode latent", lw=2.4, zorder=5),
    "x-pred/v+agree": dict(color=C["x_agree"], label=r"$x$-pred $+$ mode supervision", lw=2.2, ls="-.", zorder=4),
}

SHORT = {
    "x-pred/v": r"$x$-pred",
    "v-pred/v": r"$v$-pred",
    "eps-pred/v": r"$\epsilon$-pred",
    "x-pred/x": r"$x$-pred$^{\,\rm unw}$",
    "v-pred/x": r"$v$-pred$^{\,\rm unw}$",
    "eps-pred/x": r"$\epsilon$-pred$^{\,\rm unw}$",
    "v-pred/v+cons": r"$v$-pred$\,+\,$cons.",
    "x-pred/v-nojoint": r"$x$-pred, closed",
    "v-pred/v-nojoint": r"$v$-pred, closed",
    "x-pred/v+mode": r"$x$-pred$\,+\,$mode",
    "v-pred/v+mode": r"$v$-pred$\,+\,$mode",
    "x-pred/v+agree": r"$x$-pred$\,+\,$sup.",
}


SCALE = 1.0
"""Set by :func:`use_style`; read by the helpers so hand-placed annotations
scale with the axes rather than shrinking into illegibility."""


def fs(size: float) -> float:
    """A font size in the current figure's scale."""
    return size * SCALE


def use_style(scale: float = 1.0) -> None:
    """Apply the shared style.

    ``scale`` multiplies every type size.  Figures destined for a single column of
    a paper get reduced by a large factor on the page, so the default sizes --
    chosen for reading the PNG at full width -- end up well below the body text.
    Passing ``scale=1.45`` produces type that is still at least body size after
    the figure is scaled to the column.
    """
    global SCALE
    SCALE = scale
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": fs(9.5),
            "axes.labelsize": fs(10.2),
            "axes.titlesize": fs(10.8),
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0 * min(scale, 1.3),
            "axes.labelpad": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": fs(9.0),
            "ytick.labelsize": fs(9.0),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": fs(9.0),
            "legend.handlelength": 1.7,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "text.color": INK,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def logscale(ax, axis="y", **kw) -> bool:
    """Apply a log scale only if the axis actually has positive data on it.

    A log axis with nothing positive plotted on it raises at draw time, which
    turns a half-finished results directory into a crash rather than a
    half-finished figure.  Regenerating figures mid-run is routine, so the guard
    is worth the four lines.
    """
    getter = "get_ydata" if axis == "y" else "get_xdata"
    vals = []
    for ln in ax.get_lines():
        vals.append(np.asarray(getattr(ln, getter)(), dtype=float))
    for coll in ax.collections:
        off = np.asarray(coll.get_offsets(), dtype=float)
        if off.size:
            vals.append(off[:, 1 if axis == "y" else 0])
    for patch in ax.patches:
        h = patch.get_height() if axis == "y" else patch.get_width()
        vals.append(np.array([h], dtype=float))
    if not vals:
        return False
    cat = np.concatenate([v.ravel() for v in vals if v.size])
    if not np.any(np.isfinite(cat) & (cat > 0)):
        return False
    ax.set_yscale("log", **kw) if axis == "y" else ax.set_xscale("log", **kw)
    return True


def grid(ax, axis="both") -> None:
    ax.grid(True, axis=axis, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_tag(ax, letter: str, color: str = INK, dx: float = -0.155, dy: float = 1.14):
    """A filled rounded tag, in the figure's own idiom rather than a bare letter."""
    ax.text(
        dx, dy, letter, transform=ax.transAxes, fontsize=fs(11.5), fontweight="bold",
        va="top", ha="left", color=PAPER,
        bbox=dict(boxstyle="round,pad=0.30", facecolor=color, edgecolor="none"),
    )


def band(ax, x, mean, sem, *, color, alpha=0.20, **kw):
    mean, sem = np.asarray(mean), np.asarray(sem)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=alpha, lw=0, zorder=1)
    return ax.plot(x, mean, color=color, **kw)


def reference_line(ax, y, label, *, color=INK, ls=(0, (5, 2)), lw=1.6, side="right",
                   fontsize=None, va="bottom", pad=0.01):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=3)
    x0, x1 = ax.get_xlim()
    xt = x1 - pad * (x1 - x0) if side == "right" else x0 + pad * (x1 - x0)
    ax.text(
        xt, y, label, ha="right" if side == "right" else "left", va=va,
        fontsize=fontsize if fontsize is not None else fs(8.6), color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.16", fc=PAPER, ec="none", alpha=0.85),
    )


def corridor(ax, x, lo, hi, *, label=None, color="#B9B3C4", alpha=0.35):
    """Shade the region between the two computed anchors.

    Everything achievable without the cross-stream channel lies below the lower
    edge; the upper edge is what optimal use of it buys.  A method's position
    inside the band is the quantity of interest.
    """
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=1,
                    label=label)


def dopamine_header(fig, title: str, subtitle: str | None = None, y=0.985,
                    title_chars=118, sub_chars=175):
    """Figure title and standfirst, wrapped to the figure's own width.

    Wrapping is done here rather than by hand at the call sites because these are
    saved with ``bbox_inches="tight"``: an over-long single-line header silently
    widens the canvas and squashes every panel, which looks like a layout bug in
    the panels rather than a too-long sentence.
    """
    import textwrap

    tlines = textwrap.wrap(title, title_chars) or [""]
    size = fs(13.5)
    fig.text(0.008, y, "\n".join(tlines), fontsize=size, fontweight="bold",
             va="top", color=INK, linespacing=1.25)
    if subtitle:
        # in figure fractions, so it has to be derived from the figure's own height:
        # a constant tuned on a tall figure silently overlaps on a short one
        dy = (size * 1.25 / 72.0) / fig.get_size_inches()[1] * len(tlines)
        fig.text(0.008, y - dy, "\n".join(textwrap.wrap(subtitle, sub_chars)),
                 fontsize=fs(9.3), va="top", color=MUTED, linespacing=1.35)


def blob_legend(fig, entries, *, y=0.005, x=0.008, fontsize=None, ncol=None,
                loc="lower left"):
    """A compact colour key drawn as filled chips rather than line samples."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker=e.get("marker", "o"), ls=e.get("ls", "-"),
               color=e["color"], markersize=6.5 * min(SCALE, 1.4),
               lw=e.get("lw", 2.4), label=e["label"])
        for e in entries
    ]
    fig.legend(
        handles=handles, loc=loc, bbox_to_anchor=(x, y),
        ncol=ncol or len(entries),
        fontsize=fontsize if fontsize is not None else fs(9.4), frameon=False,
        handletextpad=0.5, columnspacing=1.5,
    )


def save(fig, path_stem: str, *, dpi=400):
    import os

    os.makedirs(os.path.dirname(path_stem) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor=PAPER)
    print(f"wrote {path_stem}.png / .pdf")
