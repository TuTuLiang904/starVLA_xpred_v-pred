"""Figure C -- the one cost, shown as a distribution rather than a statistic.

The coordination plane is the whole story in two axes.  Each generated chunk is
reduced to how much of the task each head actually did: the arm's share
``a_manip`` and the base's share ``a_aux``.  Two facts about the ground truth make
the plane readable without a caption:

* every valid plan satisfies ``a_manip + a_aux = 1`` -- the task is accomplished
  exactly once, however the work is divided -- so the data is a line, and leaving
  the line means the two heads planned against each other;
* only two points on that line are legal, because the coordination mode is binary,
  so sliding *along* the line means blending two valid plans into a third that no
  demonstration contains.

Top row is the plane, bottom row the marginal over ``a_manip`` against the ground
truth.  Reading left to right: what the data looks like, what a velocity head
does, what a clean-sample head does, and what one shared discrete latent fixes.
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
    C, GRID, INK, MUTED, PAPER,
    dopamine_header, fs, grid, panel_tag, save, use_style,
)
from velatoy.stats import load, summarise, table

ALO, AHI = 0.35, 0.65
LIM = (0.08, 0.92)
GAP = 0.5 * (AHI - ALO) / 2          # half-width of the "between the modes" band


def _artifacts(recs, exp, arm):
    for r in recs:
        if r["experiment"] == exp and r["arm"] == arm and r.get("artifacts"):
            return {k: np.asarray(v) for k, v in r["artifacts"].items()}
    return None


def _shares(art):
    live = art["aux_active"] > 0
    return art["a_manip"][live], art["a_aux"][live]


def plane(ax, am, aa, col, *, title, show_y, is_gt=False):
    ax.plot(LIM, [1 - LIM[0], 1 - LIM[1]], color=INK, lw=1.6, ls=(0, (5, 2)),
            zorder=2, alpha=0.75)
    n = min(len(am), 2200)
    idx = np.random.default_rng(0).choice(len(am), n, replace=False)
    ax.scatter(am[idx], aa[idx], s=7.0, c=col, alpha=0.32, lw=0, zorder=3,
               rasterized=True)
    for a in (ALO, AHI):
        ax.scatter([a], [1 - a], s=210, facecolor="none", edgecolor=INK, lw=2.2,
                   zorder=6)
    if not is_gt:
        ax.scatter([0.5], [0.5], marker="x", s=95, c=C["warn"], lw=2.6, zorder=6)
        frac = float(np.mean(np.abs(am - 0.5) < GAP))
        ax.text(0.96, 0.955, f"{100 * frac:.0f}% between modes",
                transform=ax.transAxes, fontsize=fs(10.6), fontweight="bold",
                va="top", ha="right", color=C["warn"] if frac > 0.03 else C["gt"])
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal")
    ax.set_xticks([0.2, 0.35, 0.65, 0.8])
    ax.set_yticks([0.2, 0.35, 0.65, 0.8])
    if show_y:
        ax.set_ylabel(r"base's share  $a_{\rm aux}$")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, loc="left", color=col, fontsize=fs(11.2))
    grid(ax)


def marginal(ax, am, col, *, gt_am=None, show_y, ymax=None):
    bins = np.linspace(*LIM, 64)
    if gt_am is not None:
        ax.hist(gt_am, bins=bins, density=True, color=MUTED, alpha=0.28, lw=0,
                zorder=2)
    ax.hist(am, bins=bins, density=True, color=col, alpha=0.85, lw=0, zorder=3)
    for a in (ALO, AHI):
        ax.axvline(a, color=INK, lw=1.5, ls=(0, (4, 2)), zorder=5)
    ax.axvspan(0.5 - GAP, 0.5 + GAP, color=C["warn"], alpha=0.13, lw=0, zorder=1)
    ax.set_xlim(*LIM)
    ax.set_xticks([0.2, 0.35, 0.65, 0.8])
    ax.set_xlabel(r"arm's share  $a_{\rm manip}$")
    # a shared, clipped ceiling: the ground truth is nearly a pair of spikes, so
    # letting each column autoscale would make the mass between the modes -- the
    # quantity the row exists to show -- invisible in three panels out of four
    if ymax:
        ax.set_ylim(0, ymax)
    if show_y:
        ax.set_ylabel("density")
    else:
        ax.set_yticklabels([])
    grid(ax, axis="x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/out/figC_distributions")
    ap.add_argument(
        "--width", default="58", choices=("58", "232"),
        help="per-token action width; 58 is VELA-0 as it stands",
    )
    args = ap.parse_args()

    use_style(scale=1.5)
    recs = load(args.results)
    print(f"loaded {len(recs)} runs")

    # the ambiguous-mode arms and the mode-conditioned arm live in different
    # experiments at 58/token, so the column spec carries its own experiment
    if args.width == "58":
        base, tok = "main", "58"
        cols = [
            ("ground truth", None, None, C["gt"]),
            (r"$v$-pred", base, "v-pred/v", C["v"]),
            (r"$x$-pred", base, "x-pred/v", C["x"]),
            (r"$x$-pred $+$ mode latent", "modewidth", "x-pred/v+mode@g1",
             C["x_mode"]),
        ]
    else:
        base, tok = "fix", "232"
        cols = [
            ("ground truth", None, None, C["gt"]),
            (r"$v$-pred", base, "v-pred/v", C["v"]),
            (r"$x$-pred", base, "x-pred/v", C["x"]),
            (r"$x$-pred $+$ mode latent", base, "x-pred/v+mode", C["x_mode"]),
        ]

    fig = plt.figure(figsize=(16.0, 9.0))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[2.45, 1.0], hspace=0.34,
                  wspace=0.16, left=0.058, right=0.988, top=0.860, bottom=0.135)

    # the reference is the recorded ground-truth allocation of the same batch, so
    # the comparison is against the data the arms were scored on rather than
    # against the two nominal mode values
    gt = _artifacts(recs, base, "x-pred/v")
    gt_am = None
    if gt is not None and "alpha_gt" in gt:
        live = gt["aux_active"] > 0
        gt_am = np.asarray(gt["alpha_gt"])[live]

    for j, (title, exp, arm, col) in enumerate(cols):
        ax_p = fig.add_subplot(gs[0, j])
        ax_m = fig.add_subplot(gs[1, j])
        if arm is None:
            am = gt_am
            aa = 1.0 - am
            plane(ax_p, am, aa, col, title=title, show_y=(j == 0), is_gt=True)
        else:
            art = _artifacts(recs, exp, arm)
            if art is None:
                for a in (ax_p, ax_m):
                    a.text(0.5, 0.5, f"no artifacts\n{arm}", ha="center",
                           va="center", transform=a.transAxes, color=MUTED)
                continue
            am, aa = _shares(art)
            plane(ax_p, am, aa, col, title=title, show_y=(j == 0))
        marginal(ax_m, am, col, gt_am=gt_am if arm is not None else None,
                 show_y=(j == 0), ymax=26)
        # inside the axes: the titles are already at the top-left of each column,
        # so an outside tag lands on top of them
        panel_tag(ax_p, "abcd"[j], col, dx=0.028, dy=0.965)

    fig.text(
        0.058, 0.055,
        r"dashed: $a_{\rm manip}+a_{\rm aux}=1$        "
        r"$\circ$  the two legal modes        "
        r"$\times$  their average",
        fontsize=fs(9.4), color=MUTED, va="top",
    )
    dopamine_header(
        fig,
        "An unresolved coordination mode makes both heads invent a plan; one shared token removes it",
        f"Generated chunks reduced to each head's realised share of the task. {tok} numbers per token, "
        "5 solver steps, seed 0. The blending is a property of squared-error point estimation, not of "
        "the output convention -- which is why it has to be removed before the two conventions can be compared.",
        title_chars=95,
    )
    save(fig, f"{args.out}_{tok}")


if __name__ == "__main__":
    main()
