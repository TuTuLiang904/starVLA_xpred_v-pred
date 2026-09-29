"""Figure D -- the same comparison against per-token action width.

This is a secondary claim and is kept in its own figure so it is not read as the
main result.  Figure B establishes what the two conventions do at VELA-0's own
tokenisation.  This one asks whether that finding is stable if the tokenisation
changes, and the answer is that it is stable in one direction only: the
clean-sample head's risk is close to flat across a sixteen-fold change in
per-token width, and both skip conventions degrade monotonically.

So the claim this figure supports is not "the clean-sample head is better".  It is
"the choice is cost-neutral now and does not have to be revisited later", which is
the useful thing to know when picking a convention for an architecture whose
tokenisation may widen.

Risk is scored against the mode-oracle optimum throughout.  These arms are handed
the coordination mode and so hold strictly more information than the denoiser that
must infer it from the noisy chunk; against the wrong anchor the clean-sample head's
excess risk goes slightly negative at the widest token and any ratio through it is
meaningless.

(a) Absolute excess risk, all three conventions.  Levels rather than ratios, so it
    is visible which arm moved.
(b) The same sweep scored on emitted chunks, with the mode-ambiguous sweep behind
    it as a control: the trend is not an artefact of supplying the mode.
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
    C, INK, MUTED,
    dopamine_header, fs, grid, logscale, panel_tag, save, use_style,
)
from velatoy.stats import load, paired, summarise, table

GROUPS = (1, 2, 4, 8, 16)


def _ratio(recs, exp, metric, num, den):
    tab = table(recs, exp, metric, key=("arm",))
    a, b = tab.get(num, {}), tab.get(den, {})
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return float("nan"), float("nan"), len(seeds)
    r = np.array([a[s] / b[s] for s in seeds if b[s]])
    return r.mean(), r.std(ddof=1) / np.sqrt(r.size), r.size


def _width_ticks(ax):
    ax.axvspan(52, 66, color=C["accent"], alpha=0.35, lw=0, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([58, 116, 232, 464, 928])
    ax.set_xticklabels(["58\nVELA-0", "116", "232", "464", "928"])
    ax.set_xlabel("action numbers per token")


def risk_panel(ax, recs):
    for param, col, lw, mk in (
        ("x", C["x"], 3.4, "o"), ("v", C["v"], 3.0, "s"), ("eps", C["eps"], 2.6, "^"),
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

    _width_ticks(ax)
    logscale(ax, "y")
    ax.set_ylabel("excess risk over optimum")
    ax.set_title("only the clean-sample head is width-stable", loc="left")
    ax.legend(loc="upper left", fontsize=fs(10.0))
    grid(ax)


def validity_panel(ax, recs):
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
    _width_ticks(ax)
    logscale(ax, "y")
    ax.set_ylabel(r"off-manifold:  $v$-pred $/$ $x$-pred")
    ax.set_title("on validity: ahead at every width", loc="left")
    ax.legend(loc="upper left", fontsize=fs(10.0))
    grid(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/out/figD_width")
    args = ap.parse_args()

    use_style(scale=1.5)
    recs = load(args.results)
    print(f"loaded {len(recs)} runs")

    fig = plt.figure(figsize=(14.2, 6.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.26,
                  left=0.075, right=0.985, top=0.775, bottom=0.145)
    axes = [fig.add_subplot(gs[0, j]) for j in range(2)]

    risk_panel(axes[0], recs)
    validity_panel(axes[1], recs)
    for ax, letter in zip(axes, "ab"):
        panel_tag(ax, letter, C["x"], dx=-0.13, dy=1.12)

    dopamine_header(
        fig,
        "The choice does not have to be revisited if the action token widens",
        "Supervision, depth, stream width and step budget fixed; only how many chunk steps share a token changes. "
        "8 paired seeds, mean $\\pm$ s.e.m.",
        title_chars=100,
    )
    save(fig, args.out)


if __name__ == "__main__":
    main()
