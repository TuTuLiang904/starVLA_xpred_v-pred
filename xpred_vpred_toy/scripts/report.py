#!/usr/bin/env python
"""Turn the raw runs into the numbers that go in the paper.

Writes ``report.md`` and ``metrics.csv``.  Every comparison is paired by seed and
carries a bootstrap interval and an exact Wilcoxon p-value, and every reference
level is the closed-form optimum computed on the same noise rather than a
best-observed baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from velatoy.stats import interaction, load, paired, summarise, table

HEADLINE = [
    ("excess_risk", "excess denoising risk over Bayes ($t{=}0.2$)", "lower"),
    ("block_det", "risk, mode-carrying block", "lower"),
    ("block_style", "risk, isotropic style block", "lower"),
    ("block_offman", "risk, off-manifold", "lower"),
    ("niv_model", "endpoint spread over noise draws", "lower"),
    ("noise_jacobian", "noise Jacobian at $t{=}0.2$", "lower"),
    ("gradient_noise", "gradient noise at fixed data", "lower"),
    ("probe_eps_manip", "own-noise decodability from the stream", "lower"),
    ("pr_manip", "participation ratio of the stream", "lower"),
    ("probe_partner_manip", "partner-chunk decodability", "higher"),
    ("probe_mode_manip", "mode decodability from the stream", "higher"),
    ("mode_acc_model", "mode chosen by the manip head alone", "higher"),
    ("xstream_utilization_det", "cross-stream gain captured", "higher"),
    ("nfe5_coord_violation", "coordination error, 5 Euler steps", "lower"),
    ("nfe2_coord_violation", "coordination error, 2 Euler steps", "lower"),
    ("nfe5_off_manifold", "off-manifold residual, 5 steps", "lower"),
    ("nfe5_mode_mismatch", "heads disagree on the mode, 5 steps", "lower"),
    ("nfe5_sliced_wasserstein", "sliced Wasserstein in latent space", "lower"),
    ("nfe5_mode_collapse", "blended-mode fraction, 5 steps", "lower"),
]

ARMS = [
    "x-pred/v", "v-pred/v", "eps-pred/v",
    "x-pred/x", "v-pred/x", "eps-pred/x",
    "v-pred/v+cons", "x-pred/v-nojoint", "v-pred/v-nojoint",
]


def fmt(v, n=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    return f"{v:.{n}f}"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    ap = argparse.ArgumentParser()
    # later directories win on same-named records, so the re-run goes last
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--csv", default="metrics.csv")
    args = ap.parse_args()

    recs = load(args.results)
    by_exp = {}
    for r in recs:
        by_exp.setdefault(r["experiment"], []).append(r)
    L = []
    L.append("# VELA-toy: x-prediction vs noised-quantity prediction\n")
    L.append(
        f"{len(recs)} runs: "
        + ", ".join(f"{k} ({len(v)})" for k, v in sorted(by_exp.items()))
        + "\n"
    )
    L.append(
        "Every arm at a given seed shares the generative model, the initial weights, "
        "the batch sequence, the noise sequence and the timestep sequence. Under the "
        "velocity weighting the three parameterizations are the *same loss function*, "
        "so a difference between them is a difference of parameterization and nothing "
        "else. Reference levels are closed-form optima evaluated on the same noise.\n"
    )

    # ------------------------------------------------------- main comparison
    for exp, title in (
        ("main", "Operating point A -- VELA-0 as it stands "
                 "(one token per chunk step: 58 action numbers per token, 512-wide stream)"),
        ("main_hd", "Operating point B -- four chunk steps per token "
                    "(232 action numbers per token, same supervision, same stream width)"),
    ):
        if exp not in by_exp:
            continue
        L.extend(operating_point(recs, exp, title))

    # -------------------------------------------------- where the payoff lives
    L.extend(sampling_payoff(recs))
    if "modewidth" in by_exp:
        L.extend(mode_width(recs))

    # ------------------------------------------- the mode-commitment follow-up
    if "fix" in by_exp:
        L.extend(mode_fix(recs))
    if "seqlen" in by_exp:
        L.extend(width_or_length(recs))

    # ------------------------------------------------------------- the sweeps
    for exp, xkey, xlabel in (
        ("tokengroup", "token_dim", "action numbers per token (4k steps)"),
        ("tokengroup_long", "token_dim", "action numbers per token (12k steps)"),
        ("width", "width", "expert stream width"),
        ("intrinsic", "style_dim", "private latent dims per stream (g = 1)"),
        ("intrinsic_hd", "style_dim", "private latent dims per stream (g = 4)"),
        ("schedule", "schedule", "timestep distribution"),
    ):
        if exp not in by_exp:
            continue
        L.append(f"\n## Sweep: {xlabel}\n")
        for metric in ("excess_risk", "block_det", "nfe5_coord_violation",
                       "nfe5_off_manifold"):
            tab = table(recs, exp, metric, key=("arm", xkey))
            xs = sorted({x for _, x in tab})
            if not xs:
                continue
            L.append(f"\n**{metric}**\n")
            L.append(f"| {xlabel} | " + " | ".join(str(x) for x in xs) + " |")
            L.append("|---" * (len(xs) + 1) + "|")
            bases = sorted({a.split("@")[0] for a, _ in tab})
            got = {}
            for base in bases:
                row = []
                for x in xs:
                    key = next(
                        (k for k in tab if k[1] == x and k[0].split("@")[0] == base),
                        None,
                    )
                    s = summarise(tab.get(key, {})) if key else {"mean": float("nan")}
                    row.append(s["mean"])
                got[base] = row
                L.append(f"| {base} | " + " | ".join(fmt(v) for v in row) + " |")
            if "x-pred/v" in got and "v-pred/v" in got:
                ratio = [
                    v / x if x and math.isfinite(v / x) else float("nan")
                    for x, v in zip(got["x-pred/v"], got["v-pred/v"])
                ]
                L.append(
                    "| **v-pred / x-pred** | "
                    + " | ".join(f"**{fmt(v, 2)}x**" for v in ratio)
                    + " |"
                )

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")

    # ------------------------------------------------------------------- csv
    fields = sorted({k for r in recs for k in r["final"]})
    meta = ["experiment", "arm", "parameterization", "loss_space", "joint_attention",
            "consistency_weight", "mode_conditioning", "mode_agree_weight",
            "width", "style_dim", "chunk", "token_group", "token_dim", "n_tokens",
            "schedule", "seed"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(meta + fields)
        for r in recs:
            w.writerow(
                [r.get(m) for m in meta] + [r["final"].get(f) for f in fields]
            )
    print(f"wrote {args.csv} ({len(recs)} rows, {len(fields)} metrics)")


FIX_ARMS = [
    "x-pred/v", "v-pred/v", "x-pred/v+mode", "v-pred/v+mode", "x-pred/v+agree",
]

# What the intervention has to achieve, and what it must not spend to get there.
FIX_DECISIVE = [
    ("nfe5_mode_mismatch", "heads disagree on the mode, 5 steps", "lower"),
    ("nfe5_mode_collapse", "blended-mode fraction, 5 steps", "lower"),
    ("nfe5_coord_violation", "coordination error, 5 steps", "lower"),
    ("nfe2_coord_violation", "coordination error, 2 steps", "lower"),
    ("nfe5_task_residual", "task residual to the nearer valid mode", "lower"),
]
FIX_MECHANISM = [
    ("jac_participation_ratio", "network Jacobian participation ratio", "lower"),
    ("jac_rank99", "directions holding 99% of the Jacobian", "lower"),
    ("probe_eps_manip", "own-noise decodability from the stream", "lower"),
    ("probe_partner_manip", "partner-chunk decodability", "higher"),
    ("niv_model", "endpoint spread over noise draws", "lower"),
    ("nfe5_off_manifold", "off-manifold residual, 5 steps", "lower"),
    ("excess_risk_oracle", "excess risk over the mode-oracle optimum", "lower"),
]


NFE_LIST = (1, 2, 4, 5, 8, 16)


def sampling_payoff(recs):
    """The comparison scored on samples rather than on denoising risk.

    Excess risk is a point-estimate quantity, and at a narrow token the two
    conventions are algebraically close enough that it separates them barely or
    not at all.  What a policy emits is not a conditional mean, though: it is the
    output of a short Euler integration, and the questions that matter there are
    whether the result is a physically realisable chunk and how few steps it takes
    to get one.  Those are different measurements and they do separate.
    """
    L = ["\n## Scored on samples, not on denoising risk\n"]
    L.append(
        "`off_manifold` is the fraction of the emitted chunk that lies outside the "
        "embodiment's reachable subspace -- joint patterns no demonstration "
        "contains, so it is hallucination rather than imprecision. "
        "`endpoint_drift` is how far the endpoint estimate still moves once the "
        "solver starts, which is the implicit-consistency claim as a measurement. "
        "Ratios are paired within seed; above 1 favours *x*-pred.\n"
    )
    for exp, label in (
        ("main", "58 numbers per token (VELA-0 as it stands)"),
        ("main_hd", "232 numbers per token"),
        ("fix", "232 numbers per token, coordination mode supplied"),
    ):
        pairs = (
            ("x-pred/v+mode", "v-pred/v+mode") if exp == "fix"
            else ("x-pred/v", "v-pred/v")
        )
        rows = []
        for metric in ("off_manifold", "coord_violation", "task_residual",
                       "endpoint_drift", "sliced_wasserstein"):
            cells = []
            for n in NFE_LIST:
                tab = table(recs, exp, f"nfe{n}_{metric}")
                a, b = tab.get((pairs[0],), {}), tab.get((pairs[1],), {})
                if len(set(a) & set(b)) < 2:
                    cells.append("--")
                    continue
                # b over a, so the number matches the column header and "above 1
                # favours x-pred" reads the same way in every table here
                p = paired(b, a)
                cells.append(f"{p['ratio']:.2f}x{stars(p['p_wilcoxon'])}")
            if any(c != "--" for c in cells):
                rows.append((metric, cells))
        if not rows:
            continue
        L.append(f"\n**{label}** -- {pairs[1]} / {pairs[0]}, by solver budget\n")
        L.append("| metric | " + " | ".join(f"NFE {n}" for n in NFE_LIST) + " |")
        L.append("|---" * (len(NFE_LIST) + 1) + "|")
        for metric, cells in rows:
            L.append(f"| {metric} | " + " | ".join(cells) + " |")
    L.append(
        "\nRead the NFE = 1 column of `off_manifold` first. A clean-sample head "
        "already holds an endpoint estimate at high noise, so one step produces a "
        "usable chunk, which is the same fact the `endpoint_drift` row states "
        "directly. Note also where *x*-pred loses: `sliced_wasserstein` at low NFE "
        "and `task_residual` at high NFE, both of which are distributional coverage "
        "rather than validity, and both of which the mode token improves.\n"
    )
    return L


def mode_width(recs):
    """The width sweep with the coordination mode supplied.

    ``main`` and ``tokengroup`` leave the mode ambiguous, which is not a neutral
    choice: the target is then bimodal, the squared-error optimum is the illegal
    midpoint of the two modes, and the only tie-breaking signal in the problem is
    the flow noise -- the quantity a clean-sample head is trained to discard.  That
    confound is worth a large fraction of the risk and it is charged entirely to
    ``x``-pred.  A real action expert conditions on language or a goal, so the mode
    is resolved before the sampler ever runs.  This is that comparison.
    """
    L = ["\n## The same sweep with the coordination mode supplied\n"]
    L.append(
        "Both arms get the shared mode token, so the comparison is between output "
        "conventions and nothing else. `ambiguous` repeats `tokengroup_long`, where "
        "the mode is left to the sampler, at the same budget.\n"
    )
    L.append(
        "Risk here is `excess_risk_oracle`, against the optimum that *knows* the "
        "coordination mode. This is not cosmetic. A mode-conditioned arm is told the "
        "mode, so it holds strictly more information than the denoiser that has to "
        "infer it from the noisy observation, and it can legitimately beat that "
        "denoiser: at 928 numbers per token, x-pred + mode reaches 0.1593 against a "
        "full-Bayes level of 0.1623, making `excess_risk` negative and any ratio "
        "through it meaningless. Against the mode-oracle level, 0.1557, the excess is "
        "positive and small. The ambiguous row keeps the same anchor for comparability, "
        "which is conservative there -- those arms are not given the mode.\n"
    )
    groups = (1, 2, 4, 8, 16)
    for metric in ("excess_risk_oracle", "gen_off_manifold",
                   "nfe1_off_manifold", "gen_coord_violation"):
        head = [f"{58 * g}" for g in groups]
        L.append(f"\n**{metric}** -- v-pred / x-pred\n")
        L.append("| mode | " + " | ".join(head) + " |")
        L.append("|---" * (len(groups) + 1) + "|")
        for exp, tag, xf, vf in (
            ("tokengroup_long", "ambiguous", "x-pred/v@g{g}", "v-pred/v@g{g}"),
            ("modewidth", "**supplied**", "x-pred/v+mode@g{g}", "v-pred/v+mode@g{g}"),
        ):
            cells = []
            for g in groups:
                tab = table(recs, exp, metric, key=("arm",))
                a = tab.get((xf.format(g=g),), {})
                b = tab.get((vf.format(g=g),), {})
                if len(set(a) & set(b)) < 2:
                    cells.append("--")
                    continue
                p = paired(b, a)          # v over x, as the header says
                cells.append(f"{p['ratio']:.2f}x{stars(p['p_wilcoxon'])}")
            L.append(f"| {tag} | " + " | ".join(cells) + " |")

    # the absolute levels too, since a ratio hides which arm moved
    L.append("\n**Absolute excess risk over the mode-oracle optimum, mode supplied**\n")
    L.append("| arm | " + " | ".join(f"{58 * g}" for g in groups) + " |")
    L.append("|---" * (len(groups) + 1) + "|")
    for param in ("x", "v", "eps"):
        cells = []
        for g in groups:
            s = summarise(
                table(recs, "modewidth", "excess_risk_oracle").get(
                    (f"{param}-pred/v+mode@g{g}",), {}
                )
            )
            cells.append(fmt(s["mean"]) if s["n"] else "--")
        L.append(f"| {param}-pred + mode | " + " | ".join(cells) + " |")
    return L


def mode_fix(recs):
    """Can the decisiveness be recovered without giving up the mechanism?

    The two tables are deliberately separate.  The first asks whether the
    intervention buys back mode commitment, which is the one thing the
    clean-sample head consistently loses.  The second asks what it cost: an
    intervention that fixes commitment by making the clean-sample head behave like
    a velocity head would show up as the mechanistic advantages collapsing, and
    would be no answer at all.
    """
    exp = "fix"
    present = [a for a in FIX_ARMS if table(recs, exp, "excess_risk").get((a,))]
    L = [
        "\n## Recovering mode commitment (232 numbers per token)\n",
        "The clean-sample head's one consistent cost is decisiveness. With a "
        "bimodal target and no discrete latent, the noise draw is the only thing "
        "that can break the tie between two equally valid modes, so a head that "
        "follows the draw is handed a tie-break while a head trained to ignore the "
        "draw is pushed toward the conditional mean -- on a bimodal target, the "
        "illegal midpoint. The invariance and the indecision have one cause, so the "
        "question is whether the tie-break can come from somewhere else.\n",
        "`+mode` supplies it explicitly: one shared discrete coordination-mode "
        "token, observed in training and drawn from its known uniform prior at "
        "sampling time, so neither head can commit to a different mode and no part "
        "of the recorded chunk leaks in. `+agree` instead supervises both streams "
        "to predict the same mode label, giving the sampler nothing to commit to; "
        "it separates *representing* the shared decision from *acting* on one draw "
        "of it. Both parameterizations get `+mode`, because handing over the mode "
        "makes the problem easier for either head and only the difference is "
        "evidence.\n",
        "A mode-conditioned arm is not solving the inference problem, so its risk "
        "is read against the mode-oracle optimum rather than the full Bayes "
        "optimum; both anchors are in the CSV for every arm.\n",
    ]

    for tag, block in (
        ("Decisiveness -- what the intervention is for", FIX_DECISIVE),
        ("Mechanism -- what it cost", FIX_MECHANISM),
    ):
        L.append(f"\n### {tag}\n")
        L.append("| metric | better | " + " | ".join(present) + " |")
        L.append("|---|:--:|" + "---:|" * len(present))
        for key, label, better in block:
            row = [
                fmt(summarise(table(recs, exp, key).get((a,), {}))["mean"])
                for a in present
            ]
            L.append(f"| {label} | {better} | " + " | ".join(row) + " |")
        ns = [
            summarise(table(recs, exp, "excess_risk").get((a,), {}))["n"]
            for a in present
        ]
        L.append("| seeds | | " + " | ".join(str(n) for n in ns) + " |")

    L.append("\n### Paired contrasts\n")
    L.append(
        "Three questions. Does the mode latent help the clean-sample head at all? "
        "Does the clean-sample head with the latent beat the velocity head with the "
        "same latent -- the like-for-like comparison? And does it beat the plain "
        "velocity head, which is the deployed alternative?\n"
    )
    contrasts = [
        ("x-pred/v+mode", "x-pred/v", "mode latent added to x-pred"),
        ("x-pred/v+mode", "v-pred/v+mode", "x vs v, both with the mode latent"),
        ("x-pred/v+mode", "v-pred/v", "x-pred + mode latent vs plain v-pred"),
        ("x-pred/v+agree", "x-pred/v", "mode supervision added to x-pred"),
    ]
    for a, b, why in contrasts:
        ta = table(recs, exp, "excess_risk").get((a,))
        tb = table(recs, exp, "excess_risk").get((b,))
        if not ta or not tb:
            continue
        L.append(f"\n**{why}** (`{a}` minus `{b}`)\n")
        L.append("| metric | better | A | B | diff | 95% CI | rel. | win | p | |")
        L.append("|---|:--:|---:|---:|---:|:--:|---:|---:|---:|:--|")
        for key, label, better in FIX_DECISIVE + FIX_MECHANISM:
            r = paired(
                table(recs, exp, key).get((a,), {}),
                table(recs, exp, key).get((b,), {}),
            )
            if r.get("n", 0) < 2 or not math.isfinite(r["diff"]):
                continue
            L.append(
                f"| {label} | {better} | {fmt(r['mean_a'])} | {fmt(r['mean_b'])} | "
                f"{r['diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                f"{100 * r['rel_change']:+.1f}% | {r['win_rate']:.2f} | "
                f"{r['p_wilcoxon']:.4f} | {stars(r['p_wilcoxon'])} |"
            )
    return L


SEQLEN_CONFIGS = [
    ("T16g1", "16 tokens x 58", "reference"),
    ("T4g1", "4 tokens x 58", "shorter sequence only"),
    ("T4g4", "4 tokens x 232", "shorter sequence and wider token"),
]


def width_or_length(recs):
    """Folding chunk steps into a token raises per-token width *and* shortens the
    sequence.  Those are confounded at operating point B; this separates them.

    Absolute risk is not comparable across the three configurations -- the short
    chunk carries a quarter of the supervision and a smaller ambient space -- so
    only the ratio between the parameterizations within a configuration is read.
    """
    exp = "seqlen"
    L = [
        "\n## Is operating point B about per-token width or about sequence length?\n",
        "Grouping four chunk steps into one token raises the per-token action "
        "width from 58 to 232 and shortens the sequence from 16 tokens to 4. Both "
        "move together in `main_hd`. `T4g1` moves only the length, at 58 numbers "
        "per token, so if the clean-sample head's advantage appears in `T4g4` but "
        "not in `T4g1`, the shorter sequence is not what produces it.\n",
        "The short chunk carries a quarter of the supervision, so absolute risk is "
        "not comparable across configurations; the ratio within a configuration "
        "is.\n",
    ]
    metrics = [
        ("excess_risk", "excess denoising risk"),
        ("nfe5_off_manifold", "off-manifold residual, 5 steps"),
        ("jac_participation_ratio", "network Jacobian participation ratio"),
        ("probe_eps_manip", "own-noise decodability"),
    ]
    L.append("| metric | " + " | ".join(f"{lab}" for _, lab, _ in SEQLEN_CONFIGS) + " |")
    L.append("|---" * (len(SEQLEN_CONFIGS) + 1) + "|")
    for key, label in metrics:
        cells = []
        for tag, _, _ in SEQLEN_CONFIGS:
            x = summarise(table(recs, exp, key).get((f"x-pred/v@{tag}",), {}))["mean"]
            v = summarise(table(recs, exp, key).get((f"v-pred/v@{tag}",), {}))["mean"]
            if not (math.isfinite(x) and math.isfinite(v)) or abs(x) < 1e-12:
                cells.append("--")
                continue
            r = paired(
                table(recs, exp, key).get((f"x-pred/v@{tag}",), {}),
                table(recs, exp, key).get((f"v-pred/v@{tag}",), {}),
            )
            cells.append(
                f"{fmt(x, 3)} / {fmt(v, 3)} = **{v / x:.2f}x** "
                f"({r.get('win_rate', float('nan')):.2f}, p={r.get('p_wilcoxon', float('nan')):.3f})"
            )
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append(
        "\nEach cell is x-pred / v-pred = ratio, with the win rate for x-pred and "
        "the exact Wilcoxon p-value in brackets.\n"
    )
    return L


def operating_point(recs, exp, title):
    """Everything the design can say at one operating point.

    Kept as a function of ``exp`` because the study has two: VELA-0's current
    configuration, and the same thing with four chunk steps folded into each
    token.  Reporting only the one where the effect is larger would be a choice
    about the conclusion rather than about the experiment.
    """
    L = [f"\n## {title}\n"]
    ref = {}
    for k in ("risk_bayes_full", "risk_bayes_local", "det_risk_full",
              "det_risk_local", "niv_bayes", "mode_acc_bayes_full",
              "mode_acc_bayes_local", "ceiling_partner_manip"):
        ref[k] = summarise(table(recs, exp, k).get(("x-pred/v",), {}))["mean"]
    L.append("Computed reference levels (identical for every arm):\n")
    L.append("| quantity | Bayes, both streams | Bayes, own stream only |")
    L.append("|---|---:|---:|")
    L.append(f"| denoising risk, $t=0.2$ | {fmt(ref['risk_bayes_full'])} | {fmt(ref['risk_bayes_local'])} |")
    L.append(f"| risk, mode-carrying block | {fmt(ref['det_risk_full'])} | {fmt(ref['det_risk_local'])} |")
    L.append(f"| mode accuracy | {fmt(ref['mode_acc_bayes_full'])} | {fmt(ref['mode_acc_bayes_local'])} |")
    L.append(f"| endpoint spread (floor) | {fmt(ref['niv_bayes'])} | -- |")
    L.append(f"| partner-chunk $R^2$ (ceiling) | {fmt(ref['ceiling_partner_manip'])} | -- |")

    present = [a for a in ARMS if table(recs, exp, "excess_risk").get((a,))]
    L.append("\n### Final metrics, mean over seeds\n")
    L.append("| metric | " + " | ".join(present) + " |")
    L.append("|---" * (len(present) + 1) + "|")
    for key, label, _ in HEADLINE:
        row = [
            fmt(summarise(table(recs, exp, key).get((a,), {}))["mean"])
            for a in present
        ]
        L.append(f"| {label} | " + " | ".join(row) + " |")
    ns = [summarise(table(recs, exp, "excess_risk").get((a,), {}))["n"] for a in present]
    L.append(f"| seeds | " + " | ".join(str(n) for n in ns) + " |")

    L.append("\n### Paired comparison: x-pred vs v-pred (both velocity-weighted)\n")
    L.append(
        "Negative `diff` favours x-pred on a lower-is-better metric. `win` is the "
        "fraction of seeds in which x-pred is the smaller of the two.\n"
    )
    L.append("| metric | better | x-pred | v-pred | diff | 95% CI | rel. | win | p | |")
    L.append("|---|:--:|---:|---:|---:|:--:|---:|---:|---:|:--|")
    for key, label, better in HEADLINE:
        r = paired(
            table(recs, exp, key).get(("x-pred/v",), {}),
            table(recs, exp, key).get(("v-pred/v",), {}),
        )
        if r.get("n", 0) < 2:
            continue
        L.append(
            f"| {label} | {better} | {fmt(r['mean_a'])} | {fmt(r['mean_b'])} | "
            f"{r['diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{100 * r['rel_change']:+.1f}% | {r['win_rate']:.2f} | "
            f"{r['p_wilcoxon']:.4f} | {stars(r['p_wilcoxon'])} |"
        )

    L.append("\n### Parameterization vs loss weighting (3 x 2)\n")
    L.append(
        "If the effect came from the `1/(1-t)^2` weighting rather than from the "
        "output convention, the columns would differ and the rows would not.\n"
    )
    for key, label, _ in (HEADLINE[0], HEADLINE[4], HEADLINE[7], HEADLINE[9]):
        L.append(f"\n**{label}**\n")
        L.append("| parameterization | velocity-weighted | unweighted |")
        L.append("|---|---:|---:|")
        for p, tag in (("x", "x-pred"), ("v", "v-pred"), ("eps", "eps-pred")):
            a = summarise(table(recs, exp, key).get((f"{tag}/v",), {}))["mean"]
            b = summarise(table(recs, exp, key).get((f"{tag}/x",), {}))["mean"]
            L.append(f"| {tag} | {fmt(a)} | {fmt(b)} |")

    L.append("\n### Does an explicit consistency penalty reproduce the effect?\n")
    L.append(
        "v-pred plus a penalty on endpoint disagreement across two independent "
        "noise draws of the same chunk. This buys, at the cost of a second forward "
        "pass, the property x-pred has by construction.\n"
    )
    L.append(
        "`gap closed` is reported only where the x-pred/v-pred gap is itself "
        "resolved -- at least three times the paired standard error. Where the two "
        "endpoints of the interval coincide to within noise, the fraction of the "
        "way between them is not a meaningful number and is left blank.\n"
    )
    L.append("| metric | v-pred | v-pred + consistency | x-pred | gap closed |")
    L.append("|---|---:|---:|---:|---:|")
    for key, label, _ in HEADLINE:
        vs = summarise(table(recs, exp, key).get(("v-pred/v",), {}))
        c = summarise(table(recs, exp, key).get(("v-pred/v+cons",), {}))["mean"]
        x = summarise(table(recs, exp, key).get(("x-pred/v",), {}))["mean"]
        v = vs["mean"]
        if not all(math.isfinite(z) for z in (v, c, x)):
            continue
        pair = paired(
            table(recs, exp, key).get(("x-pred/v",), {}),
            table(recs, exp, key).get(("v-pred/v",), {}),
        )
        resolved = (
            pair.get("n", 0) >= 2
            and abs(pair["diff"]) > 3 * np.std(pair["diffs"], ddof=1)
            / math.sqrt(len(pair["diffs"]))
        )
        share = f"{100 * (c - v) / (x - v):+.0f}%" if resolved else "--"
        L.append(f"| {label} | {fmt(v)} | {fmt(c)} | {fmt(x)} | {share} |")

    L.append("\n### The dual-head channel: 2 x 2 interaction\n")
    L.append(
        "Closing stream visibility leaves both heads reading the shared condition "
        "but stops them reading each other. The question is not whether the channel "
        "helps -- it does for both -- but whether one parameterization makes more of "
        "it, which is the interaction term.\n"
    )
    L.append("| metric | x-pred: open -> closed | v-pred: open -> closed | interaction | 95% CI | p |")
    L.append("|---|---:|---:|---:|:--:|---:|")
    for key, label, _ in (
        ("nfe5_coord_violation", "coordination error, 5 steps", ""),
        ("nfe5_mode_mismatch", "heads disagree", ""),
        ("det_risk_model", "risk, mode-carrying block", ""),
        ("mode_acc_model", "mode accuracy, manip head", ""),
        ("xstream_utilization_det", "cross-stream gain captured", ""),
    ):
        r = interaction(
            recs, exp, key,
            {"a0": "x-pred/v", "a1": "x-pred/v-nojoint",
             "b0": "v-pred/v", "b1": "v-pred/v-nojoint"},
        )
        if r.get("n", 0) < 2:
            continue
        L.append(
            f"| {label} | {r['effect_a']:+.4f} | {r['effect_b']:+.4f} | "
            f"{r['interaction']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{r['p_wilcoxon']:.4f} |"
        )
    return L


if __name__ == "__main__":
    main()
