"""The experimental conditions, and why each one is in the design.

The comparison that matters is ``x-pred`` against ``v-pred`` under the velocity
weighting, because those two are *the same loss function* (see ``flow.py``) and
differ only in what the output head emits.  Everything else in this list is here
to close a specific alternative explanation.

``eps-pred/v``
    The other skip-connection parameterization.  If the effect were about
    velocity specifically rather than about routing noise to the output, eps-pred
    would not pattern with v-pred.

``x-pred/x`` and ``v-pred/x``
    The loss-weighting axis, crossed with the parameterization axis.  Without
    this 3 x 2 grid, "x-pred wins" is confounded with "the 1/(1-t)^2 weight
    happens to suit x-pred".

``v-pred/v + consistency``
    v-pred plus an explicit penalty for disagreeing about the endpoint across two
    independent noise draws of the same clean chunk.  This buys, at the cost of a
    second forward pass and a hand-set coefficient, the noise-invariance that
    x-pred gets from its parameterization.  If it closes most of the gap, the
    mechanism proposed for the effect is the right one.  If it closes none, the
    proposed mechanism is wrong.

``x-pred/v, no joint attn`` and ``v-pred/v, no joint attn``
    Stream visibility off, so the manip and aux tokens cannot read each other and
    both heads see only the shared condition.  Run for both parameterizations so
    the quantity of interest is the *interaction*: how much each
    parameterization is able to make of the channel, not merely whether the
    channel helps on average.
"""

from __future__ import annotations

from .config import ArmConfig

# A single vivid hue per family so every figure is colour-consistent.
PALETTE = {
    "x": "#FF2E93",        # magenta   - clean-sample head
    "v": "#00C2FF",        # cyan      - velocity head
    "eps": "#7B5CFF",      # violet    - noise head
    "x_nj": "#FF9D00",     # tangerine - x-pred, channel closed
    "v_nj": "#00E5A0",     # mint      - v-pred, channel closed
    "v_cons": "#0059B3",   # deep blue - v-pred + explicit consistency
    "x_mode": "#C2185B",   # deep magenta - x-pred + discrete mode latent
    "v_mode": "#0288A8",   # deep cyan    - v-pred + discrete mode latent
    "x_agree": "#FFC400",  # amber        - x-pred + mode agreement supervision
    "bayes": "#111111",
    "bayes_local": "#8A8A8A",
    "bayes_oracle": "#4A4A4A",
}


def main_arms() -> list[ArmConfig]:
    return [
        ArmConfig(
            name="x-pred/v", parameterization="x", loss_space="v",
            label=r"$x$-pred (VELA-0)", color=PALETTE["x"],
        ),
        ArmConfig(
            name="v-pred/v", parameterization="v", loss_space="v",
            label=r"$v$-pred (flow matching)", color=PALETTE["v"],
        ),
        ArmConfig(
            name="eps-pred/v", parameterization="eps", loss_space="v",
            label=r"$\epsilon$-pred", color=PALETTE["eps"],
        ),
        ArmConfig(
            name="x-pred/x", parameterization="x", loss_space="x",
            label=r"$x$-pred, unweighted", color=PALETTE["x"],
        ),
        ArmConfig(
            name="v-pred/x", parameterization="v", loss_space="x",
            label=r"$v$-pred, unweighted", color=PALETTE["v"],
        ),
        ArmConfig(
            name="eps-pred/x", parameterization="eps", loss_space="x",
            label=r"$\epsilon$-pred, unweighted", color=PALETTE["eps"],
        ),
        ArmConfig(
            name="v-pred/v+cons", parameterization="v", loss_space="v",
            consistency_weight=1.0,
            label=r"$v$-pred + explicit consistency", color=PALETTE["v_cons"],
        ),
        ArmConfig(
            name="x-pred/v-nojoint", parameterization="x", loss_space="v",
            joint_attention=False,
            label=r"$x$-pred, channel closed", color=PALETTE["x_nj"],
        ),
        ArmConfig(
            name="v-pred/v-nojoint", parameterization="v", loss_space="v",
            joint_attention=False,
            label=r"$v$-pred, channel closed", color=PALETTE["v_nj"],
        ),
    ]


HEADLINE = ("x-pred/v", "v-pred/v")
FACTORIAL = ("x-pred/v", "v-pred/v", "eps-pred/v", "x-pred/x", "v-pred/x", "eps-pred/x")
INTERACTION = (
    "x-pred/v", "v-pred/v", "x-pred/v-nojoint", "v-pred/v-nojoint",
)


def width_arms() -> list[ArmConfig]:
    """Three parameterizations at a fixed loss, over the width of the stream.

    The widest point is chosen to match VELA-0's own ratio of expert width to
    per-token action width (1024 / 58), so the sweep says something about the
    deployed configuration rather than only about a starved one.
    """
    out = []
    for w in (16, 24, 32, 48, 64, 96, 128, 256, 512, 1024):
        for base in main_arms()[:3]:
            arm = ArmConfig(**{**base.__dict__, "width": w})
            arm.name = f"{base.name}@w{w}"
            out.append(arm)
    return out


def token_group_arms() -> list[tuple[ArmConfig, int]]:
    """Three parameterizations against the per-token action width.

    VELA-0 spends one token per chunk step, so a token carries 58 numbers against
    a 1024-wide stream -- comfortable.  Grouping ``g`` steps into one token
    carries ``58g`` numbers instead, at the same total supervision and roughly the
    same FLOPs on the target.  This is the axis on which the high-dimensional
    diffusion literature reports the skip parameterizations failing, and it is a
    live architectural choice for an action expert rather than a synthetic knob.
    """
    out = []
    for g in (1, 2, 4, 8, 16):
        for base in main_arms()[:3]:
            arm = ArmConfig(**base.__dict__)
            arm.name = f"{base.name}@g{g}"
            out.append((arm, g))
    return out


def schedule_arms() -> list[tuple[ArmConfig, tuple[float, float], str]]:
    """Does VELA-0's timestep distribution suit the head it is paired with?

    ``Beta(1.5, 1.0)`` biases sampling toward clean inputs, and the ``1/(1-t)^2``
    weight then concentrates the gradient near ``t = 1``.  Near ``t = 1`` a skip
    parameterization is nearly free -- ``x_hat = z_t + sig*net`` with ``sig -> 0``
    is already almost correct -- so the schedule spends most of the budget where
    the parameterizations cannot be told apart, and little where they differ.
    This sweep asks whether moving budget toward high noise changes the ranking.
    """
    out = []
    for beta, tag in (((1.5, 1.0), "vela"), ((1.0, 1.0), "uniform"), ((1.0, 1.5), "noisy")):
        for base in main_arms()[:2]:
            arm = ArmConfig(**base.__dict__)
            arm.name = f"{base.name}@{tag}"
            out.append((arm, beta, tag))
    return out


def fix_arms() -> list[ArmConfig]:
    """Can the mode-commitment cost be paid off without giving up the mechanism?

    The one consistent cost of the clean-sample head in this toy is decisiveness:
    with a genuinely bimodal target, the noise draw is the only thing that can
    break the tie between the two valid modes, so a head that follows the draw is
    handed a tie-break and commits, while a head trained to ignore the draw is
    pushed toward the conditional mean, which on a bimodal target is the illegal
    midpoint.  The invariance and the indecision have the same cause, so the
    question is whether the tie-break can be supplied from somewhere else.

    ``+mode``
        A shared discrete coordination-mode token, observed in training and drawn
        from its (uniform, known) prior at inference.  This relocates the symmetry
        breaking out of the noise and into an explicit latent, and shares it, so
        the two heads cannot commit to different modes.  Run for *both*
        parameterizations, because handing over the mode makes the problem easier
        for either head and only the difference between them is evidence.

    ``+agree``
        Per-stream auxiliary supervision of the same shared mode label, with no
        latent to condition on.  This separates "both heads represent the mode"
        from "both heads act on one draw of it".  If the first is sufficient, the
        cheaper intervention is enough; if only ``+mode`` works, the mechanism is
        the missing tie-break rather than a missing representation.
    """
    x, v = main_arms()[0], main_arms()[1]
    return [
        x,
        v,
        ArmConfig(
            name="x-pred/v+mode", parameterization="x", loss_space="v",
            mode_conditioning=True,
            label=r"$x$-pred + mode latent", color=PALETTE["x_mode"],
        ),
        ArmConfig(
            name="v-pred/v+mode", parameterization="v", loss_space="v",
            mode_conditioning=True,
            label=r"$v$-pred + mode latent", color=PALETTE["v_mode"],
        ),
        ArmConfig(
            name="x-pred/v+agree", parameterization="x", loss_space="v",
            mode_agree_weight=0.3,
            label=r"$x$-pred + mode supervision", color=PALETTE["x_agree"],
        ),
    ]


FIX = ("x-pred/v", "v-pred/v", "x-pred/v+mode", "v-pred/v+mode", "x-pred/v+agree")


def modewidth_arms() -> list[tuple[ArmConfig, int]]:
    """``(arm, group)``: the width sweep re-run with the coordination mode given.

    This is the comparison the deployed setting actually corresponds to, and
    running it is what makes the ``main`` result interpretable rather than
    disappointing.

    In ``main`` and ``tokengroup`` the coordination mode is left ambiguous: two
    plans are equally valid and nothing in the observation says which. Under
    squared error the optimal point estimate is then the midpoint of the two
    modes, and the *only* symmetry-breaking signal available to the network is the
    flow noise -- which is exactly the signal a clean-sample head is trained to
    discard. So ``main`` charges the clean-sample head for a property that is
    supposed to be its advantage, and the 58-dimensional result is a wash for that
    reason rather than because the mechanism is absent.

    A real action expert is not in that position. It conditions on language, a
    goal, or a task token, so the coordination mode is resolved by conditioning
    rather than left to the sampler. ``fix`` establishes at one width that
    supplying the mode removes the confound; this sweeps it, so the claim can be
    stated at VELA-0's own per-token width instead of extrapolated to it.

    All three parameterizations are included: ``eps``-pred is the limiting case of
    the skip convention and bounds the trend from the other side.
    """
    base_specs = (
        ("x", "x_mode", r"$x$-pred"),
        ("v", "v_mode", r"$v$-pred"),
        ("eps", "eps", r"$\epsilon$-pred"),
    )
    out = []
    for g in (1, 2, 4, 8, 16):
        for param, colour, label in base_specs:
            out.append((
                ArmConfig(
                    name=f"{param}-pred/v+mode@g{g}",
                    parameterization=param, loss_space="v",
                    mode_conditioning=True,
                    label=label, color=PALETTE[colour],
                ),
                g,
            ))
    return out


def seqlen_arms() -> list[tuple[ArmConfig, int, int]]:
    """``(arm, chunk, group)``: is the ``main_hd`` effect about width or length?

    Grouping four chunk steps into one token raises the per-token action width
    from 58 to 232, but it also shortens the sequence from 16 tokens to 4.  Those
    are confounded in ``main_hd``.  Three configurations separate them:

    ================  ======  =======  ===================================
    label             tokens  per-tok  role
    ================  ======  =======  ===================================
    ``T16g1``             16       58  the reference, = ``main``
    ``T4g1``               4       58  four tokens, narrow: length only
    ``T4g4``               4      232  four tokens, wide: width and length
    ================  ======  =======  ===================================

    If the clean-sample head's advantage shows up in ``T4g4`` but not in ``T4g1``,
    the shorter sequence is not what produces it.  ``T4g1`` also carries a quarter
    of the supervision, so it is not comparable to the others in absolute risk --
    only the ratio between parameterizations within a configuration is.
    """
    out = []
    for chunk, group, tag in ((16, 1, "T16g1"), (4, 1, "T4g1"), (16, 4, "T4g4")):
        for base in main_arms()[:2]:
            arm = ArmConfig(**base.__dict__)
            arm.name = f"{base.name}@{tag}"
            out.append((arm, chunk, group))
    return out


def intrinsic_arms() -> list[tuple[ArmConfig, int]]:
    """The same two headline arms as the manifold is fattened toward the ambient
    space.  The manifold assumption is an assumption; this is where it is allowed
    to fail, and the advantage should shrink with it."""
    out = []
    for style in (2, 4, 8, 16, 32, 64):
        for base in main_arms()[:2]:
            arm = ArmConfig(**base.__dict__)
            arm.name = f"{base.name}@style{style}"
            out.append((arm, style))
    return out
