"""Configuration for the VELA-toy x-pred / v-pred study.

The toy mirrors the VELA-0 action expert:

* canonical action width 80, split into ``manip`` [0, 58) and ``aux`` [58, 80)
* an action chunk of ``chunk`` steps with a near / far split
* two action streams in one joint-attention transformer, one token per chunk
  step plus a leading state token per stream
* rectified flow ``z_t = (1 - t) eps + t x`` with Beta(1.5, 1.0) time sampling,
  a velocity-space loss weight ``1 / (1 - t)^2`` floored by ``sigma_min``, and
  Euler sampling from t = 0 to t = 1

The only thing that varies between the arms of the study is what the network's
output head is interpreted as (``x``, ``v`` or ``eps``) and, on a separate axis,
whether the squared error is weighted into velocity space or left in sample
space.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# VELA-0 canonical-80 layout, verbatim.
MANIP_SLICE = (0, 58)
AUX_SLICE = (58, 80)
CANONICAL_DIM = 80


@dataclass
class DataConfig:
    """The synthetic cross-embodiment action manifold."""

    chunk: int = 16
    """Chunk horizon T. VELA-0 uses 64; the toy keeps the ratio to near_steps."""

    near: int = 8
    """Near segment length. Near tokens cannot read far tokens."""

    cond_dim: int = 12
    """Width of the 'VLM readout' condition vector c."""

    task_dim: int = 8
    """Dimension of the shared task latent tau. Drives both streams."""

    style_dim: int = 4
    """Per-stream private latent. Not recoverable from the partner stream."""

    temporal_rank: int = 4
    """Rank of the smooth temporal basis the chunk is built from."""

    n_embodiments: int = 6
    """Number of embodiments. Each has its own action-manifold embedding."""

    n_aux_inactive: int = 2
    """Embodiments with no whole-body labels; their aux stream is removed."""

    alpha_low: float = 0.35
    alpha_high: float = 0.65
    """Two valid coordination modes: base-dominant and arm-dominant."""

    mode_stochastic: bool = True
    """If True the same condition admits both coordination modes (multimodal)."""

    witness_manip: float = 0.12
    witness_aux: float = 1.60
    """Amplitude of a mode-indicating latent direction in each stream.

    Deliberately asymmetric.  Whole-body motion is an unambiguous witness of the
    coordination mode -- the base either drives in or it does not -- whereas from
    the arm alone the two modes look similar at high noise.  So the manipulation
    head has something real to gain by reading the whole-body stream, and the
    closed-form denoiser quantifies exactly how much.
    """

    hand_invalid_prob: float = 0.5
    """Fraction of embodiments whose 12-dim hand block is unlabelled."""

    variance_normalized: bool = True
    """Scale the clean action so per-channel variance matches the unit noise."""


@dataclass
class ModelConfig:
    """A miniature of ``VelaJointExpert``."""

    width: int = 512
    n_blocks: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    joint_attention: bool = True
    """If False, manip and aux tokens cannot attend to each other; both still
    read the shared condition context, so this ablates the direct channel only."""
    near_far_mask: bool = True
    n_context_tokens: int = 8

    token_group: int = 1
    """Chunk steps per action token.  1 reproduces VELA-0; larger values raise the
    per-token action width to ``58 * group`` at a fixed target, which is the
    regime where predicting a noised quantity is reported to fail."""

    mode_conditioning: bool = False
    """Give both streams a shared discrete coordination-mode token.

    The point of this switch is to relocate the symmetry breaking.  With a
    genuinely bimodal target and no discrete latent, the *only* thing that can
    break the tie between the two valid modes is the noise draw, so a head whose
    endpoint estimate follows the noise draw is handed a tie-break and commits,
    while a head trained to return the same endpoint regardless of the draw is
    pushed toward the conditional mean -- the illegal midpoint.  Sampling the mode
    from its (known, uniform) prior and conditioning *both* streams on the same
    draw supplies the tie-break explicitly, so the invariance no longer costs
    decisiveness.  It enters as a read-only context token, the same channel the
    condition uses, and it is the toy's version of a discrete flag, of which the
    real aux stream already carries several.
    """

    mode_classifier: bool = False
    """Per-stream auxiliary head predicting the shared mode from pooled tokens.

    A weaker intervention than ``mode_conditioning``: it shapes the residual
    stream to represent the shared discrete decision, without giving the sampler
    anything to commit to.  Included to separate "both heads know the mode" from
    "both heads act on the same mode".
    """


@dataclass
class FlowConfig:
    """Rectified flow, matching ``vela/models/flow.py``."""

    time_beta: tuple[float, float] = (1.5, 1.0)
    sigma_min: float = 0.05
    normalize_weight: bool = True
    num_inference_steps: int = 5


@dataclass
class ArmConfig:
    """One experimental condition."""

    name: str
    parameterization: str = "x"
    """``x``, ``v`` or ``eps``: what the output head is taken to be."""

    loss_space: str = "v"
    """``v`` applies the 1/(1-t)^2 velocity weight; ``x`` leaves weight 1."""

    joint_attention: bool = True
    consistency_weight: float = 0.0
    """Explicit two-draw endpoint-agreement penalty. A control for the claim
    that x-pred obtains this regularization for free."""

    mode_conditioning: bool = False
    """Shared discrete coordination-mode token, sampled from its prior at
    inference.  Run for both parameterizations, because giving the mode away
    makes the task easier for either head and only the *difference* is evidence."""

    mode_agree_weight: float = 0.0
    """Weight of the per-stream mode cross-entropy when ``mode_classifier`` is on."""

    width: int | None = None
    label: str = ""
    color: str = "#000000"


@dataclass
class TrainConfig:
    steps: int = 4000
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 200
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    probe_every: int = 250
    probe_batch: int = 2048
    seed: int = 0


@dataclass
class EvalConfig:
    n_samples: int = 4096
    nfe_list: tuple[int, ...] = (1, 2, 4, 5, 8, 16)
    nfe_main: int = 5
    # Denser below t = 0.1 than elsewhere, and deliberately so: the headline result
    # is that the two conventions differ at high noise and not in aggregate, and the
    # high-noise end is exactly where a Beta(1.5, 1) time distribution puts almost
    # no mass.  A grid that is uniform in t would rest that claim on two points.
    t_grid: tuple[float, ...] = (
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9,
    )
    n_noise_draws: int = 24
    """Draws per clean sample for the noise-invariance measurement."""
    probe_ridge: float = 1e-3
    sw_projections: int = 512


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    arm: ArmConfig = field(default_factory=lambda: ArmConfig(name="x-pred"))
    experiment: str = "main"
    device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
