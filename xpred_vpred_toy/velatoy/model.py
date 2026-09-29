"""A miniature of the VELA-0 joint action expert.

Faithful to the parts that matter for this study:

* two action streams (``manip`` 58-dim, ``aux`` 22-dim) with their own encoders,
  their own per-block QKV and feed-forward, and their own decoders
* one shared scaled-dot-product attention over the concatenation of both
  streams, so the heads coordinate through attention rather than through a
  bespoke fusion module
* the condition enters as keys and values only: it gets no query, no output
  projection and no feed-forward
* near tokens cannot read far tokens
* a sample whose aux slice is unlabelled has its aux tokens removed from the
  attention, not merely masked in the loss

And, critically, **there is no skip connection from the noisy input to the
output**.  Every path from ``z_t`` to the prediction passes through the
width-``d`` residual stream.  This is what makes the choice of output
parameterization consequential: a velocity or noise head has to reconstruct and
transmit its own input noise through that stream, while a clean-sample head does
not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

STREAMS = ("manip", "aux")


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = t.float()[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class JointBlock(nn.Module):
    """Per-stream projections and feed-forward, one shared attention."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.width
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        self.dropout = cfg.dropout
        self.norm1 = nn.ModuleDict({s: nn.LayerNorm(d) for s in STREAMS})
        self.qkv = nn.ModuleDict(
            {s: nn.Linear(d, 3 * d, bias=False) for s in STREAMS}
        )
        self.proj = nn.ModuleDict({s: nn.Linear(d, d) for s in STREAMS})
        self.norm2 = nn.ModuleDict({s: nn.LayerNorm(d) for s in STREAMS})
        self.ffn = nn.ModuleDict(
            {
                s: nn.Sequential(
                    nn.Linear(d, cfg.ffn_mult * d),
                    nn.GELU(),
                    nn.Linear(cfg.ffn_mult * d, d),
                )
                for s in STREAMS
            }
        )
        # read-only context: keys and values, no query, no FFN
        self.ctx_norm = nn.LayerNorm(d)
        self.ctx_kv = nn.Linear(d, 2 * d, bias=False)
        # AdaLN-zero gating on the flow time.  One low-rank projection shared by
        # the streams: the timestep is a global property of the batch element, and
        # giving each stream its own full-rank modulation was a quarter of the
        # parameter count for no measurable difference.
        rank = max(16, d // 8)
        self.mod_down = nn.Linear(d, rank)
        self.mod_up = nn.ModuleDict({s: nn.Linear(rank, 4 * d) for s in STREAMS})
        for s in STREAMS:
            nn.init.zeros_(self.mod_up[s].weight)
            nn.init.zeros_(self.mod_up[s].bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        context: torch.Tensor,
        time_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mods = {}
        tm = F.silu(self.mod_down(time_emb))
        for s in STREAMS:
            mods[s] = self.mod_up[s](tm)[:, None, :].chunk(4, dim=-1)

        qs, ks, vs, sizes = [], [], [], {}
        for s in STREAMS:
            h = self.norm1[s](tokens[s])
            scale, shift, _, _ = mods[s]
            h = h * (1.0 + scale) + shift
            q, k, v = self.qkv[s](h).chunk(3, dim=-1)
            qs.append(self._split(q))
            ks.append(self._split(k))
            vs.append(self._split(v))
            sizes[s] = tokens[s].shape[1]

        ck, cv = self.ctx_kv(self.ctx_norm(context)).chunk(2, dim=-1)
        q_all = torch.cat(qs, dim=2)
        k_all = torch.cat(ks + [self._split(ck)], dim=2)
        v_all = torch.cat(vs + [self._split(cv)], dim=2)

        attn = F.scaled_dot_product_attention(
            q_all, k_all, v_all, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        B = q_all.shape[0]
        attn = attn.transpose(1, 2).reshape(B, -1, self.n_heads * self.head_dim)

        out, off = {}, 0
        for s in STREAMS:
            n = sizes[s]
            _, _, gate_a, _ = mods[s]
            h = tokens[s] + (1.0 + gate_a) * self.proj[s](attn[:, off : off + n])
            _, _, _, gate_f = mods[s]
            out[s] = h + (1.0 + gate_f) * self.ffn[s](self.norm2[s](h))
            off += n
        return out


@dataclass
class ExpertOutput:
    manip: torch.Tensor                 # (B, T, 58)
    aux: torch.Tensor                   # (B, T, 22)
    hidden: dict[str, list[torch.Tensor]]   # stream -> per-block (B, 1+T, d)
    mode_logits: dict[str, torch.Tensor] | None = None   # stream -> (B, 2)


class ToyJointExpert(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        chunk: int,
        near: int,
        manip_dim: int,
        aux_dim: int,
        cond_dim: int,
        n_embodiments: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.chunk = chunk
        self.near = near
        self.dims = {"manip": manip_dim, "aux": aux_dim}
        # How many chunk steps share one token.  VELA-0 uses one token per step,
        # so the per-token action width is 58; grouping k steps together makes it
        # 58k, which is the axis along which the high-dimensional-diffusion
        # literature reports the skip parameterizations breaking down.  It is a
        # real design choice for an action expert, not a synthetic knob: it trades
        # sequence length for per-token width at fixed FLOPs on the target.
        self.group = cfg.token_group
        if chunk % self.group:
            raise ValueError(f"chunk {chunk} not divisible by group {self.group}")
        self.n_tokens = chunk // self.group
        self.near_tokens = near // self.group
        self.tok_dims = {s: self.dims[s] * self.group for s in STREAMS}

        self.action_in = nn.ModuleDict(
            {
                s: nn.Sequential(
                    nn.Linear(self.tok_dims[s], d), nn.SiLU(), nn.Linear(d, d)
                )
                for s in STREAMS
            }
        )
        # the state token carries the validity mask, as in VELA-0's StateEncoder
        self.state_in = nn.ModuleDict(
            {s: nn.Linear(self.dims[s], d) for s in STREAMS}
        )
        self.pos = nn.ParameterDict(
            {s: nn.Parameter(0.02 * torch.randn(1, self.n_tokens, d)) for s in STREAMS}
        )
        self.segment = nn.ParameterDict(
            {s: nn.Parameter(0.02 * torch.randn(2, d)) for s in STREAMS}
        )
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_in = nn.Linear(cond_dim, cfg.n_context_tokens * d)
        self.embod_emb = nn.Embedding(n_embodiments, d)
        # One shared context token for the discrete coordination mode.  Read-only,
        # like the condition, and shared, so the two streams cannot commit to
        # different modes: whatever tie-break it carries, both of them see it.
        self.n_ctx = cfg.n_context_tokens + 1 + (1 if cfg.mode_conditioning else 0)
        self.mode_emb = nn.Embedding(2, d) if cfg.mode_conditioning else None
        self.mode_out = (
            nn.ModuleDict({s: nn.Linear(d, 2) for s in STREAMS})
            if cfg.mode_classifier
            else None
        )
        self.ctx_pos = nn.Parameter(0.02 * torch.randn(1, self.n_ctx, d))

        self.blocks = nn.ModuleList([JointBlock(cfg) for _ in range(cfg.n_blocks)])
        self.decoders = nn.ModuleDict(
            {
                s: nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 2 * d),
                    nn.GELU(),
                    nn.Linear(2 * d, self.tok_dims[s]),
                )
                for s in STREAMS
            }
        )
        # Deliberately *not* zero-initialised.  Zero-initialising the last decoder
        # layer is the DiT convention, but it is not neutral here: it makes a
        # velocity head start at exactly x_hat = z_t, an excellent warm start,
        # while making a clean-sample head start at x_hat = 0, a degenerate one.
        # A small symmetric init gives both heads the same starting scale.
        for s in STREAMS:
            nn.init.normal_(self.decoders[s][-1].weight, std=0.02)
            nn.init.zeros_(self.decoders[s][-1].bias)

        self.register_buffer("_base_mask", self._build_mask(), persistent=False)

    # ------------------------------------------------------------------ masks

    def _build_mask(self) -> torch.Tensor:
        """(S, S) bool: True where a query may read a key.

        Composes stream visibility with the near/far rule, exactly as
        ``vela/models/attention.py`` does.  Context keys are always readable.
        """
        n = 1 + self.n_tokens
        total = 2 * n
        n_ctx = self.n_ctx

        stream_id = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
        seg = torch.zeros(total, dtype=torch.long)   # 0 = near, 1 = far
        for i, base in enumerate((0, n)):
            seg[base] = 0                            # state token is near
            for step in range(self.n_tokens):
                seg[base + 1 + step] = 0 if step < self.near_tokens else 1

        visible = stream_id[:, None] == stream_id[None, :]
        if self.cfg.joint_attention:
            visible = torch.ones(total, total, dtype=torch.bool)
        mask = visible.clone()
        if self.cfg.near_far_mask:
            near_q = (seg == 0)[:, None]
            far_k = (seg == 1)[None, :]
            mask &= ~(near_q & far_k)
        return torch.cat([mask, torch.ones(total, n_ctx, dtype=torch.bool)], dim=1)

    def attention_mask(self, aux_active: torch.Tensor) -> torch.Tensor:
        """Per-sample mask, pruning the aux stream where it does not exist."""
        B = aux_active.shape[0]
        n = 1 + self.n_tokens
        m = self._base_mask[None].expand(B, -1, -1).clone()
        dead = aux_active <= 0.0                      # (B,)
        if bool(dead.any()):
            idx = torch.arange(m.shape[1], device=m.device)
            is_aux = idx >= n
            kill = dead[:, None] & is_aux[None, :]    # (B, Q)
            m &= ~kill[:, :, None]
            kill_k = torch.cat(
                [kill, torch.zeros(B, self.n_ctx, dtype=torch.bool, device=m.device)],
                dim=1,
            )
            m &= ~kill_k[:, None, :]
            # a fully masked row would produce NaN; let dead rows read themselves
            diag = torch.zeros_like(m)
            ar = torch.arange(m.shape[1], device=m.device)
            diag[:, ar, ar] = True
            m |= diag & kill[:, :, None]
        return m[:, None]                             # broadcast over heads

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        z_manip: torch.Tensor,
        z_aux: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        embod: torch.Tensor,
        mask_manip: torch.Tensor,
        mask_aux: torch.Tensor,
        aux_active: torch.Tensor,
        *,
        mode: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> ExpertOutput:
        B = cond.shape[0]
        d = self.cfg.width
        time_emb = self.time_mlp(timestep_embedding(t.view(B), d))

        ctx = self.cond_in(cond).view(B, self.cfg.n_context_tokens, d)
        parts = [ctx, self.embod_emb(embod)[:, None]]
        if self.mode_emb is not None:
            if mode is None:
                raise ValueError("mode_conditioning is on but no mode was supplied")
            parts.append(self.mode_emb(mode.view(B))[:, None])
        ctx = torch.cat(parts, dim=1) + self.ctx_pos

        z = {"manip": z_manip, "aux": z_aux}
        msk = {"manip": mask_manip, "aux": mask_aux}
        tokens = {}
        for s in STREAMS:
            folded = z[s].reshape(B, self.n_tokens, self.tok_dims[s])
            act = self.action_in[s](folded) + self.pos[s]
            near_tok = (
                torch.arange(self.n_tokens, device=z[s].device)[None, :, None]
                < self.near_tokens
            )
            act = act + torch.where(
                near_tok, self.segment[s][0][None, None, :],
                self.segment[s][1][None, None, :],
            )
            state = self.state_in[s](msk[s][:, 0])[:, None, :]
            tokens[s] = torch.cat([state, act], dim=1)

        attn_mask = self.attention_mask(aux_active)
        hidden: dict[str, list[torch.Tensor]] = {s: [] for s in STREAMS}
        for blk in self.blocks:
            tokens = blk(tokens, ctx, time_emb, attn_mask)
            if return_hidden:
                for s in STREAMS:
                    hidden[s].append(tokens[s])

        logits = None
        if self.mode_out is not None:
            logits = {
                s: self.mode_out[s](tokens[s][:, 1:].mean(1)) for s in STREAMS
            }

        return ExpertOutput(
            manip=self.decoders["manip"](tokens["manip"][:, 1:]).reshape(
                B, self.chunk, self.dims["manip"]
            ),
            aux=self.decoders["aux"](tokens["aux"][:, 1:]).reshape(
                B, self.chunk, self.dims["aux"]
            ),
            hidden=hidden,
            mode_logits=logits,
        )
