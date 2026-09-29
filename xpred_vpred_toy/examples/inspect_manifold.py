#!/usr/bin/env python
"""Print one batch of the synthetic manifold, so the data contract is visible.

    python examples/inspect_manifold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from velatoy.config import DataConfig
from velatoy.data import VelaToyManifold


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = VelaToyManifold(DataConfig(), seed=1234, device=device)
    b = man.sample(8)
    print("generative contract")
    print(f"  chunk T                 = {man.cfg.chunk}")
    print(f"  manip / aux dims        = {man.manip_dim} / {man.aux_dim}")
    print(f"  latent dim (designed)   = {man.latent_dim}")
    print(f"  n embodiments           = {man.cfg.n_embodiments}")
    print(f"  coordination modes      = {man.cfg.alpha_low}, {man.cfg.alpha_high}")
    print()
    print("one batch of 8")
    print(f"  cond                    {tuple(b.cond.shape)}")
    print(f"  x_manip / x_aux         {tuple(b.x_manip.shape)} / {tuple(b.x_aux.shape)}")
    print(f"  mask_manip mean         {float(b.mask_manip.mean()):.3f}")
    print(f"  aux_active              {b.aux_active.tolist()}")
    print(f"  mode (0=base, 1=arm)    {b.mode.tolist()}")
    print(f"  alpha                   {[round(x, 2) for x in b.alpha.tolist()]}")
    print(f"  embod                   {b.embod.tolist()}")
    print()
    a_m = b.alpha
    a_a = 1.0 - b.alpha
    print("coordination: alpha_manip + alpha_aux = 1 by construction")
    print(f"  max |a_m + a_a - 1|     {float((a_m + a_a - 1).abs().max()):.2e}")


if __name__ == "__main__":
    main()
