#!/usr/bin/env python
"""A short, paired x-pred vs v-pred run that prints the whole story.

This is the teaching entry point.  It does three things, in the same order as
the paper figure:

  (a)  SVD of clean actions / velocity / noise, with no network trained
  (b)  train two paired arms (same seed, data, init, noise, times)
  (c)  print representation, high-noise risk, and few-step generation metrics

Default is a *smoke* budget so it finishes on one GPU in a few minutes.  Pass
``--full`` to use the paper's width and a longer budget (still one seed).

    python scripts/demo.py
    python scripts/demo.py --full --steps 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def spectrum_table(n: int = 4096) -> dict:
    import torch

    from velatoy.config import DataConfig
    from velatoy.data import VelaToyManifold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = VelaToyManifold(DataConfig(), seed=1234, device=device)
    torch.manual_seed(0)
    b = man.sample(n)
    keep = b.embod == 1
    x = b.x_manip[keep].flatten(1)
    eps = torch.randn_like(x)
    v = x - eps
    floor = 1e-5
    out = {"n_rows": int(x.shape[0]), "n_cols": int(x.shape[1]),
           "latent_dim": man.latent_dim}
    for name, mat in (("x", x), ("v", v), ("eps", eps)):
        m = mat - mat.mean(0, keepdim=True)
        s = torch.linalg.svdvals(m.float().cpu())
        s = (s / s[0]).clamp_min(floor)
        # first index whose energy is numerically gone
        rank = int((s > 1e-3).sum().item())
        out[name] = {
            "s13": float(s[12]) if s.numel() >= 13 else float("nan"),
            "s14": float(s[13]) if s.numel() >= 14 else float("nan"),
            "effective_rank_1e-3": rank,
        }
    return out


def train_pair(args) -> dict[str, dict]:
    import torch

    from velatoy.config import (
        ArmConfig, DataConfig, EvalConfig, FlowConfig, ModelConfig,
        RunConfig, TrainConfig,
    )
    from velatoy.train import run

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    width = args.width
    records = {}
    for param in ("x", "v"):
        arm = ArmConfig(
            name=f"{param}-pred/v+mode",
            parameterization=param,
            loss_space="v",
            mode_conditioning=True,
            width=width,
            label=f"{param}-pred",
        )
        cfg = RunConfig(
            data=DataConfig(),
            model=ModelConfig(
                width=width, token_group=1, mode_conditioning=True,
                n_blocks=args.blocks,
            ),
            flow=FlowConfig(),
            train=TrainConfig(
                steps=args.steps, seed=args.seed, batch=args.batch,
                probe_every=max(50, args.steps // 8),
            ),
            eval=EvalConfig(n_samples=args.eval_samples),
            arm=arm,
            experiment="demo",
            device=device,
        )
        print(f"\n=== training {arm.name}  width={width}  steps={args.steps}  "
              f"device={device} ===", flush=True)
        res = run(cfg, verbose=args.verbose)
        records[param] = res.final
    return records


def show(title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    w = max(len(r[0]) for r in rows)
    print(f"{'metric':<{w}}  {'x-pred':>10}  {'v-pred':>10}")
    for name, a, b in rows:
        print(f"{name:<{w}}  {a:>10}  {b:>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--eval-samples", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--full", action="store_true",
                    help="paper width 512 / 4 blocks, longer budget")
    ap.add_argument("--out", default="results/demo.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.full:
        args.steps = args.steps or 4000
        args.width = args.width or 512
        args.blocks = args.blocks or 4
        args.batch = args.batch or 256
        args.eval_samples = args.eval_samples or 2048
    else:
        args.steps = args.steps or 600
        args.width = args.width or 128
        args.blocks = args.blocks or 2
        args.batch = args.batch or 128
        args.eval_samples = args.eval_samples or 512

    print("x-pred vs v-pred teaching demo")
    print("paired: same seed, same data, same init, same noise, same times")
    print("only the prediction target of the output head changes\n")

    spec = spectrum_table()
    print("(a) data geometry  —  no network yet")
    print(f"    embodiment-1 clean chunks: {spec['n_rows']} x {spec['n_cols']}")
    print(f"    designed latent dim = {spec['latent_dim']}")
    print(f"    clean x   effective rank (σ>1e-3) = {spec['x']['effective_rank_1e-3']}"
          f"   σ13={spec['x']['s13']:.2e}  σ14={spec['x']['s14']:.2e}")
    print(f"    velocity  effective rank          = {spec['v']['effective_rank_1e-3']}")
    print(f"    noise ε   effective rank          = {spec['eps']['effective_rank_1e-3']}")

    rec = train_pair(args)
    keys = [
        ("jac_participation_ratio", "Jacobian participation ratio  ↓"),
        ("jac_rank99", "Jacobian 99% energy rank      ↓"),
        ("probe_eps_manip", "own-noise R²                 ↓"),
        ("probe_partner_manip", "partner-action R²            ↑"),
        ("excess_risk_oracle", "excess denoising risk        ↓"),
        ("nfe1_off_manifold", "off-manifold mass, 1 step    ↓"),
        ("nfe5_off_manifold", "off-manifold mass, 5 steps   ↓"),
        ("gen_invalid_leakage", "invalid-DoF leakage          ↓"),
        ("gen_coord_violation", "coordination violation       ↓"),
        ("gen_task_residual", "task residual                ↓"),
    ]
    rows = []
    for k, label in keys:
        a = rec["x"].get(k)
        b = rec["v"].get(k)
        if a is None or b is None:
            continue
        rows.append((label, f"{a:.4f}", f"{b:.4f}"))
    show("(b,c,d) trained pair", rows)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"spectrum": spec, "x": rec["x"], "v": rec["v"]},
                               indent=2, default=float))
    print(f"\nwrote {path}")
    print("note: smoke budgets can reverse some rankings; use --full for the paper trend.")


if __name__ == "__main__":
    main()
