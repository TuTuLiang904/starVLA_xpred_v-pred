#!/usr/bin/env python
"""Train a single experimental arm and write one JSON result.

Examples:

    python scripts/train_one.py --arm x-pred/v+mode --seed 0 --steps 4000
    python scripts/train_one.py --arm v-pred/v+mode --seed 0 --steps 4000
    python scripts/train_one.py --arm x-pred/v --no-mode --steps 800 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_arm(name: str, mode: bool):
    from velatoy.config import ArmConfig

    parameterization = name.split("-")[0]
    if parameterization not in ("x", "v", "eps"):
        raise SystemExit(f"cannot parse parameterization from {name!r}")
    return ArmConfig(
        name=name,
        parameterization=parameterization,
        loss_space="v",
        mode_conditioning=mode,
        label=name,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="x-pred/v+mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--token-group", type=int, default=1)
    ap.add_argument("--eval-samples", type=int, default=2048)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--experiment", default="single")
    ap.add_argument("--no-mode", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="narrow net, fewer eval samples, for a smoke test")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch

    from velatoy.config import (
        DataConfig, EvalConfig, FlowConfig, ModelConfig, RunConfig, TrainConfig,
    )
    from velatoy.train import run

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    width = 128 if args.quick else args.width
    eval_n = 256 if args.quick else args.eval_samples
    steps = 400 if args.quick and args.steps == 4000 else args.steps
    probe_every = 100 if args.quick else args.probe_every
    mode = not args.no_mode
    arm = parse_arm(args.arm, mode)
    arm.width = width

    cfg = RunConfig(
        data=DataConfig(),
        model=ModelConfig(width=width, token_group=args.token_group,
                          mode_conditioning=mode),
        flow=FlowConfig(),
        train=TrainConfig(
            steps=steps, seed=args.seed, batch=args.batch,
            probe_every=probe_every,
        ),
        eval=EvalConfig(n_samples=eval_n),
        arm=arm,
        experiment=args.experiment,
        device=device,
    )
    print(f"training {arm.name}  seed={args.seed}  steps={steps}  "
          f"width={width}  device={device}", flush=True)
    res = run(cfg, verbose=args.verbose)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.experiment}__{arm.name.replace('/', '-')}__s{args.seed}"
    rec = {
        "experiment": args.experiment,
        "arm": arm.name,
        "label": arm.label,
        "parameterization": arm.parameterization,
        "loss_space": arm.loss_space,
        "joint_attention": arm.joint_attention,
        "consistency_weight": arm.consistency_weight,
        "mode_conditioning": arm.mode_conditioning,
        "mode_agree_weight": arm.mode_agree_weight,
        "width": width,
        "style_dim": cfg.data.style_dim,
        "chunk": cfg.data.chunk,
        "token_group": args.token_group,
        "token_dim": 58 * args.token_group,
        "n_tokens": cfg.data.chunk // args.token_group,
        "schedule": "vela",
        "seed": args.seed,
        "final": res.final,
        "dynamics": res.dynamics,
        "artifacts": {},
    }
    path = outdir / f"{tag}.json"
    path.write_text(json.dumps(rec))
    keys = (
        "jac_participation_ratio", "jac_rank99", "probe_eps_manip",
        "probe_partner_manip", "excess_risk_oracle", "nfe1_off_manifold",
        "nfe5_off_manifold", "gen_invalid_leakage",
    )
    print(f"wrote {path}")
    for k in keys:
        if k in res.final:
            print(f"  {k:28s}  {res.final[k]:.4f}")


if __name__ == "__main__":
    main()
