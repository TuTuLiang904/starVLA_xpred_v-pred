#!/usr/bin/env python
"""Run every experiment in the study, one worker per GPU.

    python run_all.py --out results --seeds 8

Each job is one (experiment, arm, seed).  Jobs are independent and are handed to
a pool of worker processes pinned to one device each.  Results land as one JSON
per job plus a flat CSV of the final metrics, so a failed job can be re-run
without touching the others.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_jobs(args) -> list[dict]:
    """Job descriptions only -- no torch import, so the parent stays light."""
    from velatoy.arms import (
        fix_arms, intrinsic_arms, main_arms, modewidth_arms, schedule_arms,
        seqlen_arms, token_group_arms, width_arms,
    )

    jobs = []
    # Two operating points, both run at full length with the full probe suite.
    #   main    -- VELA-0 as it stands: one token per chunk step, 58 numbers per
    #              token against a 512-wide stream.
    #   main_hd -- four chunk steps per token, 232 numbers per token.  Same total
    #              supervision and the stream is still wider than the token, so
    #              nothing is starved; only the per-token load has moved.
    # Reporting both is the point: the mechanistic gap is present at either, and
    # the fit gap appears only once the per-token load is non-trivial.
    for seed in range(args.seeds):
        for arm in main_arms():
            jobs.append({"experiment": "main", "arm": arm.__dict__, "seed": seed})
    for seed in range(args.hd_seeds):
        for arm in main_arms():
            jobs.append({
                "experiment": "main_hd", "arm": arm.__dict__, "seed": seed,
                "token_group": 4,
            })
    for seed in range(args.sweep_seeds):
        for arm, group in token_group_arms():
            jobs.append({
                "experiment": "tokengroup", "arm": arm.__dict__, "seed": seed,
                "token_group": group,
            })
    # The per-token-width sweep is the study's most important figure and its
    # largest number, so three of its points are also run at full length: a short
    # budget favours a skip connection, which makes the short-budget crossing a
    # lower bound rather than an estimate.  Reporting only the bound invites the
    # obvious objection that the effect is under-training.
    for seed in range(args.hd_seeds):
        for arm, group in token_group_arms():
            if group not in (1, 4, 16):
                continue
            jobs.append({
                "experiment": "tokengroup_long", "arm": arm.__dict__, "seed": seed,
                "token_group": group,
            })
    # Does the mode-commitment cost survive an explicit discrete latent?  Run at
    # the wide operating point, where the advantage is real and the cost is worst.
    for seed in range(args.hd_seeds):
        for arm in fix_arms():
            jobs.append({
                "experiment": "fix", "arm": arm.__dict__, "seed": seed,
                "token_group": 4,
            })
    # The width sweep re-run with the coordination mode supplied, which is the
    # setting a language- or goal-conditioned action expert is actually in.  Run at
    # full length at every width, because this is the comparison the headline claim
    # rests on and a short budget favours the skip convention.
    for seed in range(args.hd_seeds):
        for arm, group in modewidth_arms():
            jobs.append({
                "experiment": "modewidth", "arm": arm.__dict__, "seed": seed,
                "token_group": group,
            })
    # The headline claim -- that the two conventions differ by noise level and not in
    # aggregate -- rests on the high-noise end of the t grid, which is precisely where
    # the training time distribution puts almost no mass.  So it gets its own run at
    # VELA-0's own width, with twice the seeds, on the grid that was densified below
    # t = 0.1.  Same arms as ``modewidth`` at g = 1, so the two are directly poolable
    # and the extra seeds are extra power on exactly the contrast being claimed.
    for seed in range(args.noise_seeds):
        for arm, group in modewidth_arms():
            if group != 1:
                continue
            jobs.append({
                "experiment": "noise", "arm": arm.__dict__, "seed": seed,
                "token_group": group,
            })
    # Width or length?  Grouping steps into a token does both at once.
    for seed in range(args.hd_seeds):
        for arm, chunk, group in seqlen_arms():
            jobs.append({
                "experiment": "seqlen", "arm": arm.__dict__, "seed": seed,
                "chunk": chunk, "token_group": group,
            })
    # The manifold-thickness sweep re-run at the operating point where the
    # parameterizations actually differ.  At g = 1 it answers a different question.
    for seed in range(args.sweep_seeds):
        for arm, style in intrinsic_arms():
            jobs.append({
                "experiment": "intrinsic_hd", "arm": arm.__dict__, "seed": seed,
                "style_dim": style, "token_group": 4,
            })
    for seed in range(args.width_seeds):
        for arm in width_arms():
            jobs.append({"experiment": "width", "arm": arm.__dict__, "seed": seed})
    for seed in range(args.sweep_seeds):
        for arm, style in intrinsic_arms():
            jobs.append({
                "experiment": "intrinsic", "arm": arm.__dict__, "seed": seed,
                "style_dim": style,
            })
    for seed in range(args.sweep_seeds):
        for arm, beta, tag in schedule_arms():
            jobs.append({
                "experiment": "schedule", "arm": arm.__dict__, "seed": seed,
                "time_beta": list(beta), "schedule": tag,
            })
    return jobs


def run_job(spec: dict, args: dict, device: str) -> dict:
    import torch

    from velatoy.config import (
        ArmConfig, DataConfig, EvalConfig, FlowConfig, ModelConfig, RunConfig,
        TrainConfig,
    )
    from velatoy.train import run

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    exp = spec["experiment"]
    arm = ArmConfig(**spec["arm"])
    data = DataConfig()
    if "style_dim" in spec:
        data = replace(data, style_dim=spec["style_dim"])
    if "chunk" in spec:
        data = replace(data, chunk=spec["chunk"], near=spec["chunk"] // 2)
    model = ModelConfig(token_group=spec.get("token_group", 1))
    flow = FlowConfig()
    if "time_beta" in spec:
        flow = replace(flow, time_beta=tuple(spec["time_beta"]))

    headline = exp in (
        "main", "main_hd", "fix", "seqlen", "tokengroup_long", "modewidth", "noise"
    )
    steps = args["steps"] if headline else args["sweep_steps"]
    n_eval = args["eval_samples"] if headline else 2048
    cfg = RunConfig(
        data=data,
        model=model,
        flow=flow,
        train=TrainConfig(
            steps=steps, seed=spec["seed"], probe_every=args["probe_every"]
        ),
        eval=EvalConfig(n_samples=n_eval),
        arm=arm,
        experiment=exp,
        device=device,
    )
    res = run(cfg, verbose=args["verbose"])
    return {
        "experiment": exp,
        "arm": arm.name,
        "label": arm.label,
        "parameterization": arm.parameterization,
        "loss_space": arm.loss_space,
        "joint_attention": arm.joint_attention,
        "consistency_weight": arm.consistency_weight,
        "mode_conditioning": arm.mode_conditioning,
        "mode_agree_weight": arm.mode_agree_weight,
        "width": arm.width or cfg.model.width,
        "style_dim": data.style_dim,
        "chunk": data.chunk,
        "token_group": model.token_group,
        "token_dim": 58 * model.token_group,
        "n_tokens": data.chunk // model.token_group,
        "schedule": spec.get("schedule", "vela"),
        "seed": spec["seed"],
        "final": res.final,
        "dynamics": res.dynamics,
        "artifacts": {k: v.tolist() for k, v in res.artifacts.items()}
        if spec["seed"] == 0 and exp in ("main", "main_hd", "fix", "modewidth")
        else {},
    }


def worker(rank: int, world: int, jobs: list[dict], args: dict, outdir: str, gpus):
    device = f"cuda:{gpus[rank % len(gpus)]}" if gpus else "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    for i, spec in enumerate(jobs):
        if i % world != rank:
            continue
        tag = f"{spec['experiment']}__{spec['arm']['name'].replace('/', '-')}__s{spec['seed']}"
        path = Path(outdir) / f"{tag}.json"
        if path.exists():
            continue
        try:
            rec = run_job(spec, args, device)
            path.write_text(json.dumps(rec))
            print(f"[rank {rank}] done {tag}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[rank {rank}] FAILED {tag}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--hd-seeds", type=int, default=8)
    ap.add_argument("--noise-seeds", type=int, default=16)
    ap.add_argument("--width-seeds", type=int, default=4)
    ap.add_argument("--sweep-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--sweep-steps", type=int, default=4000)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--eval-samples", type=int, default=4096)
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated experiment filter")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["experiment"] in keep]
    def cost(j):
        a = j["arm"]
        w = a.get("width") or 512
        c = (w / 512.0) ** 1.7 * j.get("token_group", 1) ** -0.45
        c *= j.get("chunk", 16) / 16.0
        if a.get("consistency_weight", 0.0) > 0:
            c *= 2.8
        long_run = j["experiment"] in (
            "main", "main_hd", "fix", "seqlen", "tokengroup_long", "modewidth"
        )
        return c * (12000 if long_run else 4000)

    jobs.sort(key=cost, reverse=True)
    print(f"{len(jobs)} jobs -> {outdir}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = list(range(len(visible.split(",")))) if visible else list(range(8))
    world = max(1, len(gpus) * args.workers_per_gpu)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=worker,
            args=(r, world, jobs, vars(args), str(outdir), gpus),
        )
        for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("all workers finished")


if __name__ == "__main__":
    main()
