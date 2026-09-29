#!/usr/bin/env python3
"""Run real LiLa-WAM checkpoint endpoint and NFE probes on a manifest.

This is intentionally outside LiLa-WAM.  It calls its frozen inference/model
objects but never patches them.  One invocation can compare any number of
checkpoint specs of the form ``name=checkpoint.pt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from real_mechanism.backends import NFE_GRID, SIGMA_GRID
from real_mechanism.data import load_manifest_actions


TOY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TOY_ROOT.parent
LILAWAM_ROOT = PROJECT_ROOT / "LiLa-WAM"


def _parse_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("Each --checkpoint must be name=/absolute/or/relative/path.pt")
        name, path = value.split("=", 1)
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append((name, path))
    return specs


def _prediction_type(name: str) -> str:
    lowered = name.lower()
    if "vloss" in lowered or "v_loss" in lowered:
        return "x_prediction_v_loss"
    if "xpred" in lowered or "x_prediction" in lowered:
        return "x_prediction"
    if "vpred" in lowered or "v_prediction" in lowered:
        return "v_prediction"
    raise ValueError(f"Cannot infer prediction type from {name!r}; name it vpred/xpred/xpred_vloss.")


def _decode_image(h5: h5py.File, anchor: int) -> np.ndarray:
    # LiLa's own dataset loader has the canonical RobotWin byte decoder.
    from dataloader.dataset import _decode
    return _decode(h5["observation"]["head_camera"]["rgb"][anchor])


def _state(h5: h5py.File, anchor: int) -> np.ndarray:
    left_pose = h5["endpose"]["left_endpose"][anchor]
    left_grip = np.atleast_1d(h5["endpose"]["left_gripper"][anchor])
    right_pose = h5["endpose"]["right_endpose"][anchor]
    right_grip = np.atleast_1d(h5["endpose"]["right_gripper"][anchor])
    return np.concatenate([left_pose, left_grip, right_pose, right_grip]).astype(np.float32)


def _conditions(engine, row: dict[str, str]):
    from robotwin_infer import normalize_image_np
    with h5py.File(row["hdf5_path"], "r") as h5:
        image = _decode_image(h5, int(row["anchor"]))
        state = _state(h5, int(row["anchor"]))
    # Match LiLa's dataset preprocessing exactly (resize is handled by its
    # inference processor, image normalization is ImageNet normalization).
    import cv2
    if (image.shape[1], image.shape[0]) != engine.image_size:
        image = cv2.resize(image, engine.image_size, interpolation=cv2.INTER_LINEAR)
    pixel = torch.from_numpy(normalize_image_np(image)).unsqueeze(0).to(engine.device, engine.dtype)
    state_t = torch.from_numpy(state).view(1, 1, -1).to(engine.device, engine.dtype)
    qpos = engine.model.normalize_state(state_t)
    engine.set_task(row["task"])
    with torch.no_grad():
        dino = engine.model.get_vision_features(pixel)
    return qpos, dino, engine.task_cond


@torch.no_grad()
def _head(engine, x_t: torch.Tensor, t: torch.Tensor, qpos, dino, task_cond) -> torch.Tensor:
    return engine.model.action_model(
        t=t, noisy_actions=x_t, qpos_history=qpos,
        dino_features_list=dino, task_cond=task_cond,
    )["final_pred"]


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for real LiLa-WAM checkpoints; choose --device cpu only for a tiny debugging model.")
    sys.path.insert(0, str(LILAWAM_ROOT))
    from robotwin_infer import RobotWinInference
    target_raw, rows = load_manifest_actions(args.manifest)
    specs = _parse_specs(args.checkpoint)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    target_all: list[np.ndarray] = []
    all_outputs: list[np.ndarray] = []
    grid = SIGMA_GRID if args.kind == "recovery" else NFE_GRID

    for name, checkpoint in specs:
        engine = RobotWinInference(
            config_path=str(args.config), checkpoint_path=str(checkpoint), norm_stats_path=str(args.norm_stats),
            device=args.device, dtype=dtype,
        )
        expected_horizon = int(engine.config.common.action_chunk_size)
        if target_raw.shape[1] != expected_horizon:
            raise ValueError(
                f"LiLa-WAM checkpoint expects native horizon={expected_horizon}, but manifest has "
                f"horizon={target_raw.shape[1]}. Build a separate matching manifest; do not pad/crop it."
            )
        pred_type = _prediction_type(name)
        engine.action_prediction_type = pred_type
        model_outputs = []
        normalized_target = []
        for row, raw in zip(rows[:args.max_samples], target_raw[:args.max_samples]):
            qpos, dino, task_cond = _conditions(engine, row)
            target = engine.model.normalize_action(torch.from_numpy(raw).unsqueeze(0).to(engine.device, engine.dtype))
            normalized_target.append(target[0].float().cpu().numpy())
            grid_outputs = []
            for value in grid:
                seeds = []
                for seed in range(args.noise_seeds):
                    gen = torch.Generator(device=engine.device).manual_seed(args.seed + seed)
                    if args.kind == "recovery":
                        eps = torch.randn(target.shape, generator=gen, device=engine.device, dtype=engine.dtype)
                        sigma = float(value)
                        t = torch.full((1,), 1.0 - sigma, device=engine.device, dtype=engine.dtype)
                        x_t = (1.0 - t[:, None, None]) * eps + t[:, None, None] * target
                        pred = _head(engine, x_t, t, qpos, dino, task_cond)
                        endpoint = x_t + sigma * pred if pred_type == "v_prediction" else pred
                        seeds.append(endpoint[0].float().cpu().numpy())
                    else:
                        nfe = int(value)
                        x_t = torch.randn(target.shape, generator=gen, device=engine.device, dtype=engine.dtype)
                        steps = torch.linspace(0, 1, nfe + 1, device=engine.device, dtype=engine.dtype)
                        for i in range(nfe):
                            t = steps[i].view(1)
                            pred = _head(engine, x_t, t, qpos, dino, task_cond)
                            dt = steps[i + 1] - steps[i]
                            if pred_type == "v_prediction":
                                x_t = x_t + dt * pred
                            else:
                                keep = (1.0 - steps[i + 1]) / (1.0 - steps[i]).clamp_min(engine.x_pred_v_loss_eps)
                                x_t = keep * x_t + (1.0 - keep) * pred
                        seeds.append(x_t[0].float().cpu().numpy())
                grid_outputs.append(np.stack(seeds))
            model_outputs.append(np.stack(grid_outputs))  # [grid,R,T,D]
        all_outputs.append(np.stack(model_outputs, axis=2))  # [grid,R,N,T,D]
        target_all.append(np.stack(normalized_target))
        del engine
        torch.cuda.empty_cache()
    # Exact targets must agree across checkpoints because LiLa uses common stats.
    target = target_all[0]
    if not all(np.allclose(target, other, atol=1e-5) for other in target_all[1:]):
        raise RuntimeError("Checkpoint normalizers disagree; do not aggregate these checkpoints.")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "recovery":
        # rearrange [M,grid,R,N,T,D] to protocol [M,grid,R,N,T,D]
        np.savez_compressed(output, endpoint=np.stack(all_outputs), target=target,
                            sigma=SIGMA_GRID, model=np.asarray([n for n, _ in specs]))
    else:
        np.savez_compressed(output, sample=np.stack(all_outputs), target=target,
                            nfe=NFE_GRID, model=np.asarray([n for n, _ in specs]))
    return output


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("recovery", "nfe"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="name=path; repeat for x/v/x-vloss")
    p.add_argument("--config", type=Path, default=LILAWAM_ROOT / "configs/robotwin_all.yaml")
    p.add_argument("--norm-stats", type=Path, default=LILAWAM_ROOT / "utils/stat-500-all.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--noise-seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = p.parse_args()
    print(run(args))

if __name__ == "__main__": main()
