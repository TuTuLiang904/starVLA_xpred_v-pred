#!/usr/bin/env python3
"""Real StarVLA RobotWin endpoint/NFE probe, kept outside the StarVLA tree."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from real_mechanism.backends import NFE_GRID, SIGMA_GRID
from real_mechanism.data import load_manifest_actions

TOY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TOY_ROOT.parent
STAR_ROOT = PROJECT_ROOT / "starVLA"


def _specs(values):
    result = []
    for value in values:
        name, sep, raw = value.partition("=")
        if not sep:
            raise ValueError("Each --checkpoint must be name=/path/to/model.pt")
        # Long checkpoint commands are often copied with a visual line break
        # before ``final_model``.  Whitespace at the ends is not part of a
        # filesystem path and should never turn into a misleading FileNotFound.
        path = Path("".join(raw.split())).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        if "vloss" in name or "v_loss" in name: kind = "x_prediction_v_loss"
        elif "xpred" in name or "x_prediction" in name: kind = "x_prediction"
        elif "vpred" in name or "v_prediction" in name: kind = "v_prediction"
        else: raise ValueError(f"Name {name!r} must contain vpred, xpred, or xpred_vloss")
        result.append((name, kind, path))
    return result


def _instruction(row):
    if row.get("source") == "lerobot":
        return row["instruction"]
    h5_path = Path(row["hdf5_path"])
    path = h5_path.parent.parent / "instructions" / f"{h5_path.stem}.json"
    data = json.loads(path.read_text())
    return str(data.get("seen", data.get("unseen", [row["task"]]))[0])


def _images(row):
    # Exactly the RobotWin deployment view order: head, left wrist, right wrist.
    if row.get("source") == "lerobot":
        import av
        parquet = Path(row["parquet_path"])
        root = parquet.parents[2]
        episode = int(Path(row["episode"]).stem.split("_")[-1])
        frame_index = int(row["anchor"])
        frames = []
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            video = root / "videos" / f"chunk-{episode // 1000:03d}" / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"
            with av.open(str(video)) as container:
                decoded = next(frame for i, frame in enumerate(container.decode(video=0)) if i == frame_index)
            frames.append(decoded.to_ndarray(format="rgb24"))
        return frames
    import cv2
    def _decode(value):
        return cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    with h5py.File(row["hdf5_path"], "r") as h5:
        obs = h5["observation"]
        return [_decode(obs[key]["rgb"][int(row["anchor"])]) for key in
                ("head_camera", "left_camera", "right_camera")]


def _normalise_action(raw, stats):
    low, high = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    norm = 2.0 * (raw - low) / np.maximum(high - low, 1e-6) - 1.0
    return np.where(mask, norm, raw).astype(np.float32)


def _state(row):
    if row.get("source") != "lerobot":
        raise RuntimeError("StarVLA probe requires the native LeRobot manifest for proprioceptive state.")
    import pyarrow.parquet as pq
    return np.asarray(pq.read_table(row["parquet_path"], columns=["observation.state"])["observation.state"].to_pylist()[int(row["anchor"])], dtype=np.float32)


@torch.no_grad()
def _condition(model, row):
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.training.trainer_utils.trainer_tools import resize_images
    imgs = to_pil_preserve(_images(row))
    target_size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    if target_size: imgs = resize_images([imgs], target_size=target_size)[0]
    inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=[imgs], instructions=[_instruction(row)])
    mask = inputs.get("attention_mask")
    if mask is not None: mask = mask.to(dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        last_hidden = model.qwen_vl_interface(**inputs, output_attentions=False,
                                              output_hidden_states=True, return_dict=True).hidden_states[-1]
    return last_hidden, mask


@torch.no_grad()
def _head_at(model, hidden, mask, actions, t, state):
    """The released head forward with externally controlled action/noise time."""
    head = model.action_model
    b = actions.shape[0]
    buckets = (t * head.num_timestep_buckets).long()
    features = head.action_encoder(actions, buckets)
    if head.config.add_pos_embed:
        ids = torch.arange(features.shape[1], device=features.device)
        features = features + head.position_embedding(ids).unsqueeze(0)
    future = head.future_tokens.weight.unsqueeze(0).expand(b, -1, -1)
    state_features = head.state_encoder(state) if head.state_encoder is not None else None
    sequence = torch.cat((state_features, future, features), dim=1) if state_features is not None else torch.cat((future, features), dim=1)
    output = head.model(hidden_states=sequence,
                        encoder_hidden_states=hidden, encoder_attention_mask=mask,
                        timestep=buckets, return_all_hidden_states=False)
    return head.action_decoder(output)[:, -actions.shape[1]:]


def run(a):
    if not torch.cuda.is_available(): raise RuntimeError("StarVLA checkpoint probe requires CUDA.")
    # Resolve CLI paths before entering StarVLA's required working directory.
    manifest = Path(a.manifest).resolve()
    output_path = Path(a.output).resolve()
    specs = _specs(a.checkpoint)
    raw, rows = load_manifest_actions(manifest)
    sys.path.insert(0, str(STAR_ROOT))
    # Checkpoint configs intentionally store the VLM base path relative to the
    # StarVLA repository.  Keep that convention rather than rewriting configs.
    os.chdir(STAR_ROOT)
    from starVLA.model.framework.base_framework import baseframework
    grid = SIGMA_GRID if a.kind == "recovery" else NFE_GRID
    outputs, targets = [], []
    for name, kind, checkpoint in specs:
        print(f"[starvla] loading checkpoint {name}: {checkpoint}", flush=True)
        model = baseframework.from_pretrained(str(checkpoint)).to(torch.bfloat16).to(a.device).eval()
        h = int(model.action_model.action_horizon)
        if raw.shape[1] != h:
            raise ValueError(f"{name} expects native horizon={h}, manifest is {raw.shape[1]}; build matching h{h} manifest.")
        stats = next(iter(model.norm_stats.values()))["action"]
        state_stats = next(iter(model.norm_stats.values())).get("state")
        per_sample, target_list = [], []
        selected_rows = rows[:a.max_samples]
        selected_raw = raw[:a.max_samples]
        total = len(selected_rows)
        for sample_index, (row, target_raw) in enumerate(zip(selected_rows, selected_raw), start=1):
            if sample_index == 1 or sample_index % a.progress_every == 0 or sample_index == total:
                print(f"[starvla] {name}: anchor {sample_index}/{total}", flush=True)
            hidden, mask = _condition(model, row)
            target = torch.from_numpy(_normalise_action(target_raw, stats)).unsqueeze(0).to(a.device, torch.bfloat16)
            if model.action_model.state_encoder is not None:
                if state_stats is None: raise RuntimeError("Checkpoint has state_encoder but no state normalization stats.")
                state = torch.from_numpy(_normalise_action(_state(row), state_stats)).view(1, 1, -1).to(a.device, torch.bfloat16)
            else:
                state = None
            target_list.append(target[0].float().cpu().numpy())
            per_grid = []
            for value in grid:
                seeds = []
                for seed in range(a.seed, a.seed + a.noise_seeds):
                    gen = torch.Generator(device=a.device).manual_seed(seed)
                    if a.kind == "recovery":
                        sigma = float(value); noise = torch.randn(target.shape, generator=gen, device=a.device)
                        t = torch.full((1,), 1.0 - sigma, device=a.device)
                        z = (sigma * noise + (1.0 - sigma) * target).to(dtype=target.dtype)
                        pred = _head_at(model, hidden, mask, z, t, state)
                        out = z + sigma * pred if kind == "v_prediction" else pred
                    else:
                        nfe = int(value)
                        out = torch.randn(target.shape, generator=gen, device=a.device, dtype=target.dtype)
                        for i in range(nfe):
                            t0 = i / nfe; t = torch.full((1,), t0, device=a.device)
                            pred = _head_at(model, hidden, mask, out, t, state)
                            if kind == "v_prediction": out = out + pred / nfe
                            else: out = ((nfe-i-1)/(nfe-i))*out + pred/(nfe-i)
                    seeds.append(out[0].float().cpu().numpy())
                per_grid.append(np.stack(seeds))
            per_sample.append(np.stack(per_grid))
        outputs.append(np.stack(per_sample, axis=2)); targets.append(np.stack(target_list))
        # Keep a one-checkpoint archive as a recoverable intermediate.  The
        # combined archive is written only after all requested checkpoints.
        partial = output_path.with_name(output_path.stem + f".{name}.npz")
        np.savez_compressed(partial, **{("endpoint" if a.kind == "recovery" else "sample"): np.stack(outputs[-1:]),
                                       "target": targets[-1],
                                       ("sigma" if a.kind == "recovery" else "nfe"): grid,
                                       "model": np.asarray([name])})
        print(f"[starvla] wrote intermediate {partial}", flush=True)
        del model; torch.cuda.empty_cache()
    target = targets[0]
    if not all(np.allclose(target, x, atol=1e-5) for x in targets[1:]):
        raise RuntimeError("StarVLA action normalizers differ across checkpoints; preserve separate archives.")
    out = output_path; out.parent.mkdir(parents=True, exist_ok=True)
    key = "endpoint" if a.kind == "recovery" else "sample"
    np.savez_compressed(out, **{key: np.stack(outputs), "target": target,
                                ("sigma" if a.kind == "recovery" else "nfe"): grid,
                                "model": np.asarray([x[0] for x in specs])})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("recovery", "nfe"), required=True); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=2000); p.add_argument("--noise-seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260918); p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=10, help="Print progress every N anchors.")
    print(run(p.parse_args()))
if __name__ == "__main__": main()
