"""Websocket policy server for FastWAM + LIBERO-plus evaluation.

The simulator stays in the LIBERO-plus environment while this process owns one
FastWAM model on one GPU.  Requests contain the two raw LIBERO camera images,
the 8-D simulator proprioception, and the natural-language task instruction.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed

from experiments.libero_plus.protocol import pack, unpack


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _checkpoint_action_prediction_type(checkpoint: str) -> tuple[str, Path]:
    """Read the action parameterization recorded with a training checkpoint."""
    run_config = Path(checkpoint).resolve().parents[2] / "config.yaml"
    if not run_config.is_file():
        raise FileNotFoundError(
            f"Could not find the training config next to checkpoint {checkpoint}: expected {run_config}."
        )
    value = OmegaConf.select(OmegaConf.load(run_config), "model.action_prediction_type")
    if value is None:
        raise ValueError(f"model.action_prediction_type is missing from {run_config}.")
    return str(value), run_config


class FastWAMPolicy:
    def __init__(self, checkpoint: str, dataset_stats: str, task_config: str, device: str, steps: int):
        self.device = torch.device(device)
        self.steps = int(steps)
        action_prediction_type, run_config = _checkpoint_action_prediction_type(checkpoint)
        project_root = Path(__file__).resolve().parents[2]
        config_dir = project_root / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
            cfg = compose(
                config_name="sim_libero",
                overrides=[
                    f"task={task_config}",
                    "model.load_text_encoder=true",
                    "model.skip_dit_load_from_pretrain=true",
                    "model.action_dit_pretrained_path=null",
                    f"model.action_prediction_type={action_prediction_type}",
                ],
            )

        model = instantiate(
            cfg.model,
            model_dtype=torch.bfloat16,
            device=str(self.device),
        )
        model.load_checkpoint(checkpoint)
        self.model = model.eval()

        processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(dataset_stats))
        self.processor = processor
        self.action_horizon = int(cfg.data.train.num_frames) - 1
        self.image_height = int(cfg.data.train.shape_meta.images[0].shape[1])
        self.image_width = int(cfg.data.train.shape_meta.images[0].shape[2])
        self.num_inference_steps = self.steps

        logging.info(
            "FastWAM policy ready: checkpoint=%s action_horizon=%d image=%dx%d "
            "action_prediction_type=%s training_config=%s",
            checkpoint,
            self.action_horizon,
            self.image_height,
            self.image_width,
            getattr(model, "action_prediction_type", "v_prediction"),
            run_config,
        )

    def _normalize_proprio(self, proprio: np.ndarray) -> torch.Tensor:
        state_key = self.processor.shape_meta["state"][0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    @torch.inference_mode()
    def predict_action(self, primary: np.ndarray, wrist: np.ndarray, proprio: np.ndarray, instruction: str):
        primary = _center_crop_resize(primary, self.image_width, self.image_height)
        wrist = _center_crop_resize(wrist, self.image_width, self.image_height)
        rgb = np.concatenate([primary, wrist], axis=1)
        image = torch.as_tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=self.device, dtype=self.model.torch_dtype)
        image = image * (2.0 / 255.0) - 1.0
        state = self._normalize_proprio(np.asarray(proprio, dtype=np.float32))
        prompt = DEFAULT_PROMPT.format(task=str(instruction))

        output = self.model.infer_action(
            prompt=prompt,
            input_image=image,
            action_horizon=self.action_horizon,
            proprio=state,
            num_inference_steps=self.num_inference_steps,
            seed=None,
            rand_device="cpu",
            tiled=False,
        )
        action = output["action"].detach().cpu().float()
        action_key = self.processor.shape_meta["action"][0]["key"]
        action = self.processor.normalizer.normalizers["action"][action_key].backward(action).numpy()
        return np.asarray(action, dtype=np.float32)


async def _serve(policy: FastWAMPolicy, host: str, port: int):
    try:
        from websockets.asyncio.server import serve
    except ImportError:
        from websockets.server import serve

    metadata = {
        "env": "fastwam_policy_server",
        "action_chunk_size": policy.action_horizon,
        "training_obs_image_size": [policy.image_height, policy.image_width],
        "num_cameras": 2,
        "proprio_dim": 8,
        "action_dim": 7,
        "action_prediction_type": policy.model.action_prediction_type,
    }

    async def handler(websocket):
        await websocket.send(pack(metadata))
        async for raw in websocket:
            request = unpack(raw)
            try:
                if request.get("type", "infer") != "infer":
                    raise ValueError(f"Unsupported request type: {request.get('type')}")
                action = policy.predict_action(
                    primary=request["primary"],
                    wrist=request["wrist"],
                    proprio=request["proprio"],
                    instruction=request["instruction"],
                )
                response = {"ok": True, "actions": action}
            except Exception as exc:
                logging.exception("FastWAM policy inference failed")
                response = {"ok": False, "error": repr(exc)}
            await websocket.send(pack(response))

    async with serve(handler, host, port, compression=None, max_size=None):
        logging.info("FastWAM policy server listening on %s:%d", host, port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", force=True)
    policy = FastWAMPolicy(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        task_config=args.task_config,
        device=args.device,
        steps=args.num_inference_steps,
    )
    asyncio.run(_serve(policy, host="0.0.0.0", port=args.port))


if __name__ == "__main__":
    main()
