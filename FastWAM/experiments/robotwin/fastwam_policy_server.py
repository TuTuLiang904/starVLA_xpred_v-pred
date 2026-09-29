"""FastWAM-model websocket server for RoboTwin evaluation."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np

from experiments.libero_plus.protocol import pack, unpack
from experiments.robotwin.fastwam_policy import deploy_policy


class RobotWinServerPolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.policy = deploy_policy.get_model({
            "ckpt_setting": args.checkpoint,
            "dataset_stats_path": args.dataset_stats,
            "sim_task": args.task,
            "sim_cfg_path": str(Path(args.project_root) / "configs" / "sim_robotwin.yaml"),
            "mixed_precision": args.mixed_precision,
            "device": args.device,
            "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps,
            "rand_device": "cpu",
            "tiled": False,
            "timing_enabled": False,
        })
        self.action_horizon = self.policy.action_horizon

    def infer(self, request: dict) -> np.ndarray:
        observation = {
            "observation": {
                "head_camera": {"rgb": request["head"]},
                "left_camera": {"rgb": request["left"]},
                "right_camera": {"rgb": request["right"]},
            },
            "joint_action": {"vector": np.asarray(request["state"], dtype=np.float32)},
        }
        return self.policy._infer_action_chunk(observation, str(request["instruction"]))


async def serve(policy: RobotWinServerPolicy, host: str, port: int) -> None:
    from websockets.asyncio.server import serve

    metadata = {"action_chunk_size": policy.action_horizon, "action_dim": 14}

    async def handler(websocket):
        await websocket.send(pack(metadata))
        async for raw in websocket:
            try:
                request = unpack(raw)
                if request.get("type") != "infer":
                    raise ValueError("Unsupported request type")
                action = await asyncio.to_thread(policy.infer, request)
                await websocket.send(pack({"ok": True, "actions": action}))
            except Exception as exc:
                logging.exception("FastWAM RoboTwin inference failed")
                await websocket.send(pack({"ok": False, "error": repr(exc)}))

    async with serve(handler, host, port, compression=None, max_size=None):
        logging.info("FastWAM RoboTwin server listening on %s:%d", host, port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--task", default="robotwin_uncond_3cam_384_1e-4")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=24)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    asyncio.run(serve(RobotWinServerPolicy(args), "0.0.0.0", args.port))


if __name__ == "__main__":
    main()
