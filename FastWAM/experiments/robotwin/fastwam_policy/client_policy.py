"""RoboTwin-side websocket client for a FastWAM policy server.

This module intentionally contains no FastWAM imports: it runs in the
RoboTwin conda environment while the model itself remains in ``fastwam``.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _default(value: Any):
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": value.shape,
            "data": value.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value)!r}")


def _object_hook(value: dict):
    if value.get("__ndarray__"):
        return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    return value


def _pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_default, use_bin_type=True)


def _unpack(value: bytes) -> Any:
    return msgpack.unpackb(value, raw=False, object_hook=_object_hook)


class FastWAMRobotWinClient:
    def __init__(self, host: str, port: int, replan_steps: int = 24) -> None:
        self.uri = f"ws://{host}:{int(port)}"
        self.ws = self._connect()
        self.metadata = _unpack(self.ws.recv())
        self.action_horizon = int(self.metadata["action_chunk_size"])
        self.replan_steps = min(max(1, int(replan_steps)), self.action_horizon)
        self.actions: deque[np.ndarray] = deque()

    def _connect(self):
        deadline = time.time() + 300
        last_error = None
        while time.time() < deadline:
            try:
                return websockets.sync.client.connect(
                    self.uri, compression=None, max_size=None, open_timeout=30, proxy=None
                )
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for FastWAM server {self.uri}: {last_error!r}")

    def reset(self, task_description: str = "") -> None:
        self.actions.clear()

    def should_request_observation(self) -> bool:
        return not self.actions

    def _request_chunk(self, observation: dict, instruction: str) -> None:
        obs = observation["observation"]
        request = {
            "type": "infer",
            "head": obs["head_camera"]["rgb"],
            "left": obs["left_camera"]["rgb"],
            "right": obs["right_camera"]["rgb"],
            "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "instruction": str(instruction),
        }
        self.ws.send(_pack(request))
        response = _unpack(self.ws.recv())
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "FastWAM policy server failed"))
        for action in np.asarray(response["actions"], dtype=np.float32)[: self.replan_steps]:
            self.actions.append(action)

    def step(self, task_env, observation: dict | None) -> None:
        if not self.actions:
            if observation is None:
                raise ValueError("A replan step requires an observation")
            self._request_chunk(observation, task_env.get_instruction())
        task_env.take_action(self.actions.popleft(), action_type="qpos")

    def close(self) -> None:
        self.ws.close()


def get_model(usr_args: dict):
    return FastWAMRobotWinClient(
        host=str(usr_args.get("host", "127.0.0.1")),
        port=int(usr_args.get("port", 5694)),
        replan_steps=int(usr_args.get("replan_steps", 24)),
    )


def reset_model(model: FastWAMRobotWinClient):
    model.reset()


def eval(TASK_ENV, model: FastWAMRobotWinClient, observation: dict | None):
    model.step(TASK_ENV, observation)
