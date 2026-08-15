"""LIBERO-plus-side websocket client for a FastWAM policy server."""

from __future__ import annotations

import time

import websockets.sync.client

from experiments.libero_plus.protocol import pack, unpack


class FastWAMClient:
    def __init__(self, host: str, port: int, timeout: float = 300.0):
        self.uri = f"ws://{host}:{port}"
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            try:
                self.websocket = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    open_timeout=30,
                    # Policy servers are local processes. Do not route their
                    # websocket handshake through the machine's HTTP proxy.
                    proxy=None,
                )
                self.metadata = unpack(self.websocket.recv())
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for FastWAM server {self.uri}: {last_error!r}")

    @property
    def action_chunk_size(self) -> int:
        return int(self.metadata["action_chunk_size"])

    def predict_action(self, primary, wrist, proprio, instruction):
        self.websocket.send(
            pack(
                {
                    "type": "infer",
                    "primary": primary,
                    "wrist": wrist,
                    "proprio": proprio,
                    "instruction": instruction,
                }
            )
        )
        response = unpack(self.websocket.recv())
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "FastWAM server inference failed"))
        return response["actions"]

    def close(self):
        try:
            self.websocket.close()
        except Exception:
            pass
