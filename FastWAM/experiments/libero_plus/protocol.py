"""Small msgpack protocol shared by the FastWAM server and LIBERO-plus client."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


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
    raise TypeError(f"Cannot encode value of type {type(value)!r}")


def _object_hook(value: dict):
    if value.get("__ndarray__"):
        array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"]))
    return value


def pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_default, use_bin_type=True)


def unpack(value: bytes) -> Any:
    return msgpack.unpackb(value, raw=False, object_hook=_object_hook)
