#!/usr/bin/env python3
"""Merge independently-run one-checkpoint probe archives along the model axis."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("recovery", "nfe"), required=True)
    p.add_argument("--archive", action="append", required=True,
                   help="name=archive.npz; repeat once per checkpoint")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    value_key, grid_key = ("endpoint", "sigma") if a.kind == "recovery" else ("sample", "nfe")
    values, names, target, grid = [], [], None, None
    for spec in a.archive:
        name, sep, raw_path = spec.partition("=")
        if not sep:
            raise ValueError("--archive requires name=path.npz")
        data = np.load(Path(raw_path), allow_pickle=False)
        for key in (value_key, "target", grid_key, "model"):
            if key not in data:
                raise ValueError(f"{raw_path} lacks {key}")
        if target is None:
            target, grid = data["target"], data[grid_key]
        elif not np.allclose(target, data["target"], atol=1e-5):
            raise ValueError("Target arrays differ; do not merge these archives.")
        elif not np.array_equal(grid, data[grid_key]):
            raise ValueError("Recovery/NFE grids differ; do not merge these archives.")
        values.append(data[value_key])
        names.append(name)
    if not values:
        raise ValueError("No archives supplied")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **{value_key: np.concatenate(values, axis=0),
                                    "target": target, grid_key: grid,
                                    "model": np.asarray(names)})
    print(a.output)


if __name__ == "__main__":
    main()
