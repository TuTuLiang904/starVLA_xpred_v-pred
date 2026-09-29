#!/usr/bin/env python3
"""Only validates the archive protocol. Never use its results in a paper."""
from __future__ import annotations
import argparse
from pathlib import Path
from real_mechanism.backends import SyntheticBackend

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    backend = SyntheticBackend(a.manifest)
    print(backend.recovery(a.output_dir / "synthetic_recovery.npz"))
    print(backend.nfe(a.output_dir / "synthetic_nfe.npz"))
if __name__ == "__main__": main()
