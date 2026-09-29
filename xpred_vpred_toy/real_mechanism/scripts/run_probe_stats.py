#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from real_mechanism.probe_metrics import summarise_recovery, summarise_nfe

def main() -> None:
    p = argparse.ArgumentParser(description="Summarise portable endpoint-recovery or offline-NFE archives.")
    p.add_argument("--kind", choices=("recovery", "nfe"), required=True)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    (summarise_recovery if a.kind == "recovery" else summarise_nfe)(a.archive, a.output)
    print(f"wrote {a.output}")
if __name__ == "__main__": main()
