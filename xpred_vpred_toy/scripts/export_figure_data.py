"""Dump the numbers behind each figure as CSV, so the figures survive the runs.

The figure scripts read ``results_*/`` directly, which is 200 MB of per-run JSON and
awkward to ship.  This writes one tidy CSV per figure panel group -- every series
that appears in a figure, with its mean, standard error and seed count -- so a
reader can check a plotted point against a number, or redraw everything without the
raw runs.

    python export_figure_data.py --results results_full,results_v2,results_v3
"""

from __future__ import annotations

import argparse
import csv
import pathlib

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from velatoy.stats import load, summarise, table


# (figure, panel, experiment, metric, arm suffix) -- the arm suffix distinguishes
# the operating points that share an experiment, e.g. "@g1" from "@g4".
SERIES: list[tuple[str, str, str, list[str], list[str]]] = [
    ("figA", "b  jacobian rank vs t", "main_hd",
     [f"jac_participation_ratio_t{t}" for t in (10, 30, 50, 70, 90)]
     + ["jac_participation_ratio", "jac_rank99"],
     ["x-pred/v", "v-pred/v", "eps-pred/v"]),
    ("figA", "c  own-noise decodability", "main_hd",
     ["probe_eps_manip", "probe_eps_manip_shuf"],
     ["x-pred/v", "v-pred/v", "eps-pred/v"]),
    ("figA", "d  partner decodability", "main_hd",
     ["probe_partner_manip", "probe_partner_manip_shuf", "probe_partner_ceiling"],
     ["x-pred/v", "v-pred/v", "x-pred/v-nojoint", "v-pred/v-nojoint"]),
    ("figB", "a  risk vs t", "noise",
     [f"excess_risk_oracle_t{t}" for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 90)],
     ["x-pred/v+mode@g1", "v-pred/v+mode@g1", "eps-pred/v+mode@g1"]),
    ("figB", "b  invariance vs t", "noise",
     [f"niv_model_t{t}" for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 90)],
     ["x-pred/v+mode@g1", "v-pred/v+mode@g1"]),
    ("figB", "c  the trade at 58/token", "modewidth",
     ["excess_risk_oracle", "gen_off_manifold", "gen_coord_violation",
      "gen_task_residual", "gen_invalid_leakage", "nfe1_off_manifold",
      "gen_mode_mismatch"],
     ["x-pred/v+mode@g1", "v-pred/v+mode@g1"]),
    ("figB", "d  off-manifold vs NFE", "modewidth",
     [f"nfe{n}_off_manifold" for n in (1, 2, 4, 5, 8, 16)],
     ["x-pred/v+mode@g1", "v-pred/v+mode@g1"]),
    ("figC", "mode-gap rate", "main",
     ["gen_mode_collapse", "gen_mode_mismatch", "gen_coord_violation"],
     ["x-pred/v", "v-pred/v"]),
    ("figC", "mode-gap rate, mode supplied", "modewidth",
     ["gen_mode_collapse", "gen_mode_mismatch", "gen_coord_violation"],
     ["x-pred/v+mode@g1", "v-pred/v+mode@g1"]),
    ("figD", "width sweep", "modewidth",
     ["excess_risk_oracle", "excess_risk", "gen_off_manifold",
      "nfe1_off_manifold", "jac_participation_ratio"],
     [f"{p}-pred/v+mode@g{g}"
      for g in (1, 2, 4, 8, 16) for p in ("x", "v", "eps")]),
    ("table", "budget check, 4k", "noise",
     [f"excess_risk_oracle_t{t}" for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 90)],
     ["x-pred/v+mode@g1", "v-pred/v+mode@g1"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/data")
    args = ap.parse_args()

    recs = load(args.results)
    print(f"loaded {len(recs)} runs")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for fig, panel, exp, metrics, arms in SERIES:
        for metric in metrics:
            t = table(recs, exp, metric)
            for arm in arms:
                s = summarise(t.get((arm,), {}))
                if not s["n"]:
                    continue
                rows.append({
                    "figure": fig, "panel": panel, "experiment": exp,
                    "arm": arm, "metric": metric,
                    "mean": f"{s['mean']:.6g}", "sem": f"{s['sem']:.6g}",
                    "seeds": s["n"],
                })

    path = out / "figure_data.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} series)")

    missing = {(f, e) for f, _, e, _, _ in SERIES} - {
        (r["figure"], r["experiment"]) for r in rows
    }
    if missing:
        print("no data found for:", sorted(missing))


if __name__ == "__main__":
    main()
