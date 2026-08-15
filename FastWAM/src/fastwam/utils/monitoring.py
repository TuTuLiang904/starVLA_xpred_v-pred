from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f"{body}</svg>\n"
    )


def write_loss_curve_svg(records: Iterable[dict], output_path: str | Path, max_points: int = 2000) -> None:
    """Write a dependency-free SVG for train/eval loss history."""
    series = {
        "train/loss": [],
        "train/loss_action": [],
        "train/loss_video": [],
        "eval/val_loss": [],
    }
    for record in records:
        kind = record.get("kind")
        step = record.get("step")
        if not isinstance(step, (int, float)):
            continue
        if kind == "train":
            for key, name in (("loss", "train/loss"), ("loss_action", "train/loss_action"), ("loss_video", "train/loss_video")):
                value = record.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    series[name].append((float(step), float(value)))
        elif kind == "eval":
            value = record.get("val_loss")
            if isinstance(value, (int, float)) and math.isfinite(value):
                series["eval/val_loss"].append((float(step), float(value)))

    nonempty = {name: values for name, values in series.items() if values}
    if not nonempty:
        return

    width, height = 960, 520
    left, right, top, bottom = 78, 32, 46, 64
    points = [point for values in nonempty.values() for point in values]
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    y_min, y_max = min(y for _, y in points), max(y for _, y in points)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    y_pad = max((y_max - y_min) * 0.05, 1e-8)
    y_min -= y_pad
    y_max += y_pad

    plot_w = width - left - right
    plot_h = height - top - bottom

    def point_to_svg(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_min) / (x_max - x_min) * plot_w
        py = top + (y_max - y) / (y_max - y_min) * plot_h
        return px, py

    body = [
        '<text x="24" y="30" font-family="sans-serif" font-size="18">Training loss</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left}" y="{height - 20}" font-family="sans-serif" font-size="12">step {x_min:.0f}</text>',
        f'<text x="{left + plot_w - 80}" y="{height - 20}" font-family="sans-serif" font-size="12">step {x_max:.0f}</text>',
        f'<text x="8" y="{top + 12}" font-family="sans-serif" font-size="12">{y_max:.4g}</text>',
        f'<text x="8" y="{top + plot_h}" font-family="sans-serif" font-size="12">{y_min:.4g}</text>',
    ]
    colors = {
        "train/loss": "#2563eb",
        "train/loss_action": "#dc2626",
        "train/loss_video": "#16a34a",
        "eval/val_loss": "#9333ea",
    }
    legend_y = 30
    for idx, (name, values) in enumerate(nonempty.items()):
        stride = max(math.ceil(len(values) / max(max_points, 1)), 1)
        plotted = values[::stride]
        if plotted[-1] != values[-1]:
            plotted.append(values[-1])
        polyline = " ".join(f"{px:.2f},{py:.2f}" for px, py in (point_to_svg(x, y) for x, y in plotted))
        color = colors[name]
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{polyline}"/>')
        x_legend = 390 + idx * 140
        body.append(f'<line x1="{x_legend}" y1="{legend_y - 5}" x2="{x_legend + 18}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{x_legend + 24}" y="{legend_y}" font-family="sans-serif" font-size="12">{name}</text>')

    Path(output_path).write_text(_svg_document(width, height, "".join(body)), encoding="utf-8")


def write_histogram_svg(values: np.ndarray, output_path: str | Path, bins: int = 40) -> None:
    """Write an SVG histogram for absolute action errors."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    counts, edges = np.histogram(values, bins=max(int(bins), 2))
    width, height = 960, 420
    left, right, top, bottom = 74, 28, 42, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(int(counts.max()), 1)
    bar_w = plot_w / len(counts)
    body = [
        '<text x="24" y="28" font-family="sans-serif" font-size="18">Absolute action error distribution</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for idx, count in enumerate(counts):
        bar_h = count / max_count * plot_h
        x = left + idx * bar_w
        y = top + plot_h - bar_h
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 1, 0):.2f}" height="{bar_h:.2f}" fill="#2563eb"/>')
    body.extend(
        [
            f'<text x="{left}" y="{height - 20}" font-family="sans-serif" font-size="12">0</text>',
            f'<text x="{left + plot_w - 90}" y="{height - 20}" font-family="sans-serif" font-size="12">{edges[-1]:.4g}</text>',
            f'<text x="8" y="{top + 12}" font-family="sans-serif" font-size="12">{max_count}</text>',
            f'<text x="{left + plot_w / 2 - 50}" y="{height - 20}" font-family="sans-serif" font-size="12">absolute error</text>',
        ]
    )
    Path(output_path).write_text(_svg_document(width, height, "".join(body)), encoding="utf-8")
