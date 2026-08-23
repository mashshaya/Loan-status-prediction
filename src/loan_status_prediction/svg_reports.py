from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


SVG_STYLE = """
<style>
  text { font-family: Helvetica, Arial, sans-serif; fill: #172026; }
  .title { font-size: 20px; font-weight: 700; }
  .axis { stroke: #83919c; stroke-width: 1; }
  .grid { stroke: #d8e0e6; stroke-width: 1; }
  .line { fill: none; stroke: #0f766e; stroke-width: 3; }
  .baseline { fill: none; stroke: #94a3b8; stroke-width: 2; stroke-dasharray: 6 6; }
  .bar { fill: #2563eb; }
  .label { font-size: 12px; }
  .small { font-size: 11px; fill: #52616b; }
</style>
"""


def write_svg(path: str | Path, body: str, width: int = 760, height: int = 520) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n{SVG_STYLE}\n{body}\n</svg>\n'
    )
    output_path.write_text(svg, encoding="utf-8")
    return str(output_path)


def _scale(value: float, minimum: float, maximum: float, start: float, end: float) -> float:
    if maximum == minimum:
        return (start + end) / 2
    return start + (value - minimum) * (end - start) / (maximum - minimum)


def line_chart_svg(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: str | Path,
    x_label: str,
    y_label: str,
    baseline_diagonal: bool = False,
) -> str:
    width, height = 760, 520
    left, right, top, bottom = 80, 40, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = float(data[x_col].min()), float(data[x_col].max())
    y_min, y_max = float(data[y_col].min()), float(data[y_col].max())
    if x_min >= 0 and x_max <= 1:
        x_min, x_max = 0.0, 1.0
    if y_min >= 0 and y_max <= 1:
        y_min, y_max = 0.0, 1.0

    points = []
    for row in data[[x_col, y_col]].itertuples(index=False):
        x = _scale(float(row[0]), x_min, x_max, left, left + plot_w)
        y = _scale(float(row[1]), y_min, y_max, top + plot_h, top)
        points.append(f"{x:.2f},{y:.2f}")

    grid = []
    for i in range(6):
        gx = left + plot_w * i / 5
        gy = top + plot_h * i / 5
        grid.append(f'<line class="grid" x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" />')
        grid.append(f'<line class="grid" x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" />')

    baseline = ""
    if baseline_diagonal:
        baseline = f'<polyline class="baseline" points="{left},{top + plot_h} {left + plot_w},{top}" />'

    body = f"""
<rect width="{width}" height="{height}" fill="#f8fafc"/>
<text class="title" x="{left}" y="38">{escape(title)}</text>
{''.join(grid)}
<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" />
<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" />
{baseline}
<polyline class="line" points="{' '.join(points)}" />
<text class="label" x="{left + plot_w / 2 - 60}" y="{height - 28}">{escape(x_label)}</text>
<text class="label" transform="translate(24 {top + plot_h / 2 + 50}) rotate(-90)">{escape(y_label)}</text>
<text class="small" x="{left}" y="{top + plot_h + 24}">{x_min:.2f}</text>
<text class="small" x="{left + plot_w - 28}" y="{top + plot_h + 24}">{x_max:.2f}</text>
<text class="small" x="{left - 42}" y="{top + plot_h + 4}">{y_min:.2f}</text>
<text class="small" x="{left - 42}" y="{top + 4}">{y_max:.2f}</text>
"""
    return write_svg(output_path, body, width, height)


def bar_chart_svg(
    data: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output_path: str | Path,
    max_bars: int = 15,
) -> str:
    plot_data = data.head(max_bars).copy()
    width = 900
    row_h = 30
    height = 95 + row_h * len(plot_data)
    left, top, right = 300, 62, 45
    plot_w = width - left - right
    max_value = float(plot_data[value_col].max()) if len(plot_data) else 1.0

    rows = []
    for i, row in enumerate(plot_data.itertuples(index=False)):
        label = str(getattr(row, label_col))
        value = float(getattr(row, value_col))
        y = top + i * row_h
        bar_w = 0 if max_value == 0 else plot_w * value / max_value
        rows.append(
            f'<text class="label" x="20" y="{y + 18}">{escape(label[:42])}</text>'
            f'<rect class="bar" x="{left}" y="{y}" width="{bar_w:.1f}" height="20" rx="4"/>'
            f'<text class="small" x="{left + bar_w + 8}" y="{y + 15}">{value:.4f}</text>'
        )

    body = f"""
<rect width="{width}" height="{height}" fill="#f8fafc"/>
<text class="title" x="20" y="36">{escape(title)}</text>
{''.join(rows)}
"""
    return write_svg(output_path, body, width, height)


def confusion_matrix_svg(matrix: list[list[int]], title: str, output_path: str | Path) -> str:
    width, height = 560, 460
    x0, y0, cell = 150, 95, 140
    max_value = max(max(row) for row in matrix)
    cells = []
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            intensity = 0.18 + 0.72 * value / max_value if max_value else 0.18
            color = f"rgba(37, 99, 235, {intensity:.3f})"
            x, y = x0 + j * cell, y0 + i * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#f8fafc" stroke-width="4"/>'
                f'<text class="label" x="{x + 54}" y="{y + 58}">{labels[i][j]}</text>'
                f'<text class="title" x="{x + 45}" y="{y + 95}">{value}</text>'
            )

    body = f"""
<rect width="{width}" height="{height}" fill="#f8fafc"/>
<text class="title" x="40" y="42">{escape(title)}</text>
<text class="label" x="{x0 + 34}" y="{y0 - 16}">Predicted 0</text>
<text class="label" x="{x0 + cell + 34}" y="{y0 - 16}">Predicted 1</text>
<text class="label" x="58" y="{y0 + 75}">Actual 0</text>
<text class="label" x="58" y="{y0 + cell + 75}">Actual 1</text>
{''.join(cells)}
"""
    return write_svg(output_path, body, width, height)
