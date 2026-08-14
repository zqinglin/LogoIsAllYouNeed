import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


INPUT_CSV = "outputs/comparison_report.csv"
OUTPUT_PNG = "outputs/plots/sora_rank_consistency_alluvial_quantile.png"
OUTPUT_PDF = "outputs/plots/sora_rank_consistency_alluvial_quantile.pdf"
OUTPUT_BASELINE_PNG = "outputs/plots/sora_rank_consistency_alluvial_quantile_baseline.png"
OUTPUT_BASELINE_PDF = "outputs/plots/sora_rank_consistency_alluvial_quantile_baseline.pdf"

N_QUANTILE_BINS = 8

GENERATOR_COLORS = {
    "alpha": "#A8DADC",
    "hunyuan": "#FFD6A5",
    "pika": "#B8E0D2",
    "ray2": "#FFB4A2",
    "sora": "#CDB4DB",
}

FIG_BG = "#FFFCF8"
LEFT_BAR = "#C8D1DD"
RIGHT_BAR = "#DDE3EC"
TEXT_MAIN = "#0B1220"
TEXT_SUB = "#374151"
EDGE_SOFT = "#E5E7EB"


@dataclass
class FlowSeg:
    left_bin: str
    right_bin: str
    generator: str
    value: float
    y0l: float
    y1l: float
    y0r: float
    y1r: float


def parse_generator(video_filename: str) -> str:
    # Examples: 0000_hunyuan_1724.mp4, 0000_ray2_1.mp4, 0000_sora_0.mp4
    m = re.match(r"^\d+_([^_]+)_", str(video_filename))
    if m:
        return m.group(1)
    return "unknown"


def build_quantile_edges(scores: pd.Series, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(scores.to_numpy(), q)
    edges = np.unique(np.round(edges, 6))

    if len(edges) < 3:
        edges = np.quantile(scores.to_numpy(), q)
        edges = np.unique(np.round(edges, 6))

    if len(edges) < 3:
        raise RuntimeError("Not enough score variation to build quantile bins.")

    global_min = float(scores.min())
    global_max = float(scores.max())
    edges[0] = min(edges[0], global_min) - 1e-6
    edges[-1] = max(edges[-1], global_max) + 1e-6
    return edges


def quantile_labels(edges: np.ndarray, labels: List[str]) -> Dict[str, str]:
    display_map: Dict[str, str] = {}
    for i, lb in enumerate(labels):
        l, r = edges[i], edges[i + 1]
        bracket = "]" if i == len(labels) - 1 else ")"
        display_map[lb] = f"{lb}\n[{l:.2f}, {r:.2f}{bracket}"
    return display_map


def build_shared_quantile_labels(orig_edges: np.ndarray, wm_edges: np.ndarray) -> List[str]:
    n_bins = min(len(orig_edges) - 1, len(wm_edges) - 1)
    return [f"Q{i+1}" for i in range(n_bins)]


def trim_edges(edges: np.ndarray, n_bins: int) -> np.ndarray:
    if len(edges) - 1 == n_bins:
        return edges
    return np.quantile(np.linspace(edges[0], edges[-1], 2048), np.linspace(0.0, 1.0, n_bins + 1))


def quantile_labels_pair(orig_edges: np.ndarray, wm_edges: np.ndarray) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    labels = build_shared_quantile_labels(orig_edges, wm_edges)
    n_bins = len(labels)
    orig_e = trim_edges(orig_edges, n_bins)
    wm_e = trim_edges(wm_edges, n_bins)

    left_display = quantile_labels(orig_e, labels)
    right_display = quantile_labels(wm_e, labels)
    return labels, left_display, right_display


def bezier_band(
    x0: float,
    x1: float,
    y0l: float,
    y1l: float,
    y0r: float,
    y1r: float,
    curvature: float = 0.33,
) -> Path:
    cx0 = x0 + (x1 - x0) * curvature
    cx1 = x1 - (x1 - x0) * curvature

    verts = [
        (x0, y1l),
        (cx0, y1l),
        (cx1, y1r),
        (x1, y1r),
        (x1, y0r),
        (cx1, y0r),
        (cx0, y0l),
        (x0, y0l),
        (x0, y1l),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def compute_layout(df_flow: pd.DataFrame, left_order: List[str], right_order: List[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]], List[FlowSeg]]:
    total = float(df_flow["value"].sum())
    gap = 0.008
    usable = 1.0 - gap * (max(len(left_order), len(right_order)) - 1)

    left_totals = (
        df_flow.groupby("orig_bin", as_index=False, observed=False)["value"]
        .sum()
        .set_index("orig_bin")["value"]
        .to_dict()
    )
    right_totals = (
        df_flow.groupby("wm_bin", as_index=False, observed=False)["value"]
        .sum()
        .set_index("wm_bin")["value"]
        .to_dict()
    )

    # Align left/right bins to the same vertical score bands to make drift direction more readable.
    aligned_totals = {
        b: max(left_totals.get(b, 0.0), right_totals.get(b, 0.0)) for b in left_order
    }

    y_top = 1.0
    aligned_pos: Dict[str, Tuple[float, float]] = {}
    for key in left_order:
        h = (aligned_totals.get(key, 0.0) / total) * usable if total > 0 else 0.0
        y1 = y_top
        y0 = y1 - h
        aligned_pos[key] = (y0, y1)
        y_top = y0 - gap

    left_pos = dict(aligned_pos)
    right_pos = dict(aligned_pos)

    # Allocate segments inside each bin to reduce visual clutter.
    left_cursor = {}
    right_cursor = {}
    for k in left_order:
        y0, y1 = aligned_pos[k]
        container_h = y1 - y0
        left_h = (left_totals.get(k, 0.0) / total) * usable if total > 0 else 0.0
        right_h = (right_totals.get(k, 0.0) / total) * usable if total > 0 else 0.0
        left_cursor[k] = y0 + max((container_h - left_h) / 2, 0.0)
        right_cursor[k] = y0 + max((container_h - right_h) / 2, 0.0)
    flows: List[FlowSeg] = []

    left_rank = {k: i for i, k in enumerate(left_order)}
    right_rank = {k: i for i, k in enumerate(right_order)}

    # Sort by left bin then right bin then generator for consistent deterministic layout.
    for row in df_flow.sort_values(["orig_bin", "wm_bin", "generator"], key=lambda s: s.map({**left_rank, **right_rank}).fillna(0) if s.name in ["orig_bin", "wm_bin"] else s).itertuples(index=False):
        h = (float(row.value) / total) * usable if total > 0 else 0.0

        y0l = left_cursor[row.orig_bin]
        y1l = y0l + h
        left_cursor[row.orig_bin] = y1l

        y0r = right_cursor[row.wm_bin]
        y1r = y0r + h
        right_cursor[row.wm_bin] = y1r

        flows.append(
            FlowSeg(
                left_bin=row.orig_bin,
                right_bin=row.wm_bin,
                generator=row.generator,
                value=float(row.value),
                y0l=y0l,
                y1l=y1l,
                y0r=y0r,
                y1r=y1r,
            )
        )

    return left_pos, right_pos, flows


def render_alluvial(
    flows: List[FlowSeg],
    left_pos: Dict[str, Tuple[float, float]],
    right_pos: Dict[str, Tuple[float, float]],
    left_order: List[str],
    right_order: List[str],
    left_display_map: Dict[str, str],
    right_display_map: Dict[str, str],
    generators: List[str],
    output_png: str,
    output_pdf: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16.0, 8.2), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)

    x_left = 0.16
    x_right = 0.84
    bar_w = 0.022

    for f in flows:
        color = GENERATOR_COLORS.get(f.generator, "#9CA3AF")
        path = bezier_band(
            x_left + bar_w,
            x_right,
            f.y0l,
            f.y1l,
            f.y0r,
            f.y1r,
            curvature=0.36,
        )
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=0.42))

    for lb in left_order:
        y0, y1 = left_pos[lb]
        ax.add_patch(
            Rectangle(
                (x_left, y0),
                bar_w,
                y1 - y0,
                facecolor=LEFT_BAR,
                edgecolor=EDGE_SOFT,
                lw=1.2,
            )
        )
        ax.text(
            x_left - 0.022,
            (y0 + y1) / 2,
            left_display_map.get(lb, lb),
            ha="right",
            va="center",
            fontsize=12,
            color=TEXT_MAIN,
            fontweight="medium",
            linespacing=1.15,
        )

    for rb in right_order:
        y0, y1 = right_pos[rb]
        ax.add_patch(
            Rectangle(
                (x_right, y0),
                bar_w,
                y1 - y0,
                facecolor=RIGHT_BAR,
                edgecolor=EDGE_SOFT,
                lw=1.2,
            )
        )
        ax.text(
            x_right + bar_w + 0.022,
            (y0 + y1) / 2,
            right_display_map.get(rb, rb),
            ha="left",
            va="center",
            fontsize=12,
            color=TEXT_MAIN,
            fontweight="medium",
            linespacing=1.15,
        )

    ax.text(
        x_left + bar_w / 2,
        -0.035,
        "Original Score Quantile Interval",
        ha="center",
        va="top",
        fontsize=14.5,
        color=TEXT_SUB,
        fontweight="semibold",
    )
    ax.text(
        x_right + bar_w / 2,
        -0.035,
        "Watermarked Score Quantile Interval",
        ha="center",
        va="top",
        fontsize=14.5,
        color=TEXT_SUB,
        fontweight="semibold",
    )

    handles = []
    labels_legend = []
    for gen in generators:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=GENERATOR_COLORS.get(gen, "#9CA3AF"), edgecolor="none", alpha=0.65))
        labels_legend.append(gen)
    ax.legend(
        handles,
        labels_legend,
        title="Generator",
        frameon=True,
        fancybox=False,
        edgecolor=EDGE_SOFT,
        facecolor=FIG_BG,
        ncol=min(5, len(labels_legend)),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.155),
        fontsize=12.5,
        title_fontsize=13.5,
        borderpad=0.55,
        columnspacing=1.25,
        handlelength=1.6,
    )

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.048, 1.01)
    ax.axis("off")

    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(output_png, dpi=420, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise RuntimeError(f"Input csv is empty: {INPUT_CSV}")

    required_cols = {"video_filename", "orig_total", "wm_total"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in input CSV: {sorted(missing)}")

    df = df.dropna(subset=["orig_total", "wm_total"]).copy()
    df["generator"] = df["video_filename"].apply(parse_generator)

    orig_edges = build_quantile_edges(df["orig_total"], N_QUANTILE_BINS)
    wm_edges = build_quantile_edges(df["wm_total"], N_QUANTILE_BINS)
    labels, left_display_map, right_display_map = quantile_labels_pair(orig_edges, wm_edges)

    n_bins = len(labels)
    orig_edges = trim_edges(orig_edges, n_bins)
    wm_edges = trim_edges(wm_edges, n_bins)

    df["orig_bin"] = pd.cut(df["orig_total"], bins=orig_edges, labels=labels, include_lowest=True, right=True)
    df["wm_bin"] = pd.cut(df["wm_total"], bins=wm_edges, labels=labels, include_lowest=True, right=True)
    df = df.dropna(subset=["orig_bin", "wm_bin"])

    generators = sorted(df["generator"].unique())

    # Actual observed crossing pattern.
    df_flow_actual = (
        df.groupby(["orig_bin", "wm_bin", "generator"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "value"})
    )
    # Show higher-score quantiles at top, lower at bottom for more intuitive trend reading.
    left_order = labels[::-1]
    right_order = labels[::-1]
    left_pos, right_pos, flows_actual = compute_layout(df_flow_actual, left_order, right_order)
    render_alluvial(
        flows_actual,
        left_pos,
        right_pos,
        left_order,
        right_order,
        left_display_map,
        right_display_map,
        generators,
        OUTPUT_PNG,
        OUTPUT_PDF,
    )

    # No-crossing baseline: keep each sample in its original quantile bucket.
    df_flow_baseline = (
        df.groupby(["orig_bin", "generator"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "value"})
    )
    df_flow_baseline["wm_bin"] = df_flow_baseline["orig_bin"]
    df_flow_baseline = df_flow_baseline[["orig_bin", "wm_bin", "generator", "value"]]

    left_pos_b, right_pos_b, flows_baseline = compute_layout(df_flow_baseline, left_order, right_order)
    render_alluvial(
        flows_baseline,
        left_pos_b,
        right_pos_b,
        left_order,
        right_order,
        left_display_map,
        right_display_map,
        generators,
        OUTPUT_BASELINE_PNG,
        OUTPUT_BASELINE_PDF,
    )


if __name__ == "__main__":
    main()
