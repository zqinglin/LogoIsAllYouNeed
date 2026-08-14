import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS: List[str] = [
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_to_video_alignment",
    "factual_consistency",
]

METRIC_LABELS: Dict[str, str] = {
    "visual_quality": "Visual Quality",
    "temporal_consistency": "Temporal Consistency",
    "dynamic_degree": "Dynamic Degree",
    "text_to_video_alignment": "Text-Video Alignment",
    "factual_consistency": "Factual Consistency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-style figures for Prompt Resistance experiment."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="outputs/prompt_resistance_v1.1_pairs.csv",
        help="Detailed pairwise CSV from prompt_resistance_pair_v1_1.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/prompt_resistance_plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=3.35,
        help="Figure width in inches, tuned for single-column layout.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=3.15,
        help="Figure height in inches, near-square for paper readability.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def mean_and_ci95(values: np.ndarray) -> tuple:
    if values.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    sem = np.std(values, ddof=1) / np.sqrt(values.size)
    return mean, 1.96 * sem


def plot_delta_pair(ax: plt.Axes, delta_base: np.ndarray, delta_immune: np.ndarray, y_label: str) -> None:
    # Conference-style muted palette.
    base_color = "#3F5A9A"
    immune_color = "#A55D3A"
    line_color = "#9AA2B2"

    x_base = np.zeros_like(delta_base, dtype=float)
    x_immune = np.ones_like(delta_immune, dtype=float)

    rng = np.random.default_rng(2026)
    jitter_base = rng.normal(0.0, 0.04, size=delta_base.shape[0])
    jitter_immune = rng.normal(0.0, 0.04, size=delta_immune.shape[0])

    for i in range(min(delta_base.shape[0], delta_immune.shape[0])):
        ax.plot(
            [x_base[i] + jitter_base[i], x_immune[i] + jitter_immune[i]],
            [delta_base[i], delta_immune[i]],
            color=line_color,
            alpha=0.18,
            linewidth=0.55,
            zorder=1,
        )

    ax.scatter(
        x_base + jitter_base,
        delta_base,
        s=9,
        color=base_color,
        alpha=0.62,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        x_immune + jitter_immune,
        delta_immune,
        s=9,
        color=immune_color,
        alpha=0.62,
        edgecolors="none",
        zorder=2,
    )

    mean_b, ci_b = mean_and_ci95(delta_base)
    mean_i, ci_i = mean_and_ci95(delta_immune)

    ax.errorbar(
        [0, 1],
        [mean_b, mean_i],
        yerr=[ci_b, ci_i],
        fmt="o",
        markersize=4.2,
        color="black",
        ecolor="black",
        elinewidth=1.05,
        capsize=2.5,
        zorder=3,
    )

    ax.axhline(0.0, color="#6E7685", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base Prompt", "Immunity Prompt"])
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#D7DCE5", linewidth=0.6, alpha=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_metric_figure(
    metric: str,
    delta_base: np.ndarray,
    delta_immune: np.ndarray,
    out_dir: str,
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ylabel = f"Score Drift ({METRIC_LABELS[metric]})"
    plot_delta_pair(ax, delta_base, delta_immune, ylabel)

    stem = f"prompt_resistance_{metric}_paired"
    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")

    fig.savefig(png_path, dpi=dpi, transparent=False)
    fig.savefig(pdf_path, transparent=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    apply_style()

    df = pd.read_csv(args.input_csv)

    for metric in METRICS:
        col_base = f"delta_base_{metric}"
        col_immune = f"delta_immune_{metric}"

        if col_base not in df.columns or col_immune not in df.columns:
            print(f"Skip metric '{metric}': missing required columns.")
            continue

        sub = df[[col_base, col_immune]].dropna()
        if sub.empty:
            print(f"Skip metric '{metric}': no valid paired rows.")
            continue

        delta_base = sub[col_base].to_numpy(dtype=float)
        delta_immune = sub[col_immune].to_numpy(dtype=float)

        save_metric_figure(
            metric=metric,
            delta_base=delta_base,
            delta_immune=delta_immune,
            out_dir=args.output_dir,
            fig_w=args.fig_width,
            fig_h=args.fig_height,
            dpi=args.dpi,
        )

    print("Done. Figures exported (PNG + PDF):")
    print(args.output_dir)


if __name__ == "__main__":
    main()
