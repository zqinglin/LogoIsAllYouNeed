#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


MODEL_NAME_MAP = {
    "videoscore_with_variance": "VideoScore-v1.1",
    "videoscore_v1_with_variance": "VideoScore-v1",
    "videoscore_v1_official_with_variance": "VideoScore-v1-official",
    "internvl2_with_variance": "InternVL2-8B",
    "llava_onevision_with_variance": "LLaVA-OneVision-7B",
    "llava_next_with_variance": "LLaVA-NeXT-Video-7B",
}

METRIC_TITLE_MAP = {
    "visual_quality": "Visual Quality",
    "temporal_consistency": "Temporal Consistency",
    "dynamic_degree": "Dynamic Degree",
    "text_to_video_alignment": "Text-Video Alignment",
    "factual_consistency": "Factual Consistency",
}

EXCLUDED_MODELS = {"VideoScore-v1"}
INLINE_LABEL_MODELS = {"VideoScore-v1.1", "VideoScore-v1-official"}
CODE_ROOT = Path(__file__).resolve().parents[4]


def _parse_model_with_suffix(path: Path, suffix: str) -> str:
    stem = path.stem
    prefix = "results_"
    if not (stem.startswith(prefix) and stem.endswith(suffix)):
        return stem
    raw = stem[len(prefix) : -len(suffix)]
    return MODEL_NAME_MAP.get(raw, raw)


def parse_model_from_alpha_trend_filename(path: Path) -> str:
    return _parse_model_with_suffix(path, "_alpha_trend_all_metrics")


def parse_model_from_original_vs_wm_filename(path: Path) -> str:
    return _parse_model_with_suffix(path, "_original_vs_watermarked")


def load_alpha_trend_tables(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("results_*_alpha_trend_all_metrics.csv"))
    frames = []
    for csv_path in csv_paths:
        if " copy_" in csv_path.name:
            continue
        df = pd.read_csv(csv_path)
        if "Watermark_Alpha" not in df.columns:
            continue
        metric_cols = [c for c in df.columns if c.endswith("_mean")]
        if not metric_cols:
            continue

        long_df = df.melt(
            id_vars=["Watermark_Alpha"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="score",
        )
        long_df["metric"] = long_df["metric"].str.replace("_mean", "", regex=False)
        long_df["model"] = parse_model_from_alpha_trend_filename(csv_path)
        frames.append(long_df)

    if not frames:
        raise RuntimeError(f"No valid alpha trend csv found under: {input_dir}")

    return pd.concat(frames, ignore_index=True)


def load_original_baselines(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("results_*_original_vs_watermarked.csv"))
    frames = []
    for csv_path in csv_paths:
        if " copy_" in csv_path.name:
            continue
        df = pd.read_csv(csv_path)
        if "Is_Original" not in df.columns:
            continue

        original_rows = df[df["Is_Original"].astype(str).str.lower() == "original"]
        if original_rows.empty:
            continue
        original = original_rows.iloc[0]

        metric_cols = [c for c in df.columns if c.endswith("_mean")]
        if not metric_cols:
            continue

        model_name = parse_model_from_original_vs_wm_filename(csv_path)
        for col in metric_cols:
            frames.append(
                {
                    "model": model_name,
                    "metric": col.replace("_mean", ""),
                    "original_score": float(original[col]),
                }
            )

    if not frames:
        raise RuntimeError(f"No valid original baseline csv found under: {input_dir}")
    return pd.DataFrame(frames)


def build_overall_metric(df_long: pd.DataFrame) -> pd.DataFrame:
    overall = (
        df_long.groupby(["Watermark_Alpha", "model"], as_index=False)["score"]
        .mean()
        .assign(metric="overall_mean")
    )
    return pd.concat([df_long, overall], ignore_index=True)


def compute_delta_vs_original(df_all: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    baseline_overall = (
        baselines.groupby("model", as_index=False)["original_score"].mean().assign(metric="overall_mean")
    )
    merged = df_all.merge(
        pd.concat([baselines, baseline_overall], ignore_index=True),
        on=["model", "metric"],
        how="left",
    )
    missing = merged[merged["original_score"].isna()][["model", "metric"]].drop_duplicates()
    if not missing.empty:
        missing_str = ", ".join([f"{r.model}:{r.metric}" for r in missing.itertuples()])
        raise RuntimeError(f"Missing original baselines for: {missing_str}")
    merged["score"] = merged["score"] - merged["original_score"]
    return merged.drop(columns=["original_score"])


def metric_display_name(metric: str) -> str:
    if metric == "overall_mean":
        return "Overall Mean (5 Metrics)"
    return METRIC_TITLE_MAP.get(metric, metric.replace("_", " ").title())


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def draw_plots_per_metric(df_all: pd.DataFrame, output_dir: Path, output_stem: str) -> None:
    _apply_plot_style()

    ordered_metrics = [
        "visual_quality",
        "temporal_consistency",
        "dynamic_degree",
        "text_to_video_alignment",
        "factual_consistency",
        "overall_mean",
    ]
    available_metrics = [m for m in ordered_metrics if m in set(df_all["metric"])]

    # Muted but distinguishable palette suitable for print and projection.
    palette = ["#0D3B66", "#EE964B", "#1982C4", "#2A9D8F", "#6A4C93", "#C1121F", "#5F6C7B"]
    model_order = sorted(df_all["model"].unique())
    color_map = {model: palette[i % len(palette)] for i, model in enumerate(model_order)}
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X"]
    marker_map = {model: marker_cycle[i % len(marker_cycle)] for i, model in enumerate(model_order)}

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        data = df_all[df_all["metric"] == metric].sort_values("Watermark_Alpha")

        ymin = data["score"].min()
        ymax = data["score"].max()
        pad = max((ymax - ymin) * 0.22, 0.05)

        for model in model_order:
            series = data[data["model"] == model]
            if series.empty:
                continue
            ax.plot(
                series["Watermark_Alpha"],
                series["score"],
                marker=marker_map[model],
                markersize=5,
                linewidth=2.4,
                alpha=0.96,
                color=color_map[model],
                label=model,
            )

            # Keep terminal markers but avoid direct text labels to prevent overlap.
            x_last = float(series["Watermark_Alpha"].iloc[-1])
            y_last = float(series["score"].iloc[-1])
            ax.scatter([x_last], [y_last], s=30, color=color_map[model], zorder=5)

        # Add inline labels only for key VideoScore variants for quick reviewer recognition.
        label_candidates = []
        for model in model_order:
            if model not in INLINE_LABEL_MODELS:
                continue
            series = data[data["model"] == model]
            if series.empty:
                continue
            label_candidates.append((model, float(series["score"].iloc[-1])))

        label_candidates.sort(key=lambda x: x[1])
        adjusted_positions = {}
        min_sep = 0.008
        for model, y in label_candidates:
            if not adjusted_positions:
                adjusted_positions[model] = y
                continue
            prev_model = list(adjusted_positions.keys())[-1]
            prev_y = adjusted_positions[prev_model]
            adjusted_positions[model] = y if (y - prev_y) >= min_sep else prev_y + min_sep

        for model, y in label_candidates:
            y_text = adjusted_positions.get(model, y)
            ax.text(
                1.02,
                y_text,
                model,
                color=color_map[model],
                fontsize=10,
                va="center",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
            )

        ax.axhline(0.0, color="#111111", linewidth=1.3, linestyle="-", alpha=0.75)
        ax.set_xlabel("Watermark Alpha")
        ax.set_ylabel("Delta Score")
        ax.set_xlim(0.08, 1.08)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, axis="both")

        handles = [
            Line2D(
                [0],
                [0],
                color=color_map[m],
                marker=marker_map[m],
                linewidth=2.4,
                markersize=5,
                label=m,
            )
            for m in model_order
        ]
        ax.legend(
            handles=handles,
            labels=model_order,
            loc="upper center",
            ncol=min(3, len(model_order)),
            bbox_to_anchor=(0.5, -0.18),
            fontsize=10,
        )

        fig.tight_layout()

        metric_tag = metric.replace("_", "-")
        png_path = output_dir / f"{output_stem}_{metric_tag}.png"
        pdf_path = output_dir / f"{output_stem}_{metric_tag}.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot alpha experiment delta trends vs original as publication-quality line charts.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=CODE_ROOT / "outputs/LargeScaleEval/detailed_analysis",
        help="Directory containing results_*_alpha_trend_all_metrics.csv",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=CODE_ROOT / "outputs/LargeScaleEval/detailed_analysis/fig_alpha_delta_all_models",
        help="Output file prefix. Each metric will be exported as {prefix}_{metric}.png/.pdf",
    )
    args = parser.parse_args()

    df = load_alpha_trend_tables(args.input_dir)
    df = df[~df["model"].isin(EXCLUDED_MODELS)].copy()
    if df.empty:
        raise RuntimeError("No data left after excluding models.")
    df_all = build_overall_metric(df)
    baselines = load_original_baselines(args.input_dir)
    baselines = baselines[~baselines["model"].isin(EXCLUDED_MODELS)].copy()
    if baselines.empty:
        raise RuntimeError("No baseline data left after excluding models.")
    df_delta = compute_delta_vs_original(df_all, baselines)

    output_dir = args.output_prefix.parent
    output_stem = args.output_prefix.name
    draw_plots_per_metric(df_delta, output_dir, output_stem)


if __name__ == "__main__":
    main()
