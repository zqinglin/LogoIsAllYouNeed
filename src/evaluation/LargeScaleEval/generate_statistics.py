import glob
import os
import re

import numpy as np
import pandas as pd
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = os.environ.get("RESULTS_DIR", str(CODE_ROOT / "outputs/LargeScaleEval"))
PLOTS_DIR = os.path.join(RESULTS_DIR, "detailed_analysis")


def extract_generator_model(video_name):
    base_name = re.sub(r"_alpha_([0-1]\.\d+)\.mp4$", ".mp4", video_name)
    base_name = base_name.replace("_sora_watermark", "")
    parts = base_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def extract_watermark_alpha(video_name):
    # True alpha watermark suffix only: *_alpha_0.1.mp4 ... *_alpha_1.0.mp4
    match = re.search(r"_alpha_([0-1]\.\d+)\.mp4$", video_name)
    if match:
        return float(match.group(1))
    if "_sora_watermark" in video_name:
        return 1.0
    return 0.0


def is_original(video_name):
    # Original means no explicit watermark suffix and not sora watermark variant.
    if re.search(r"_alpha_([0-1]\.\d+)\.mp4$", video_name):
        return False
    if "_sora_watermark" in video_name:
        return False
    return True


def watermark_type(video_name):
    if re.search(r"_alpha_([0-1]\.\d+)\.mp4$", video_name):
        return "alpha_gradient"
    if "_sora_watermark" in video_name:
        return "sora_dynamic"
    return "original"


def sample_id(video_name):
    # Use first token as sample id (e.g., 0001_*) for coverage checks.
    return video_name.split("_")[0]


def metric_short(c):
    return c.replace("_mean", "")


def build_growth_vs_original(df, score_cols):
    df_orig = df[df["Is_Original"]]
    df_wm = df[~df["Is_Original"]]
    if df_orig.empty or df_wm.empty:
        return pd.DataFrame()

    orig_by_gen = df_orig.groupby("Generator")[score_cols].mean()
    wm_by_gen_alpha = df_wm.groupby(["Generator", "Watermark_Alpha"])[score_cols].mean()

    rows = []
    for (gen, alpha), row in wm_by_gen_alpha.iterrows():
        if gen not in orig_by_gen.index:
            continue
        base = orig_by_gen.loc[gen]
        out = {"Generator": gen, "Watermark_Alpha": alpha}
        for c in score_cols:
            out[f"{metric_short(c)}_delta_vs_original"] = row[c] - base[c]
            if base[c] != 0 and not np.isnan(base[c]):
                out[f"{metric_short(c)}_pct_vs_original"] = (row[c] - base[c]) / abs(base[c]) * 100.0
            else:
                out[f"{metric_short(c)}_pct_vs_original"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def build_slope_table(df, score_cols):
    df_wm = df[~df["Is_Original"]]
    if df_wm.empty:
        return pd.DataFrame()

    rows = []
    for gen, gdf in df_wm.groupby("Generator"):
        out = {"Generator": gen}
        for c in score_cols:
            agg = gdf.groupby("Watermark_Alpha")[c].mean().dropna()
            if len(agg) >= 2:
                x = agg.index.values.astype(float)
                y = agg.values.astype(float)
                slope = np.polyfit(x, y, 1)[0]
                out[f"{metric_short(c)}_slope"] = slope
                out[f"{metric_short(c)}_delta_0p1_to_1p0"] = (
                    agg.loc[1.0] - agg.loc[0.1] if 0.1 in agg.index and 1.0 in agg.index else np.nan
                )
            else:
                out[f"{metric_short(c)}_slope"] = np.nan
                out[f"{metric_short(c)}_delta_0p1_to_1p0"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_csv(file_path):
    df = pd.read_csv(file_path)
    df["Generator"] = df["video"].apply(extract_generator_model)
    df["Watermark_Alpha"] = df["video"].apply(extract_watermark_alpha)
    df["Is_Original"] = df["video"].apply(is_original)
    df["Watermark_Type"] = df["video"].apply(watermark_type)
    df["Sample_ID"] = df["video"].apply(sample_id)

    score_cols = [c for c in df.columns if c.endswith("_mean")]
    if not score_cols:
        score_cols = [
            "visual_quality",
            "temporal_consistency",
            "dynamic_degree",
            "text_to_video_alignment",
            "factual_consistency",
        ]
        score_cols = [c for c in score_cols if c in df.columns]

    std_cols = [c.replace("_mean", "_std") for c in score_cols if c.replace("_mean", "_std") in df.columns]
    base_out = file_path.replace(".csv", "")
    out_prefix = os.path.basename(base_out)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n{'=' * 80}\nANALYSIS FOR: {os.path.basename(file_path)}\n{'=' * 80}")

    # 1) Coverage and sanity
    print("\n--- 1. COVERAGE & SANITY CHECK ---")
    coverage = pd.DataFrame({
        "total_rows": [len(df)],
        "unique_videos": [df["video"].nunique()],
        "unique_sample_ids": [df["Sample_ID"].nunique()],
        "original_rows": [int(df["Is_Original"].sum())],
        "watermarked_rows": [int((~df["Is_Original"]).sum())],
    })
    type_counts = df["Watermark_Type"].value_counts().rename("count").to_frame()
    gen_counts = df.groupby(["Generator", "Watermark_Type"]).size().rename("count").reset_index()
    print(coverage.to_string(index=False))
    print("Watermark type counts:")
    print(type_counts.to_string())

    # 2) Overall metric summary
    print("\n--- 2. OVERALL METRICS (MEAN ± AVG_STD) ---")
    overall_mean = df[score_cols].mean(numeric_only=True)
    overall_std_mean = df[std_cols].mean(numeric_only=True) if std_cols else pd.Series(dtype=float)
    for c in score_cols:
        m = overall_mean[c]
        s_col = c.replace("_mean", "_std")
        s = overall_std_mean[s_col] if s_col in overall_std_mean else np.nan
        if np.isnan(s):
            print(f"{metric_short(c)}: {m:.3f}")
        else:
            print(f"{metric_short(c)}: {m:.3f} ± {s:.3f}")

    # 3) Original vs Watermarked
    print("\n--- 3. ORIGINAL VS WATERMARKED ---")
    orig_vs_wm = df.groupby("Is_Original")[score_cols].mean().round(3)
    orig_vs_wm.index = orig_vs_wm.index.map({True: "Original", False: "Watermarked"})
    print(orig_vs_wm.to_string())

    # 4) Generator-specific summary
    print("\n--- 4. GENERATOR-SPECIFIC SUMMARY ---")
    original_by_gen = df[df["Is_Original"]].groupby("Generator")[score_cols].mean().round(3)
    wm_by_gen = df[~df["Is_Original"]].groupby("Generator")[score_cols].mean().round(3)
    if not original_by_gen.empty:
        print("Original by generator:")
        print(original_by_gen.to_string())
    else:
        print("No original rows found.")
    if not wm_by_gen.empty:
        print("Watermarked by generator:")
        print(wm_by_gen.to_string())

    # 5) Alpha trends
    print("\n--- 5. ALPHA TREND (ALL GENERATORS, WATERMARKED ONLY) ---")
    wm_df = df[~df["Is_Original"]]
    alpha_trend = wm_df.groupby("Watermark_Alpha")[score_cols].mean().round(3)
    print(alpha_trend.to_string())

    # 6) Cross-tab visual quality and full generator-alpha table
    print("\n--- 6. GENERATOR x ALPHA CROSS-TAB ---")
    vq_col = [c for c in score_cols if "visual_quality" in c]
    vq_col = vq_col[0] if vq_col else score_cols[0]
    pivot_vq = pd.pivot_table(
        wm_df,
        values=vq_col,
        index="Generator",
        columns="Watermark_Alpha",
        aggfunc="mean",
    ).round(3)
    print(pivot_vq.to_string())

    gen_alpha_all_metrics = wm_df.groupby(["Generator", "Watermark_Alpha"])[score_cols].mean().round(4)

    # 7) Growth analysis (score increase / decrease)
    print("\n--- 7. GROWTH ANALYSIS (VS ORIGINAL) ---")
    growth_vs_original = build_growth_vs_original(df, score_cols)
    slope_table = build_slope_table(df, score_cols)
    if not growth_vs_original.empty:
        # Print concise leaderboard using visual quality
        col = "visual_quality_delta_vs_original"
        if col in growth_vs_original.columns:
            leaderboard = (
                growth_vs_original.groupby("Generator")[col]
                .mean()
                .sort_values(ascending=False)
                .round(3)
                .to_frame(name="avg_visual_quality_delta_vs_original")
            )
            print("Average visual_quality delta vs original by generator:")
            print(leaderboard.to_string())
    else:
        print("Not enough data to build growth-vs-original tables.")

    # 8) Export all detailed tables
    coverage.to_csv(f"{PLOTS_DIR}/{out_prefix}_coverage_summary.csv", index=False)
    type_counts.to_csv(f"{PLOTS_DIR}/{out_prefix}_watermark_type_counts.csv")
    gen_counts.to_csv(f"{PLOTS_DIR}/{out_prefix}_generator_type_counts.csv", index=False)
    orig_vs_wm.to_csv(f"{PLOTS_DIR}/{out_prefix}_original_vs_watermarked.csv")
    if not original_by_gen.empty:
        original_by_gen.to_csv(f"{PLOTS_DIR}/{out_prefix}_original_by_generator.csv")
    if not wm_by_gen.empty:
        wm_by_gen.to_csv(f"{PLOTS_DIR}/{out_prefix}_watermarked_by_generator.csv")
    alpha_trend.to_csv(f"{PLOTS_DIR}/{out_prefix}_alpha_trend_all_metrics.csv")
    pivot_vq.to_csv(f"{PLOTS_DIR}/{out_prefix}_pivot_generator_alpha_visual_quality.csv")
    gen_alpha_all_metrics.to_csv(f"{PLOTS_DIR}/{out_prefix}_generator_alpha_all_metrics.csv")
    if not growth_vs_original.empty:
        growth_vs_original.round(4).to_csv(f"{PLOTS_DIR}/{out_prefix}_growth_vs_original.csv", index=False)
    if not slope_table.empty:
        slope_table.round(6).to_csv(f"{PLOTS_DIR}/{out_prefix}_alpha_slope_by_generator.csv", index=False)

    print(f"\n(Detailed tables exported to: {PLOTS_DIR})")


def main():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.csv"))
    csv_files = [f for f in csv_files if "summary" not in f and "pivot" not in f and "trend" not in f]
    csv_files = [f for f in csv_files if "detailed_analysis" not in f]

    print(f"Found {len(csv_files)} evaluation files.")
    for f in csv_files:
        summarize_csv(f)


if __name__ == "__main__":
    main()
