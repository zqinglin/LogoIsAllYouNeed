#!/usr/bin/env python3
import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regress score gain on AHI-style attention metrics."
    )
    parser.add_argument(
        "--differential_summary_csv",
        type=str,
        default="outputs/differential_analysis/differential_analysis_summary_all_workers.csv",
    )
    parser.add_argument(
        "--scores_original_csv",
        type=str,
        default="outputs/videoscore1_v1_scores_original.csv",
    )
    parser.add_argument(
        "--scores_watermarked_csv",
        type=str,
        default="outputs/videoscore1_v1_scores_watermarked.csv",
    )
    parser.add_argument("--score_col", type=str, default="vs1_total_score")
    parser.add_argument("--frame_agg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/differential_analysis",
    )
    return parser.parse_args()


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fit_ols(y: np.ndarray, x_cols: List[np.ndarray]):
    x = np.column_stack([np.ones(len(y))] + x_cols)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    resid = y - y_hat
    sst = np.sum((y - np.mean(y)) ** 2)
    sse = np.sum(resid ** 2)
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    return beta, y_hat, r2


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_attn = pd.read_csv(args.differential_summary_csv)
    required_attn = [
        "video_filename",
        "frame_idx",
        "ahi",
        "wm_density_ratio",
        "subject_topk_share",
        "competition_ratio",
    ]
    missing = [c for c in required_attn if c not in df_attn.columns]
    if missing:
        raise ValueError(
            "Differential summary CSV is missing required columns: "
            f"{missing}. Re-run analyze_attention_differential.py with updated version."
        )

    df_attn = _safe_numeric(df_attn, required_attn[1:])
    df_attn = df_attn[df_attn["frame_idx"] >= 0].copy()
    metric_cols = ["ahi", "wm_density_ratio", "subject_topk_share", "competition_ratio"]

    if args.frame_agg == "max":
        df_video = df_attn.groupby("video_filename", as_index=False)[metric_cols].max()
    else:
        df_video = df_attn.groupby("video_filename", as_index=False)[metric_cols].mean()

    df_org = pd.read_csv(args.scores_original_csv)
    df_wm = pd.read_csv(args.scores_watermarked_csv)

    score_cols = [
        "vs1_visual_quality",
        "vs1_temporal_consistency",
        "vs1_dynamic_degree",
        "vs1_text_alignment",
        "vs1_factual_consistency",
        "vs1_total_score",
    ]
    keep_cols = ["video_filename"] + [c for c in score_cols if c in df_org.columns and c in df_wm.columns]
    df_org = _safe_numeric(df_org[keep_cols].copy(), keep_cols[1:])
    df_wm = _safe_numeric(df_wm[keep_cols].copy(), keep_cols[1:])

    merged = df_video.merge(df_org, on="video_filename", how="inner", suffixes=("", "_org"))
    merged = merged.merge(df_wm, on="video_filename", how="inner", suffixes=("_org", "_wm"))

    for c in keep_cols[1:]:
        merged[f"delta_{c}"] = merged[f"{c}_wm"] - merged[f"{c}_org"]

    target_col = f"delta_{args.score_col}"
    if target_col not in merged.columns:
        raise ValueError(f"Target delta column not found: {target_col}")

    use_cols = ["ahi", "wm_density_ratio", "subject_topk_share", "competition_ratio", target_col]
    reg_df = merged[use_cols + ["video_filename"]].dropna().copy()
    if len(reg_df) < 10:
        raise ValueError(
            "Too few valid samples for regression: "
            f"{len(reg_df)}. This usually means differential summary rows are cached-only (AHI empty). "
            "Re-run run_differential_attention_8gpu.sh with OVERWRITE=1 to recompute AHI metrics."
        )

    y = reg_df[target_col].to_numpy(dtype=float)
    x1 = reg_df["ahi"].to_numpy(dtype=float)
    x2 = reg_df["wm_density_ratio"].to_numpy(dtype=float)
    x3 = reg_df["subject_topk_share"].to_numpy(dtype=float)
    x4 = reg_df["competition_ratio"].to_numpy(dtype=float)

    beta, y_hat, r2 = fit_ols(y, [x1, x2, x3, x4])

    pearson = float(np.corrcoef(x1, y)[0, 1]) if np.std(x1) > 0 and np.std(y) > 0 else np.nan
    spearman = float(pd.Series(x1).rank().corr(pd.Series(y).rank()))

    # Bin-wise monotonic sanity check for NeurIPS-style narrative.
    reg_df["ahi_bin"] = pd.qcut(reg_df["ahi"], q=min(5, reg_df["ahi"].nunique()), duplicates="drop")
    bin_stats = (
        reg_df.groupby("ahi_bin", observed=True)[target_col]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    merged_out = os.path.join(args.output_dir, "ahi_regression_merged.csv")
    bin_out = os.path.join(args.output_dir, "ahi_regression_bins.csv")
    report_out = os.path.join(args.output_dir, "ahi_regression_report.txt")

    reg_df.to_csv(merged_out, index=False)
    bin_stats.to_csv(bin_out, index=False)

    lines = []
    lines.append("AHI vs Score Gain Regression Report")
    lines.append(f"N={len(reg_df)}")
    lines.append(f"Target={target_col}")
    lines.append(f"Pearson(ahi, target)={pearson:.6f}")
    lines.append(f"Spearman(ahi, target)={spearman:.6f}")
    lines.append("OLS: target ~ 1 + ahi + wm_density_ratio + subject_topk_share + competition_ratio")
    lines.append(f"Intercept={beta[0]:.6f}")
    lines.append(f"beta_ahi={beta[1]:.6f}")
    lines.append(f"beta_wm_density_ratio={beta[2]:.6f}")
    lines.append(f"beta_subject_topk_share={beta[3]:.6f}")
    lines.append(f"beta_competition_ratio={beta[4]:.6f}")
    lines.append(f"R2={r2:.6f}")

    with open(report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved:")
    print(merged_out)
    print(bin_out)
    print(report_out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
