#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore multiple AHI variants against score gain.")
    parser.add_argument(
        "--differential_summary_wm_csv",
        type=str,
        default="outputs/differential_analysis/differential_analysis_summary_all_workers.csv",
    )
    parser.add_argument(
        "--differential_summary_org_csv",
        type=str,
        default="",
        help="Optional original-video differential summary CSV for delta features.",
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


def _extract_source(filename: str) -> str:
    m = re.search(r"_(alpha|hunyuan|pika|ray2|sora)_", str(filename))
    return m.group(1) if m else "other"


def _extract_prompt_id(filename: str) -> str:
    m = re.match(r"^(\d+)_", str(filename))
    return m.group(1) if m else "unknown"


def build_video_features(df_attn: pd.DataFrame, prefix: str) -> pd.DataFrame:
    base_cols = ["video_filename", "frame_idx", "ahi", "subject_topk_share", "competition_ratio", "wm_density_ratio"]
    missing = [c for c in base_cols if c not in df_attn.columns]
    if missing:
        raise ValueError(f"Missing required columns in differential summary: {missing}")

    df = _safe_numeric(df_attn.copy(), [c for c in base_cols if c != "video_filename"])
    df = df[df["frame_idx"] >= 0].copy()

    g = df.groupby("video_filename", as_index=False).agg(
        ahi_mean=("ahi", "mean"),
        ahi_std=("ahi", "std"),
        ahi_max=("ahi", "max"),
        ahi_p90=("ahi", lambda s: float(np.nanpercentile(s, 90))),
        subject_mean=("subject_topk_share", "mean"),
        subject_std=("subject_topk_share", "std"),
        subject_max=("subject_topk_share", "max"),
        comp_mean=("competition_ratio", "mean"),
        comp_std=("competition_ratio", "std"),
        wm_density_mean=("wm_density_ratio", "mean"),
        wm_density_std=("wm_density_ratio", "std"),
        frame_count=("frame_idx", "count"),
    )

    eps = 1e-12
    g["ahi_std"] = g["ahi_std"].fillna(0.0)
    g["subject_std"] = g["subject_std"].fillna(0.0)
    g["comp_std"] = g["comp_std"].fillna(0.0)
    g["wm_density_std"] = g["wm_density_std"].fillna(0.0)

    g["ahi_cv"] = g["ahi_std"] / (g["ahi_mean"] + eps)
    g["ahi_stable"] = g["ahi_mean"] * (1.0 / (1.0 + g["ahi_cv"]))
    g["cahi_mean"] = g["ahi_mean"] / (g["ahi_mean"] + g["subject_mean"] + eps)
    g["ahi_margin"] = g["ahi_mean"] - g["subject_mean"]

    rename_map = {c: f"{prefix}{c}" for c in g.columns if c != "video_filename"}
    return g.rename(columns=rename_map)


def fit_ols(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    x2 = np.column_stack([np.ones(len(y)), x])
    beta, _, _, _ = np.linalg.lstsq(x2, y, rcond=None)
    yhat = x2 @ beta
    sst = np.sum((y - np.mean(y)) ** 2)
    sse = np.sum((y - yhat) ** 2)
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    return beta, float(r2)


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def within_group_corr(df: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> float:
    parts = []
    for _, g in df.groupby(group_col):
        if len(g) < 2:
            continue
        x = g[x_col] - g[x_col].mean()
        y = g[y_col] - g[y_col].mean()
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(tmp) >= 2:
            parts.append(tmp)
    if not parts:
        return np.nan
    z = pd.concat(parts, ignore_index=True)
    return corr_safe(z["x"].to_numpy(dtype=float), z["y"].to_numpy(dtype=float))


def evaluate_feature(df: pd.DataFrame, feat: str, y_col: str) -> Dict[str, float]:
    tmp = df[[feat, y_col, "source", "prompt_id"]].dropna().copy()
    if len(tmp) < 10:
        return {
            "feature": feat,
            "n": len(tmp),
            "pearson": np.nan,
            "spearman": np.nan,
            "slope": np.nan,
            "r2_simple": np.nan,
            "corr_within_source": np.nan,
            "corr_within_prompt": np.nan,
        }

    x = tmp[feat].to_numpy(dtype=float)
    y = tmp[y_col].to_numpy(dtype=float)
    pearson = corr_safe(x, y)
    spearman = float(tmp[feat].rank().corr(tmp[y_col].rank()))
    beta, r2 = fit_ols(y, x.reshape(-1, 1))

    return {
        "feature": feat,
        "n": len(tmp),
        "pearson": pearson,
        "spearman": spearman,
        "slope": float(beta[1]),
        "r2_simple": r2,
        "corr_within_source": within_group_corr(tmp, feat, y_col, "source"),
        "corr_within_prompt": within_group_corr(tmp, feat, y_col, "prompt_id"),
    }


def evaluate_multivariate(df: pd.DataFrame, y_col: str, feature_sets: Sequence[Sequence[str]]) -> pd.DataFrame:
    rows = []
    for feats in feature_sets:
        cols = list(feats) + [y_col, "source", "prompt_id"]
        tmp = df[cols].dropna().copy()
        if len(tmp) < 20:
            continue

        y = tmp[y_col].to_numpy(dtype=float)
        x = tmp[list(feats)].to_numpy(dtype=float)
        _, r2_plain = fit_ols(y, x)

        # Fixed effects by source and prompt_id.
        d_source = pd.get_dummies(tmp["source"], prefix="src", drop_first=True)
        d_prompt = pd.get_dummies(tmp["prompt_id"], prefix="pid", drop_first=True)
        x_fe = np.column_stack([x, d_source.to_numpy(dtype=float), d_prompt.to_numpy(dtype=float)])
        _, r2_fe = fit_ols(y, x_fe)

        rows.append(
            {
                "model": "+".join(feats),
                "n": len(tmp),
                "num_features": len(feats),
                "r2_plain": r2_plain,
                "r2_with_source_prompt_FE": r2_fe,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("r2_with_source_prompt_FE", ascending=False)
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df_wm_attn = pd.read_csv(args.differential_summary_wm_csv)
    wm_feat = build_video_features(df_wm_attn, prefix="wm_")

    merged_feat = wm_feat
    has_delta = False
    if args.differential_summary_org_csv and os.path.isfile(args.differential_summary_org_csv):
        df_org_attn = pd.read_csv(args.differential_summary_org_csv)
        org_feat = build_video_features(df_org_attn, prefix="org_")
        merged_feat = wm_feat.merge(org_feat, on="video_filename", how="inner")
        for c in [
            "ahi_mean",
            "ahi_std",
            "ahi_stable",
            "cahi_mean",
            "ahi_margin",
            "subject_mean",
            "comp_mean",
        ]:
            merged_feat[f"delta_{c}"] = merged_feat[f"wm_{c}"] - merged_feat[f"org_{c}"]
        has_delta = True

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

    data = merged_feat.merge(df_org, on="video_filename", how="inner", suffixes=("", "_org"))
    data = data.merge(df_wm, on="video_filename", how="inner", suffixes=("_org", "_wm"))

    for c in keep_cols[1:]:
        data[f"delta_{c}"] = data[f"{c}_wm"] - data[f"{c}_org"]

    y_col = f"delta_{args.score_col}"
    if y_col not in data.columns:
        raise ValueError(f"Target not found: {y_col}")

    data["source"] = data["video_filename"].map(_extract_source)
    data["prompt_id"] = data["video_filename"].map(_extract_prompt_id)

    candidate_feats = [
        "wm_ahi_mean",
        "wm_ahi_std",
        "wm_ahi_stable",
        "wm_cahi_mean",
        "wm_ahi_margin",
        "wm_subject_mean",
        "wm_comp_mean",
        "wm_ahi_max",
        "wm_ahi_p90",
    ]
    if has_delta:
        candidate_feats += [
            "delta_ahi_mean",
            "delta_ahi_std",
            "delta_ahi_stable",
            "delta_cahi_mean",
            "delta_ahi_margin",
            "delta_subject_mean",
            "delta_comp_mean",
        ]

    uni_rows = [evaluate_feature(data, f, y_col) for f in candidate_feats if f in data.columns]
    uni_df = pd.DataFrame(uni_rows).sort_values("corr_within_prompt", ascending=False)

    mv_sets: List[List[str]] = [
        ["wm_ahi_mean"],
        ["wm_ahi_stable"],
        ["wm_cahi_mean"],
        ["wm_ahi_margin"],
        ["wm_ahi_mean", "wm_subject_mean"],
        ["wm_ahi_stable", "wm_subject_mean"],
        ["wm_ahi_mean", "wm_ahi_std", "wm_subject_mean"],
    ]
    if has_delta:
        mv_sets += [
            ["delta_ahi_mean"],
            ["delta_ahi_stable"],
            ["delta_cahi_mean"],
            ["delta_ahi_mean", "delta_subject_mean"],
            ["delta_ahi_stable", "delta_subject_mean"],
        ]

    mv_sets = [s for s in mv_sets if all(c in data.columns for c in s)]
    mv_df = evaluate_multivariate(data, y_col, mv_sets)

    out_uni = os.path.join(args.output_dir, "ahi_variants_univariate.csv")
    out_mv = os.path.join(args.output_dir, "ahi_variants_multivariate.csv")
    out_data = os.path.join(args.output_dir, "ahi_variants_dataset.csv")
    out_report = os.path.join(args.output_dir, "ahi_variants_report.txt")

    uni_df.to_csv(out_uni, index=False)
    mv_df.to_csv(out_mv, index=False)
    data.to_csv(out_data, index=False)

    lines = []
    lines.append("AHI Variants Exploration Report")
    lines.append(f"N={len(data)}")
    lines.append(f"Target={y_col}")
    lines.append(f"HasDeltaFeatures={has_delta}")
    lines.append("")
    lines.append("Top univariate features by within-prompt correlation:")
    for _, r in uni_df.head(8).iterrows():
        lines.append(
            f"- {r['feature']}: corr_within_prompt={r['corr_within_prompt']:.4f}, "
            f"pearson={r['pearson']:.4f}, spearman={r['spearman']:.4f}, r2_simple={r['r2_simple']:.4f}"
        )

    lines.append("")
    lines.append("Top multivariate models by R2 with source+prompt fixed effects:")
    if mv_df.empty:
        lines.append("- None")
    else:
        for _, r in mv_df.head(8).iterrows():
            lines.append(
                f"- {r['model']}: r2_plain={r['r2_plain']:.4f}, "
                f"r2_with_FE={r['r2_with_source_prompt_FE']:.4f}, n={int(r['n'])}"
            )

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved:")
    print(out_uni)
    print(out_mv)
    print(out_data)
    print(out_report)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
