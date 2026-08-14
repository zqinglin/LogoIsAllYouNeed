import os
from typing import Dict

import numpy as np
import pandas as pd


EVAL_DIR = "outputs"
OUT_DIR = os.path.join(EVAL_DIR, "detailed_comparison")

REPORT_FILES = {
    "sora": "comparison_report.csv",
    "gemini": "comparison_report_gemini.csv",
    "kling": "comparison_report_kling.csv",
    "gray": "comparison_report_gray.csv",
    "sora_flipped": "comparison_report_sora_flipped.csv",
}

DIMENSIONS = [
    ("v", "visual_quality"),
    ("t", "temporal_consistency"),
    ("d", "dynamic_degree"),
    ("a", "text_to_video_alignment"),
    ("f", "factual_consistency"),
]


def summarize_series(s: pd.Series) -> Dict[str, float]:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
            "positive_rate": np.nan,
            "negative_rate": np.nan,
            "zero_rate": np.nan,
        }

    eps = 1e-9
    pos_rate = float((s > eps).mean())
    neg_rate = float((s < -eps).mean())
    zero_rate = float((s.abs() <= eps).mean())

    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "median": float(s.median()),
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
        "positive_rate": pos_rate,
        "negative_rate": neg_rate,
        "zero_rate": zero_rate,
    }


def build_dimension_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for short, name in DIMENSIONS:
        out[f"delta_{name}"] = out[f"wm_{short}"] - out[f"orig_{short}"]
    return out


def build_master_row(style: str, df: pd.DataFrame) -> Dict[str, float]:
    row: Dict[str, float] = {
        "style": style,
        "num_videos": int(df.shape[0]),
        "avg_original_score": float(df["orig_total"].mean()),
        "avg_watermarked_score": float(df["wm_total"].mean()),
    }

    overall = summarize_series(df["delta"])
    for k, v in overall.items():
        row[f"overall_{k}"] = v

    for _, dim_name in DIMENSIONS:
        d = summarize_series(df[f"delta_{dim_name}"])
        for k, v in d.items():
            row[f"{dim_name}_{k}"] = v

    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    master_rows = []

    for style, filename in REPORT_FILES.items():
        path = os.path.join(EVAL_DIR, filename)
        if not os.path.exists(path):
            print(f"Skip {style}: file not found -> {path}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"Skip {style}: empty report -> {path}")
            continue

        required = {"video_filename", "orig_total", "wm_total", "delta"}
        if not required.issubset(set(df.columns)):
            print(f"Skip {style}: missing required columns in {path}")
            continue

        df = build_dimension_deltas(df)

        master_rows.append(build_master_row(style, df))

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        master_df = master_df.sort_values("overall_mean", ascending=True).round(6)
        out_path = os.path.join(OUT_DIR, "master_detailed_comparison_table.csv")
        master_df.to_csv(out_path, index=False)
        print("Generated master detailed report:")
        print(f" - {out_path}")
    else:
        print("No detailed report generated. Check source CSV files.")


if __name__ == "__main__":
    main()
