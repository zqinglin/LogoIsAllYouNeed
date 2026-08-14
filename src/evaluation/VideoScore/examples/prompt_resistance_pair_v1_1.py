import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


VIDEOSCORE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

try:
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as exc:
    print(f"FATAL: Failed to import VideoScore model class: {exc}")
    sys.exit(1)


MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"
METRIC_NAMES = [
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_to_video_alignment",
    "factual_consistency",
]

VS1_1_REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

IMMUNITY_SENTENCE = (
    "Please strictly ignore any watermarks, logos, or text overlays "
    "when evaluating the quality."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt resistance paired experiment for VideoScore v1.1"
    )
    parser.add_argument(
        "--origin_dir",
        type=str,
        default="data/videos/GenVideos/my_videos/Videos",
        help="Directory containing origin videos.",
    )
    parser.add_argument(
        "--watermark_dir",
        type=str,
        default="data/videos/GenVideos/my_videos/watermarked_videos",
        help="Directory containing watermarked videos.",
    )
    parser.add_argument(
        "--pair_csv",
        type=str,
        default="",
        help="Optional CSV with columns: origin_file, watermarked_file, prompt(optional).",
    )
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default="data_metadata/video_to_prompt_full.csv",
        help="CSV containing filename->prompt mapping.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/prompt_resistance_v1.1_pairs.csv",
        help="Detailed output CSV path.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="outputs/prompt_resistance_v1.1_summary.csv",
        help="Summary statistics CSV path.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="Number of sampled frames per video.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Limit number of pairs for a quick run. 0 means all pairs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def _read_video_pyav(container: av.container.input.InputContainer, indices: np.ndarray) -> List[Image.Image]:
    frames: List[Image.Image] = []
    container.seek(0)
    start_index, end_index = int(indices[0]), int(indices[-1])
    needed = set(int(i) for i in indices)

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in needed:
            frames.append(frame.to_image())

    return frames


def sample_video_frames(video_path: str, max_frames: int) -> List[Image.Image]:
    with av.open(video_path) as container:
        total_frames = container.streams.video[0].frames
        if total_frames is None or total_frames <= 0:
            decoded = [f.to_image() for f in container.decode(video=0)]
            if not decoded:
                return []
            if len(decoded) <= max_frames:
                return decoded
            idx = np.linspace(0, len(decoded) - 1, max_frames, dtype=int)
            return [decoded[int(i)] for i in idx]

        if total_frames <= max_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        return _read_video_pyav(container, indices)


def build_eval_prompt(text_prompt: str, with_immunity: bool) -> str:
    prompt = VS1_1_REGRESSION_QUERY_PROMPT.format(text_prompt=text_prompt)
    if with_immunity:
        prompt += "\n" + IMMUNITY_SENTENCE + "\n"
    return prompt


def score_video(
    video_path: str,
    text_prompt: str,
    with_immunity: bool,
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    device: str,
    max_frames: int,
) -> Dict[str, float]:
    try:
        frames = sample_video_frames(video_path, max_frames=max_frames)
        if not frames:
            return {k: np.nan for k in METRIC_NAMES}

        eval_prompt = build_eval_prompt(text_prompt, with_immunity) + "<image> " * len(frames)
        inputs = processor(text=eval_prompt, images=[frames], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        values = logits.detach().float().cpu().tolist()
        values = values[: len(METRIC_NAMES)]
        if len(values) < len(METRIC_NAMES):
            values += [np.nan] * (len(METRIC_NAMES) - len(values))

        return {k: float(v) for k, v in zip(METRIC_NAMES, values)}
    except Exception:
        return {k: np.nan for k in METRIC_NAMES}


def load_prompt_map(prompt_csv: str) -> Dict[str, str]:
    df = pd.read_csv(prompt_csv)
    return dict(zip(df["filename"], df["prompt"]))


def collect_pairs(args: argparse.Namespace, prompt_map: Dict[str, str]) -> List[Tuple[str, str, str, str, str]]:
    pairs: List[Tuple[str, str, str, str, str]] = []

    if args.pair_csv:
        pair_df = pd.read_csv(args.pair_csv)
        required = {"origin_file", "watermarked_file"}
        if not required.issubset(set(pair_df.columns)):
            raise ValueError("pair_csv must include columns: origin_file, watermarked_file")

        for row in pair_df.itertuples(index=False):
            origin_file = str(getattr(row, "origin_file"))
            wm_file = str(getattr(row, "watermarked_file"))
            prompt = getattr(row, "prompt", None)
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = prompt_map.get(origin_file) or prompt_map.get(wm_file) or "A high-quality video."

            origin_path = os.path.join(args.origin_dir, origin_file)
            wm_path = os.path.join(args.watermark_dir, wm_file)
            if os.path.exists(origin_path) and os.path.exists(wm_path):
                pairs.append((origin_file, wm_file, origin_path, wm_path, prompt))
    else:
        origin_files = {f for f in os.listdir(args.origin_dir) if f.endswith(".mp4")}
        wm_files = {f for f in os.listdir(args.watermark_dir) if f.endswith(".mp4")}
        common = sorted(origin_files & wm_files)

        for fname in common:
            prompt = prompt_map.get(fname, "A high-quality video.")
            pairs.append(
                (
                    fname,
                    fname,
                    os.path.join(args.origin_dir, fname),
                    os.path.join(args.watermark_dir, fname),
                    prompt,
                )
            )

    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    return pairs


def paired_t_test(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        return np.nan

    try:
        from scipy.stats import ttest_rel

        return float(ttest_rel(x[mask], y[mask], nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def paired_sign_test(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        return np.nan

    d = y[mask] - x[mask]
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return np.nan

    # Two-sided exact sign test under H0(p=0.5).
    k = min(pos, neg)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * cdf))


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in METRIC_NAMES:
        delta_base = df[f"delta_base_{metric}"].to_numpy(dtype=float)
        delta_immune = df[f"delta_immune_{metric}"].to_numpy(dtype=float)
        immunity_effect = df[f"delta_immunity_effect_{metric}"].to_numpy(dtype=float)

        rows.append(
            {
                "metric": metric,
                "mean_delta_base": float(np.nanmean(delta_base)),
                "std_delta_base": float(np.nanstd(delta_base)),
                "mean_delta_immune": float(np.nanmean(delta_immune)),
                "std_delta_immune": float(np.nanstd(delta_immune)),
                "mean_immunity_effect": float(np.nanmean(immunity_effect)),
                "std_immunity_effect": float(np.nanstd(immunity_effect)),
                "p_value_paired_t_delta_base_vs_delta_immune": paired_t_test(delta_base, delta_immune),
                "p_value_sign_test_delta_base_vs_delta_immune": paired_sign_test(delta_base, delta_immune),
                "n_valid_pairs": int(np.sum((~np.isnan(delta_base)) & (~np.isnan(delta_immune)))),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    prompt_map = load_prompt_map(args.prompt_csv)
    pairs = collect_pairs(args, prompt_map)

    if not pairs:
        print("No valid origin/watermarked pairs found.")
        return

    print(f"Using device: {args.device}")
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    rows = []

    for origin_file, wm_file, origin_path, wm_path, prompt_text in tqdm(
        pairs, desc="Evaluating pairs"
    ):
        origin_base = score_video(
            origin_path, prompt_text, False, model, processor, args.device, args.max_frames
        )
        origin_immune = score_video(
            origin_path, prompt_text, True, model, processor, args.device, args.max_frames
        )
        wm_base = score_video(
            wm_path, prompt_text, False, model, processor, args.device, args.max_frames
        )
        wm_immune = score_video(
            wm_path, prompt_text, True, model, processor, args.device, args.max_frames
        )

        row: Dict[str, object] = {
            "origin_file": origin_file,
            "watermarked_file": wm_file,
            "prompt": prompt_text,
        }

        for metric in METRIC_NAMES:
            row[f"origin_base_{metric}"] = origin_base[metric]
            row[f"origin_immune_{metric}"] = origin_immune[metric]
            row[f"watermarked_base_{metric}"] = wm_base[metric]
            row[f"watermarked_immune_{metric}"] = wm_immune[metric]

            delta_base = wm_base[metric] - origin_base[metric]
            delta_immune = wm_immune[metric] - origin_immune[metric]
            delta_immunity_effect = delta_immune - delta_base

            row[f"delta_base_{metric}"] = delta_base
            row[f"delta_immune_{metric}"] = delta_immune
            row[f"delta_immunity_effect_{metric}"] = delta_immunity_effect

        rows.append(row)

    detailed_df = pd.DataFrame(rows)
    summary_df = summarize_results(detailed_df)

    detailed_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print("\nPrompt resistance experiment completed.")
    print(f"Detailed pairs CSV: {args.output_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    print("\nSummary preview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
