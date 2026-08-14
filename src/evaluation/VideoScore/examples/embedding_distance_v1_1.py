import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

try:
    import av
except ImportError:
    print("FATAL: Missing dependency 'av'. Please run this script inside your videoscore environment.")
    sys.exit(1)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

HIGH_ANCHOR = "high quality, clear, stable, professional video"
LOW_ANCHOR = "low quality, noisy, unstable, artifact-heavy video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding distance experiment for VideoScore v1.1"
    )
    parser.add_argument(
        "--origin_dir",
        type=str,
        default="data/videos/GenVideos/my_videos/Videos",
    )
    parser.add_argument(
        "--watermark_dir",
        type=str,
        default="data/videos/GenVideos/my_videos/watermarked_videos",
    )
    parser.add_argument(
        "--pair_csv",
        type=str,
        default="",
        help="Optional CSV with columns origin_file, watermarked_file, prompt(optional), alpha(optional), dynamic_strength(optional).",
    )
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default="data_metadata/video_to_prompt_full.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/embedding_distance_v1.1_pairs.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="outputs/embedding_distance_v1.1_summary.csv",
    )
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
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


def pooled_text_embedding(
    text: str,
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    device: str,
) -> torch.Tensor:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            output_hidden_states=False,
            return_dict=True,
        )

    hidden = out.last_hidden_state[0]
    attn = inputs.get("attention_mask", torch.ones(hidden.shape[0], device=hidden.device))[0].bool()
    pooled = hidden[attn].mean(dim=0)
    return F.normalize(pooled.float(), dim=0)


def pooled_video_visual_embedding(
    video_path: str,
    prompt_text: str,
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    device: str,
    max_frames: int,
) -> Optional[torch.Tensor]:
    frames = sample_video_frames(video_path, max_frames=max_frames)
    if not frames:
        return None

    # Keep a neutral query to reduce language leakage into visual representation.
    text = f"Rate this video quality briefly. Prompt context: {prompt_text}." + (" <image>" * len(frames))
    inputs = processor(text=[text], images=[frames], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            pixel_values=inputs.get("pixel_values", None),
            pixel_attention_mask=inputs.get("pixel_attention_mask", None),
            output_hidden_states=False,
            return_dict=True,
        )

    image_hidden_states = out.image_hidden_states
    if image_hidden_states is None:
        return None

    pooled = image_hidden_states.mean(dim=(0, 1))
    return F.normalize(pooled.float(), dim=0)


def load_prompt_map(prompt_csv: str) -> Dict[str, str]:
    df = pd.read_csv(prompt_csv)
    return dict(zip(df["filename"], df["prompt"]))


def collect_pairs(
    origin_dir: str,
    watermark_dir: str,
    pair_csv: str,
    prompt_map: Dict[str, str],
    max_pairs: int,
) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []

    if pair_csv:
        pdf = pd.read_csv(pair_csv)
        required = {"origin_file", "watermarked_file"}
        if not required.issubset(set(pdf.columns)):
            raise ValueError("pair_csv must include columns: origin_file, watermarked_file")

        for row in pdf.itertuples(index=False):
            origin_file = str(getattr(row, "origin_file"))
            wm_file = str(getattr(row, "watermarked_file"))
            origin_path = os.path.join(origin_dir, origin_file)
            wm_path = os.path.join(watermark_dir, wm_file)
            if not (os.path.exists(origin_path) and os.path.exists(wm_path)):
                continue

            prompt = getattr(row, "prompt", None)
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = prompt_map.get(origin_file) or prompt_map.get(wm_file) or "A high-quality video."

            item: Dict[str, object] = {
                "origin_file": origin_file,
                "watermarked_file": wm_file,
                "origin_path": origin_path,
                "watermarked_path": wm_path,
                "prompt": prompt,
            }
            if hasattr(row, "alpha"):
                item["alpha"] = getattr(row, "alpha")
            if hasattr(row, "dynamic_strength"):
                item["dynamic_strength"] = getattr(row, "dynamic_strength")
            pairs.append(item)
    else:
        origin_files = {f for f in os.listdir(origin_dir) if f.endswith(".mp4")}
        wm_files = {f for f in os.listdir(watermark_dir) if f.endswith(".mp4")}
        common = sorted(origin_files & wm_files)

        for fname in common:
            pairs.append(
                {
                    "origin_file": fname,
                    "watermarked_file": fname,
                    "origin_path": os.path.join(origin_dir, fname),
                    "watermarked_path": os.path.join(watermark_dir, fname),
                    "prompt": prompt_map.get(fname, "A high-quality video."),
                }
            )

    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]

    return pairs


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    dm = df["delta_margin"].dropna().to_numpy(dtype=float)
    dch = df["delta_cos_high"].dropna().to_numpy(dtype=float)
    dcl = df["delta_cos_low"].dropna().to_numpy(dtype=float)

    row = {
        "n_pairs": int(df.shape[0]),
        "n_valid_margin": int(np.sum(~np.isnan(df["delta_margin"].to_numpy(dtype=float)))),
        "mean_delta_cos_high": float(np.nanmean(dch)) if dch.size else np.nan,
        "mean_delta_cos_low": float(np.nanmean(dcl)) if dcl.size else np.nan,
        "mean_delta_margin": float(np.nanmean(dm)) if dm.size else np.nan,
        "std_delta_margin": float(np.nanstd(dm)) if dm.size else np.nan,
        "ratio_delta_margin_gt_0": float(np.mean(dm > 0)) if dm.size else np.nan,
    }

    # A simple exact sign-test (no scipy dependency).
    if dm.size:
        pos = int(np.sum(dm > 0))
        neg = int(np.sum(dm < 0))
        n = pos + neg
        if n > 0:
            import math

            k = min(pos, neg)
            cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
            row["p_value_sign_test_delta_margin"] = float(min(1.0, 2.0 * cdf))
        else:
            row["p_value_sign_test_delta_margin"] = np.nan
    else:
        row["p_value_sign_test_delta_margin"] = np.nan

    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    prompt_map = load_prompt_map(args.prompt_csv)
    pairs = collect_pairs(
        origin_dir=args.origin_dir,
        watermark_dir=args.watermark_dir,
        pair_csv=args.pair_csv,
        prompt_map=prompt_map,
        max_pairs=args.max_pairs,
    )

    if not pairs:
        print("No valid pairs found.")
        return

    print(f"Using device: {args.device}")
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    high_anchor = pooled_text_embedding(HIGH_ANCHOR, model, processor, args.device)
    low_anchor = pooled_text_embedding(LOW_ANCHOR, model, processor, args.device)

    rows: List[Dict[str, object]] = []

    for item in tqdm(pairs, desc="Embedding distance"):
        origin_vec = pooled_video_visual_embedding(
            video_path=item["origin_path"],
            prompt_text=item["prompt"],
            model=model,
            processor=processor,
            device=args.device,
            max_frames=args.max_frames,
        )
        wm_vec = pooled_video_visual_embedding(
            video_path=item["watermarked_path"],
            prompt_text=item["prompt"],
            model=model,
            processor=processor,
            device=args.device,
            max_frames=args.max_frames,
        )

        row = {
            "origin_file": item["origin_file"],
            "watermarked_file": item["watermarked_file"],
            "prompt": item["prompt"],
            "alpha": item.get("alpha", np.nan),
            "dynamic_strength": item.get("dynamic_strength", np.nan),
        }

        if origin_vec is None or wm_vec is None:
            row.update(
                {
                    "cos_orig_high": np.nan,
                    "cos_wm_high": np.nan,
                    "delta_cos_high": np.nan,
                    "cos_orig_low": np.nan,
                    "cos_wm_low": np.nan,
                    "delta_cos_low": np.nan,
                    "delta_margin": np.nan,
                }
            )
            rows.append(row)
            continue

        cos_orig_high = float(F.cosine_similarity(origin_vec, high_anchor, dim=0).item())
        cos_wm_high = float(F.cosine_similarity(wm_vec, high_anchor, dim=0).item())
        cos_orig_low = float(F.cosine_similarity(origin_vec, low_anchor, dim=0).item())
        cos_wm_low = float(F.cosine_similarity(wm_vec, low_anchor, dim=0).item())

        delta_cos_high = cos_wm_high - cos_orig_high
        delta_cos_low = cos_wm_low - cos_orig_low
        delta_margin = delta_cos_high - delta_cos_low

        row.update(
            {
                "cos_orig_high": cos_orig_high,
                "cos_wm_high": cos_wm_high,
                "delta_cos_high": delta_cos_high,
                "cos_orig_low": cos_orig_low,
                "cos_wm_low": cos_wm_low,
                "delta_cos_low": delta_cos_low,
                "delta_margin": delta_margin,
            }
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    summary = compute_summary(df)

    # Optional monotonic trend with alpha if available.
    if df["alpha"].notna().any():
        alpha_df = df[["alpha", "delta_margin"]].dropna()
        if len(alpha_df) >= 2:
            corr = np.corrcoef(alpha_df["alpha"].to_numpy(float), alpha_df["delta_margin"].to_numpy(float))[0, 1]
            summary["pearson_alpha_vs_delta_margin"] = float(corr)

    if df["dynamic_strength"].notna().any():
        dyn_df = df[["dynamic_strength", "delta_margin"]].dropna()
        if len(dyn_df) >= 2:
            corr = np.corrcoef(
                dyn_df["dynamic_strength"].to_numpy(float),
                dyn_df["delta_margin"].to_numpy(float),
            )[0, 1]
            summary["pearson_dynamic_strength_vs_delta_margin"] = float(corr)

    df.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print("\nEmbedding distance experiment completed.")
    print(f"Detailed CSV: {args.output_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    print("\nSummary preview:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
