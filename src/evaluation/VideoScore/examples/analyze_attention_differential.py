
import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor

# --- Path Setup ---
VIDEOSCORE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

try:
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as e:
    print(f"FATAL: Could not import from mantis. Path: {VIDEOSCORE_PROJECT_PATH}")
    sys.exit(1)

MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"
DEFAULT_VIDEO_DIR = "data/videos/GenVideos/my_videos/watermarked_videos"
DEFAULT_OUTPUT_DIR = "outputs/differential_analysis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch differential attribution maps for VideoScore v1.1"
    )
    parser.add_argument("--video_path", type=str, default="", help="Single video path (optional).")
    parser.add_argument("--video_dir", type=str, default=DEFAULT_VIDEO_DIR, help="Directory of videos for batch mode.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--num_frames", type=int, default=2, help="Number of decoded frames per video.")
    parser.add_argument("--max_videos", type=int, default=0, help="Limit number of videos in batch mode. 0 means all.")
    parser.add_argument("--mask_size", type=int, default=16, help="Square mask size in pixels.")
    parser.add_argument("--stride", type=int, default=8, help="Mask stride in pixels.")
    parser.add_argument("--max_image_edge", type=int, default=384, help="Resize longer frame edge to this value.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing maps.")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of parallel workers.")
    parser.add_argument("--worker_rank", type=int, default=0, help="Current worker rank in [0, num_workers-1].")
    parser.add_argument(
        "--watermark_bbox",
        type=str,
        default="0.72,0.82,0.98,0.98",
        help="Normalized watermark bbox x1,y1,x2,y2 for AHI metrics.",
    )
    parser.add_argument(
        "--subject_topk_frac",
        type=float,
        default=0.01,
        help="Top-k fraction outside watermark used as a proxy for subject competition.",
    )
    parser.add_argument(
        "--save_raw_map",
        action="store_true",
        help="Save raw differential map as .npy for each frame.",
    )
    parser.add_argument(
        "--watermark_motion_video",
        type=str,
        default="",
        help="Optional watermark-only motion video for dynamic bbox tracking (OpenCV contour).",
    )
    parser.add_argument(
        "--motion_threshold",
        type=int,
        default=15,
        help="Grayscale threshold for dynamic watermark extraction from motion video.",
    )
    return parser.parse_args()


def parse_norm_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("watermark_bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("watermark_bbox must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return x1, y1, x2, y2


def bbox_mask(height: int, width: int, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    xx1 = int(round(x1 * width))
    yy1 = int(round(y1 * height))
    xx2 = int(round(x2 * width))
    yy2 = int(round(y2 * height))
    xx1 = max(0, min(width - 1, xx1))
    yy1 = max(0, min(height - 1, yy1))
    xx2 = max(xx1 + 1, min(width, xx2))
    yy2 = max(yy1 + 1, min(height, yy2))
    m = np.zeros((height, width), dtype=bool)
    m[yy1:yy2, xx1:xx2] = True
    return m


def extract_dynamic_bboxes_from_motion_video(
    motion_video_path: str,
    num_frames: int,
    threshold: int,
) -> Dict[int, Tuple[float, float, float, float]]:
    """Extract per-frame normalized bbox from watermark motion video via contour union."""
    import cv2

    out: Dict[int, Tuple[float, float, float, float]] = {}
    if not motion_video_path or num_frames <= 0:
        return out

    cap = cv2.VideoCapture(motion_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open watermark_motion_video: {motion_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError("Invalid watermark motion video metadata.")

    if total_frames <= num_frames:
        sample_indices = np.arange(total_frames, dtype=int)
    else:
        sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, src_idx in enumerate(sample_indices.tolist()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        x_min, y_min = frame_w, frame_h
        x_max, y_max = 0, 0
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + bw)
            y_max = max(y_max, y + bh)

        x1 = max(0.0, min(1.0, float(x_min) / float(frame_w)))
        y1 = max(0.0, min(1.0, float(y_min) / float(frame_h)))
        x2 = max(0.0, min(1.0, float(x_max) / float(frame_w)))
        y2 = max(0.0, min(1.0, float(y_max) / float(frame_h)))
        if x2 > x1 and y2 > y1:
            out[i] = (x1, y1, x2, y2)

    cap.release()
    return out


def compute_attention_metrics(raw_map: np.ndarray, wm_mask: np.ndarray, topk_frac: float) -> dict:
    eps = 1e-12
    global_sum = float(np.sum(raw_map))
    wm_sum = float(np.sum(raw_map[wm_mask])) if np.any(wm_mask) else 0.0
    ahi = wm_sum / (global_sum + eps)

    wm_area_ratio = float(np.mean(wm_mask))
    global_mean_density = global_sum / float(raw_map.size + eps)
    wm_mean_density = wm_sum / float(np.sum(wm_mask) + eps)
    wm_density_ratio = wm_mean_density / (global_mean_density + eps)

    non_wm = raw_map[~wm_mask]
    if non_wm.size == 0:
        subject_topk_share = 0.0
    else:
        k = max(1, int(round(non_wm.size * max(1e-6, min(topk_frac, 1.0)))))
        topk_vals = np.partition(non_wm, -k)[-k:]
        subject_topk_share = float(np.sum(topk_vals)) / (global_sum + eps)

    competition_ratio = subject_topk_share / (ahi + eps)
    return {
        "ahi": float(ahi),
        "wm_energy": float(wm_sum),
        "global_energy": float(global_sum),
        "wm_area_ratio": float(wm_area_ratio),
        "wm_density_ratio": float(wm_density_ratio),
        "subject_topk_share": float(subject_topk_share),
        "competition_ratio": float(competition_ratio),
    }

# --- Helper Functions ---
def _read_video_frames(video_path, max_frames):
    """Reads up to max_frames from a video and returns them as a list of PIL Images."""
    frames = []
    try:
        with av.open(video_path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_frames:
                    break
                frames.append(frame.to_image())
        return frames
    except Exception as e:
        print(f"FATAL: Could not read video file {video_path}. Error: {e}")
        return None

def resize_keep_aspect(image, max_edge):
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return image
    scale = float(max_edge) / float(long_edge)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return image.resize((nw, nh), Image.BICUBIC)


def get_score(model, processor, image, device):
    """Gets the visual quality score for a single image."""
    prompt = "User: Rate the visual quality of the image.<image> Assistant:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    return score

def process_video(video_path, model, processor, args):
    video_name = os.path.basename(video_path)
    video_stem, _ = os.path.splitext(video_name)
    video_out_dir = os.path.join(args.output_dir, video_stem)
    os.makedirs(video_out_dir, exist_ok=True)

    video_frames = _read_video_frames(video_path, args.num_frames)
    if not video_frames:
        return []

    video_frames = [resize_keep_aspect(f, args.max_image_edge) for f in video_frames]

    rows = []
    dynamic_bbox_map: Dict[int, Tuple[float, float, float, float]] = {}
    if args.watermark_motion_video:
        try:
            dynamic_bbox_map = extract_dynamic_bboxes_from_motion_video(
                args.watermark_motion_video,
                num_frames=len(video_frames),
                threshold=args.motion_threshold,
            )
        except Exception as exc:
            print(f"WARN: dynamic bbox extraction failed for {video_name}: {exc}")

    for i, original_image in enumerate(video_frames):
        output_path = os.path.join(video_out_dir, f"differential_map_frame_{i:04d}.png")
        raw_map_path = os.path.join(video_out_dir, f"differential_map_frame_{i:04d}.npy")
        if (not args.overwrite) and os.path.exists(output_path):
            rows.append(
                {
                    "video_filename": video_name,
                    "frame_idx": i,
                    "benchmark_score": np.nan,
                    "output_path": output_path,
                    "ahi": np.nan,
                    "wm_energy": np.nan,
                    "global_energy": np.nan,
                    "wm_area_ratio": np.nan,
                    "wm_density_ratio": np.nan,
                    "subject_topk_share": np.nan,
                    "competition_ratio": np.nan,
                    "bbox_source": "cached",
                    "bbox_x1": np.nan,
                    "bbox_y1": np.nan,
                    "bbox_x2": np.nan,
                    "bbox_y2": np.nan,
                }
            )
            continue

        width, height = original_image.size
        influence_heatmap = np.zeros((height, width))

        benchmark_score = get_score(model, processor, original_image, args.device)

        y_steps = range(0, height - args.mask_size + 1, args.stride)
        x_steps = range(0, width - args.mask_size + 1, args.stride)

        for y in y_steps:
            for x in x_steps:
                masked_image = original_image.copy()
                draw = ImageDraw.Draw(masked_image)
                draw.rectangle([x, y, x + args.mask_size, y + args.mask_size], fill="grey")
                perturbed_score = get_score(model, processor, masked_image, args.device)
                score_diff = abs(benchmark_score - perturbed_score)
                influence_heatmap[y : y + args.mask_size, x : x + args.mask_size] += score_diff

        raw_map = influence_heatmap.copy()
        frame_bbox: Optional[Tuple[float, float, float, float]] = dynamic_bbox_map.get(i)
        bbox_source = "dynamic" if frame_bbox is not None else "static"
        if frame_bbox is None:
            frame_bbox = args._watermark_bbox_tuple

        wm_mask = bbox_mask(height, width, frame_bbox)
        metrics = compute_attention_metrics(raw_map, wm_mask, args.subject_topk_frac)

        if np.max(influence_heatmap) > 0:
            influence_heatmap = (influence_heatmap - np.min(influence_heatmap)) / (
                np.max(influence_heatmap) - np.min(influence_heatmap)
            )

        cmap = plt.get_cmap("jet")
        heatmap_pil = Image.fromarray((cmap(influence_heatmap)[:, :, :3] * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize(original_image.size, Image.LANCZOS)

        overlay_frame = Image.blend(original_image.convert("RGBA"), heatmap_resized.convert("RGBA"), alpha=0.6)
        overlay_frame.convert("RGB").save(output_path)
        if args.save_raw_map:
            np.save(raw_map_path, raw_map.astype(np.float32))

        rows.append(
            {
                "video_filename": video_name,
                "frame_idx": i,
                "benchmark_score": float(benchmark_score),
                "output_path": output_path,
                "ahi": metrics["ahi"],
                "wm_energy": metrics["wm_energy"],
                "global_energy": metrics["global_energy"],
                "wm_area_ratio": metrics["wm_area_ratio"],
                "wm_density_ratio": metrics["wm_density_ratio"],
                "subject_topk_share": metrics["subject_topk_share"],
                "competition_ratio": metrics["competition_ratio"],
                "bbox_source": bbox_source,
                "bbox_x1": frame_bbox[0],
                "bbox_y1": frame_bbox[1],
                "bbox_x2": frame_bbox[2],
                "bbox_y2": frame_bbox[3],
            }
        )

    return rows


def collect_video_paths(args):
    if args.video_path:
        return [args.video_path]

    if not os.path.isdir(args.video_dir):
        raise ValueError(f"video_dir not found: {args.video_dir}")

    videos = [os.path.join(args.video_dir, f) for f in sorted(os.listdir(args.video_dir)) if f.endswith(".mp4")]
    if args.num_workers > 1:
        if not (0 <= args.worker_rank < args.num_workers):
            raise ValueError("worker_rank must satisfy 0 <= worker_rank < num_workers")
        videos = [v for i, v in enumerate(videos) if i % args.num_workers == args.worker_rank]
    if args.max_videos and args.max_videos > 0:
        videos = videos[: args.max_videos]
    return videos


def differential_analysis():
    args = parse_args()
    args._watermark_bbox_tuple = parse_norm_bbox(args.watermark_bbox)
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Starting Batch Differential Attention Analysis ---")
    print(f"Using device: {args.device}")
    print(f"Worker: rank {args.worker_rank}/{args.num_workers}")
    print(f"Watermark bbox: {args.watermark_bbox}")
    if args.watermark_motion_video:
        print(f"Dynamic watermark video: {args.watermark_motion_video}")
    print("Loading model and processor...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    video_paths = collect_video_paths(args)
    if not video_paths:
        print("No videos found to process.")
        return

    all_rows = []
    for vpath in tqdm(video_paths, desc="Videos"):
        try:
            rows = process_video(vpath, model, processor, args)
            all_rows.extend(rows)
        except Exception as exc:
            all_rows.append(
                {
                    "video_filename": os.path.basename(vpath),
                    "frame_idx": -1,
                    "benchmark_score": np.nan,
                    "output_path": f"ERROR: {exc}",
                }
            )

    if args.num_workers > 1:
        summary_csv = os.path.join(
            args.output_dir,
            f"differential_analysis_summary_worker_{args.worker_rank:02d}.csv",
        )
    else:
        summary_csv = os.path.join(args.output_dir, "differential_analysis_summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_csv, index=False)

    print("\n--- Analysis Complete! ---")
    print(f"Generated maps for {len(video_paths)} videos")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    differential_analysis()
