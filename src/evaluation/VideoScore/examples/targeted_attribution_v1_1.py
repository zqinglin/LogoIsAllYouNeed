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
METRIC_INDEX = {
    "visual_quality": 0,
    "temporal_consistency": 1,
    "dynamic_degree": 2,
    "text_to_video_alignment": 3,
    "factual_consistency": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted attribution by gradients wrt pixels for VideoScore v1.1"
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument(
        "--origin_video_path",
        type=str,
        default="",
        help="Optional original (non-watermarked) video. If provided, per-frame bbox is extracted from frame differences.",
    )
    parser.add_argument("--text_prompt", type=str, default="A high-quality video.")
    parser.add_argument(
        "--target_metric",
        type=str,
        default="visual_quality",
        choices=list(METRIC_INDEX.keys()),
    )
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument(
        "--max_image_edge",
        type=int,
        default=448,
        help="Resize each frame so its longer edge is at most this value.",
    )
    parser.add_argument(
        "--watermark_bbox",
        type=str,
        default="",
        help="Normalized bbox x1,y1,x2,y2 (e.g. 0.72,0.82,0.98,0.98).",
    )
    parser.add_argument(
        "--frame_bbox_csv",
        type=str,
        default="",
        help="Optional per-frame bbox CSV with columns frame_idx,x1,y1,x2,y2 (normalized).",
    )
    parser.add_argument(
        "--watermark_motion_video",
        type=str,
        default="",
        help="Optional watermark motion clip (e.g. sora watermark video) used to auto-extract per-frame dynamic bbox.",
    )
    parser.add_argument(
        "--motion_threshold",
        type=int,
        default=15,
        help="Grayscale threshold for non-black watermark region extraction.",
    )
    parser.add_argument(
        "--diff_threshold",
        type=int,
        default=18,
        help="Threshold (0-255) for |watermarked-origin| difference mask when origin_video_path is provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/targeted_attribution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def parse_bbox(bbox_str: str) -> Optional[Tuple[float, float, float, float]]:
    if not bbox_str:
        return None
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("watermark_bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("watermark_bbox must be normalized and satisfy 0<=x1<x2<=1, 0<=y1<y2<=1")
    return x1, y1, x2, y2


def load_frame_bbox_csv(csv_path: str) -> Dict[int, Tuple[float, float, float, float]]:
    if not csv_path:
        return {}
    df = pd.read_csv(csv_path)
    required = {"frame_idx", "x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        raise ValueError("frame_bbox_csv must contain frame_idx,x1,y1,x2,y2")

    out: Dict[int, Tuple[float, float, float, float]] = {}
    for row in df.itertuples(index=False):
        out[int(row.frame_idx)] = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
    return out


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


def sample_video_frames(video_path: str, max_frames: int) -> Tuple[List[Image.Image], np.ndarray, int]:
    with av.open(video_path) as container:
        total_frames = container.streams.video[0].frames
        if total_frames is None or total_frames <= 0:
            decoded = [f.to_image() for f in container.decode(video=0)]
            if not decoded:
                return [], np.array([], dtype=int), 0
            if len(decoded) <= max_frames:
                idx = np.arange(len(decoded), dtype=int)
                return decoded, idx, len(decoded)
            idx = np.linspace(0, len(decoded) - 1, max_frames, dtype=int)
            return [decoded[int(i)] for i in idx], idx, len(decoded)

        if total_frames <= max_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        return _read_video_pyav(container, indices), indices, int(total_frames)


def read_video_frames_by_indices(video_path: str, indices: np.ndarray) -> List[Image.Image]:
    if indices.size == 0:
        return []
    with av.open(video_path) as container:
        return _read_video_pyav(container, indices)


def resize_frame_keep_aspect(frame: Image.Image, max_edge: int) -> Image.Image:
    w, h = frame.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return frame

    scale = float(max_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return frame.resize((new_w, new_h), Image.BICUBIC)


def normalized_bbox_to_mask(h: int, w: int, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    xx1 = int(round(x1 * w))
    yy1 = int(round(y1 * h))
    xx2 = int(round(x2 * w))
    yy2 = int(round(y2 * h))
    xx1 = max(0, min(w - 1, xx1))
    yy1 = max(0, min(h - 1, yy1))
    xx2 = max(xx1 + 1, min(w, xx2))
    yy2 = max(yy1 + 1, min(h, yy2))
    mask = np.zeros((h, w), dtype=bool)
    mask[yy1:yy2, xx1:xx2] = True
    return mask


def overlay_saliency(frame: Image.Image, saliency: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    s = saliency.copy()
    if np.max(s) > np.min(s):
        s = (s - np.min(s)) / (np.max(s) - np.min(s))

    cmap = plt.get_cmap("magma")
    heatmap = (cmap(s)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize(frame.size, Image.BILINEAR)
    overlay = Image.blend(frame.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=0.55)
    overlay.convert("RGB").save(out_path)


def extract_dynamic_bboxes_from_motion_video(
    motion_video_path: str,
    num_frames: int,
    threshold: int,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Extract a dynamic normalized bbox for each sampled frame from a watermark-only motion video.
    This mirrors the contour-based logic in make_dynamic_mask.py.
    """
    import cv2

    frame_bbox_map: Dict[int, Tuple[float, float, float, float]] = {}
    if not motion_video_path or num_frames <= 0:
        return frame_bbox_map

    cap = cv2.VideoCapture(motion_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open watermark motion video: {motion_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError("Invalid motion video metadata (frame count/size).")

    if total_frames <= num_frames:
        sample_indices = np.arange(total_frames, dtype=int)
    else:
        sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for out_i, src_i in enumerate(sample_indices.tolist()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_i))
        ret, frame = cap.read()
        if not ret:
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
            frame_bbox_map[out_i] = (x1, y1, x2, y2)

    cap.release()
    return frame_bbox_map


def extract_bboxes_from_video_difference(
    watermarked_frames: List[Image.Image],
    origin_frames: List[Image.Image],
    threshold: int,
) -> Dict[int, Tuple[float, float, float, float]]:
    import cv2

    out: Dict[int, Tuple[float, float, float, float]] = {}
    n = min(len(watermarked_frames), len(origin_frames))
    for i in range(n):
        wm = np.asarray(watermarked_frames[i].convert("RGB"), dtype=np.int16)
        og = np.asarray(origin_frames[i].convert("RGB"), dtype=np.int16)
        if wm.shape != og.shape:
            continue

        diff = np.abs(wm - og).mean(axis=2).astype(np.uint8)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        h, w = diff.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + bw)
            y_max = max(y_max, y + bh)

        x1 = max(0.0, min(1.0, float(x_min) / float(w)))
        y1 = max(0.0, min(1.0, float(y_min) / float(h)))
        x2 = max(0.0, min(1.0, float(x_max) / float(w)))
        y2 = max(0.0, min(1.0, float(y_max) / float(h)))
        if x2 > x1 and y2 > y1:
            out[i] = (x1, y1, x2, y2)

    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    static_bbox = parse_bbox(args.watermark_bbox)
    frame_bbox_map = load_frame_bbox_csv(args.frame_bbox_csv)

    frames, sampled_indices, _ = sample_video_frames(args.video_path, args.max_frames)
    if not frames:
        print("No frames sampled from video.")
        return
    frames = [resize_frame_keep_aspect(f, args.max_image_edge) for f in frames]

    print(f"Using device: {args.device}")
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    # We only need gradients wrt pixel inputs, not model parameters.
    for p in model.parameters():
        p.requires_grad_(False)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    metric_idx = METRIC_INDEX[args.target_metric]

    # OOM-safe retry by progressively reducing frame count.
    trial_counts: List[int] = []
    fcnt = len(frames)
    while fcnt >= 1:
        trial_counts.append(fcnt)
        fcnt = fcnt // 2
        if fcnt == 0:
            break

    target_logit = None
    grads = None
    used_frames: List[Image.Image] = []
    last_err = None

    for try_n, frame_count in enumerate(trial_counts, start=1):
        try:
            used_frames = frames[:frame_count]
            prompt = f"Rate the {args.target_metric} for this video. Prompt: {args.text_prompt}. " + ("<image> " * len(used_frames))

            inputs = processor(text=[prompt], images=[used_frames], return_tensors="pt", padding=True)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            pixel_values = inputs["pixel_values"].detach().clone().to(dtype=torch.bfloat16).requires_grad_(True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    pixel_values=pixel_values,
                    pixel_attention_mask=inputs.get("pixel_attention_mask", None),
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                target_logit = outputs.logits[0, metric_idx]

            grad_tensor = torch.autograd.grad(target_logit, pixel_values, retain_graph=False, create_graph=False)[0]
            grads = np.abs(grad_tensor.detach().float().cpu().numpy()[0]).mean(axis=1)
            print(f"Attribution succeeded with {frame_count} frame(s) on attempt {try_n}.")
            break

        except RuntimeError as exc:
            last_err = exc
            oom_msg = str(exc).lower()
            if "out of memory" in oom_msg and args.device.startswith("cuda"):
                print(f"OOM at {frame_count} frame(s), retrying with fewer frames...")
                torch.cuda.empty_cache()
                continue
            raise

    if grads is None or target_logit is None:
        raise RuntimeError(f"Targeted attribution failed after retries. Last error: {last_err}")

    auto_dynamic_bbox_map: Dict[int, Tuple[float, float, float, float]] = {}
    if args.watermark_motion_video:
        auto_dynamic_bbox_map = extract_dynamic_bboxes_from_motion_video(
            motion_video_path=args.watermark_motion_video,
            num_frames=len(used_frames),
            threshold=args.motion_threshold,
        )
        print(
            f"Dynamic bbox extracted for {len(auto_dynamic_bbox_map)}/{len(used_frames)} frames "
            f"from: {args.watermark_motion_video}"
        )

    diff_bbox_map: Dict[int, Tuple[float, float, float, float]] = {}
    if args.origin_video_path:
        origin_frames = read_video_frames_by_indices(args.origin_video_path, sampled_indices[: len(used_frames)])
        origin_frames = [resize_frame_keep_aspect(f, args.max_image_edge) for f in origin_frames]
        diff_bbox_map = extract_bboxes_from_video_difference(
            watermarked_frames=used_frames,
            origin_frames=origin_frames,
            threshold=args.diff_threshold,
        )
        print(
            f"Difference-based bbox extracted for {len(diff_bbox_map)}/{len(used_frames)} frames "
            f"from origin-video alignment."
        )

    rows = []
    wm_means = []
    bg_means = []

    for i, frame in enumerate(used_frames):
        sal = grads[i]
        h, w = sal.shape

        if i in frame_bbox_map:
            bbox = frame_bbox_map[i]
        elif i in diff_bbox_map:
            bbox = diff_bbox_map[i]
        elif i in auto_dynamic_bbox_map:
            bbox = auto_dynamic_bbox_map[i]
        else:
            bbox = static_bbox

        if bbox is None:
            wm_ratio = np.nan
            wm_mean = np.nan
            bg_mean = np.nan
        else:
            wm_mask = normalized_bbox_to_mask(h, w, bbox)
            bg_mask = ~wm_mask
            wm_mean = float(sal[wm_mask].mean()) if wm_mask.any() else np.nan
            bg_mean = float(sal[bg_mask].mean()) if bg_mask.any() else np.nan
            wm_ratio = float(wm_mean / (bg_mean + 1e-12)) if not np.isnan(bg_mean) else np.nan
            wm_means.append(wm_mean)
            bg_means.append(bg_mean)

        out_path = os.path.join(args.output_dir, f"frame_{i:04d}_{args.target_metric}_attribution.png")
        overlay_saliency(frame, sal, out_path)

        rows.append(
            {
                "frame_idx": i,
                "target_metric": args.target_metric,
                "target_logit": float(target_logit.detach().cpu().item()),
                "wm_grad_mean": wm_mean,
                "bg_grad_mean": bg_mean,
                "wm_to_bg_ratio": wm_ratio,
                "overlay_path": out_path,
            }
        )

    frame_csv = os.path.join(args.output_dir, f"targeted_attribution_{args.target_metric}_frames.csv")
    pd.DataFrame(rows).to_csv(frame_csv, index=False)

    summary = {
        "video_path": args.video_path,
        "target_metric": args.target_metric,
        "target_logit": float(target_logit.detach().cpu().item()),
        "n_frames": len(used_frames),
        "mean_wm_grad": float(np.mean(wm_means)) if wm_means else np.nan,
        "mean_bg_grad": float(np.mean(bg_means)) if bg_means else np.nan,
        "mean_wm_to_bg_ratio": float(np.mean(np.array(wm_means) / (np.array(bg_means) + 1e-12))) if wm_means else np.nan,
        "n_frames_with_bbox": len(wm_means),
        "used_diff_bbox": bool(args.origin_video_path),
        "used_dynamic_bbox": bool(args.watermark_motion_video),
    }
    summary_csv = os.path.join(args.output_dir, f"targeted_attribution_{args.target_metric}_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\nTargeted attribution completed.")
    print(f"Frame-level CSV: {frame_csv}")
    print(f"Summary CSV: {summary_csv}")
    if not np.isnan(summary["mean_wm_to_bg_ratio"]):
        print(f"Mean watermark/non-watermark attribution ratio: {summary['mean_wm_to_bg_ratio']:.4f}")
    print(f"Overlay frames saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
