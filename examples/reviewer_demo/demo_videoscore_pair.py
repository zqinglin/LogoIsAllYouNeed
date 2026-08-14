import argparse
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

ASPECT_NAMES = [
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_alignment",
    "factual_consistency",
]

PROMPT_TEMPLATE = """
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


def read_video_frames(video_path: str, max_frames: int) -> list[Image.Image]:
    with av.open(video_path) as container:
        total_frames = container.streams.video[0].frames
        if total_frames <= 0:
            raise RuntimeError(f"No frames found in {video_path}")
        if total_frames > max_frames:
            indices = np.arange(0, total_frames, total_frames / max_frames).astype(int)
        else:
            indices = np.arange(total_frames)

        frames = []
        start_index = int(indices[0])
        end_index = int(indices[-1])
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                arr = frame.to_ndarray(format="rgb24")
                frames.append(Image.fromarray(arr))
    if not frames:
        raise RuntimeError(f"Failed to decode sampled frames from {video_path}")
    return frames


def evaluate_video(model, processor, device: str, video_path: str, prompt: str, max_frames: int) -> dict:
    frames = read_video_frames(video_path, max_frames=max_frames)
    eval_prompt = PROMPT_TEMPLATE.format(text_prompt=prompt) + "<image> " * len(frames)
    inputs = processor(text=eval_prompt, images=[frames], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    scores = [round(float(x), 3) for x in logits]
    result = {name: score for name, score in zip(ASPECT_NAMES, scores)}
    result["total_score"] = round(sum(scores) / len(scores), 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal pair demo for watermark score inflation.")
    parser.add_argument("--orig-video", required=True, help="Path to original video.")
    parser.add_argument("--wm-video", required=True, help="Path to watermarked video.")
    parser.add_argument("--prompt-text", required=True, help="Generation prompt text for this pair.")
    parser.add_argument("--model-name", default="TIGER-Lab/VideoScore-v1.1")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    local_videoscore = code_root / "src/evaluation/VideoScore"
    if str(local_videoscore) not in sys.path:
        sys.path.insert(0, str(local_videoscore))

    from transformers import AutoProcessor
    from mantis.models.idefics2 import Idefics2ForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] Loading model: {args.model_name}")
    print(f"[INFO] Device: {device}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Idefics2ForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).eval().to(device)

    print("[INFO] Evaluating original video...")
    original = evaluate_video(
        model=model,
        processor=processor,
        device=device,
        video_path=args.orig_video,
        prompt=args.prompt_text,
        max_frames=args.max_frames,
    )
    print("[INFO] Evaluating watermarked video...")
    watermarked = evaluate_video(
        model=model,
        processor=processor,
        device=device,
        video_path=args.wm_video,
        prompt=args.prompt_text,
        max_frames=args.max_frames,
    )

    delta = {k: round(watermarked[k] - original[k], 3) for k in original.keys()}

    print("\n=== Demo Result (VideoScore v1.1) ===")
    for k in ASPECT_NAMES + ["total_score"]:
        print(f"{k:>22}: orig={original[k]:.3f} | wm={watermarked[k]:.3f} | delta={delta[k]:+.3f}")

    payload = {
        "model": args.model_name,
        "orig_video": os.path.relpath(args.orig_video),
        "wm_video": os.path.relpath(args.wm_video),
        "prompt_text": args.prompt_text,
        "max_frames": args.max_frames,
        "scores_original": original,
        "scores_watermarked": watermarked,
        "delta": delta,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
