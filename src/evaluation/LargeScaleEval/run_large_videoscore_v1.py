import os
import glob
import pandas as pd
import numpy as np
import re
import torch
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Manager
from pathlib import Path

from mantis.models.qwen2_vl import Qwen2VLForSequenceClassification
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


# ================================
GPU_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
MODEL_NAME = "TIGER-Lab/VideoScore-Qwen2-VL"
NUM_PASSES = 3
MIN_FREE_MEMORY_GB = float(os.environ.get("MIN_FREE_MEMORY_GB", "32"))

# IMPORTANT: Qwen2-VL video tokens grow quickly with long videos.
# We cap frames and per-frame pixels to avoid OOM in large-scale runs.
BASE_NFRAMES = 24
NFRAMES_JITTER_CHOICES = [0, 4, 8]
MAX_FRAMES_HARD = 32
MAX_FRAME_PIXELS = 256 * 256
MIN_FRAME_PIXELS = 56 * 56

CODE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CODE_ROOT / "data/videos/GenVideos/my_videos")))
VIDEO_DIR = os.environ.get("VIDEO_DIR", str(DATA_ROOT / "alpha_gradients_videos"))
ORIGINAL_DIR = os.environ.get("ORIGINAL_DIR", str(DATA_ROOT / "Videos"))
PROMPT_CSV = os.environ.get("PROMPT_CSV", str(CODE_ROOT / "data_metadata/video_to_prompt_full.csv"))
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    str(CODE_ROOT / "outputs/LargeScaleEval/results_videoscore_v1_with_variance.csv"),
)
# ================================

REGRESSION_QUERY_PROMPT = """
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
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


model = None
processor = None
worker_gpu = None


def init_worker(gpu_queue):
    global model, processor, worker_gpu
    worker_gpu = gpu_queue.get()
    device = f"cuda:{worker_gpu}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"Loading VideoScore-v1(Qwen2-VL) on GPU {worker_gpu}...")

    # Prefer flash attention when available; fall back to default attention.
    try:
        model = Qwen2VLForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
    except Exception:
        model = Qwen2VLForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

    processor = Qwen2VLProcessor.from_pretrained(MODEL_NAME)


def get_base_filename(video_file):
    base = re.sub(r'_alpha_([0-1]\.\d+)\.mp4$', '.mp4', video_file)
    if base == video_file:
        base = video_file.replace('_sora_watermark', '')
    return base


def _is_oom_error(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or "cuda out of memory" in str(exc).lower()


def _build_oom_result(video_filename, prompt):
    return {
        "video": video_filename,
        "prompt": prompt,
        "visual_quality_mean": np.nan,
        "visual_quality_std": np.nan,
        "temporal_consistency_mean": np.nan,
        "temporal_consistency_std": np.nan,
        "dynamic_degree_mean": np.nan,
        "dynamic_degree_std": np.nan,
        "text_to_video_alignment_mean": np.nan,
        "text_to_video_alignment_std": np.nan,
        "factual_consistency_mean": np.nan,
        "factual_consistency_std": np.nan,
    }


def get_available_gpu_devices(candidate_gpus, min_free_memory_gb):
    min_free_mib = int(min_free_memory_gb * 1024)
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, text=True)
        free_map = {}
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            gpu_idx = int(parts[0])
            free_mib = int(parts[1])
            free_map[gpu_idx] = free_mib

        available = [g for g in candidate_gpus if free_map.get(g, 0) >= min_free_mib]
        skipped = [(g, free_map.get(g, 0)) for g in candidate_gpus if g not in available]
        if skipped:
            skipped_msg = ", ".join([f"GPU{g}:{m}MiB" for g, m in skipped])
            print(f"Skipping busy GPUs (<{min_free_mib} MiB free): {skipped_msg}")
        return available
    except Exception as e:
        print(f"Warning: GPU preflight check failed ({e}). Falling back to all configured GPUs.")
        return list(candidate_gpus)


def evaluate_video(video_path, prompt):
    global model, processor, worker_gpu
    video_filename = os.path.basename(video_path)
    device = f"cuda:{worker_gpu}"

    try:
        pass_scores = []
        for _ in range(NUM_PASSES):
            nframes_target = int(np.clip(BASE_NFRAMES + np.random.choice(NFRAMES_JITTER_CHOICES), 8, MAX_FRAMES_HARD))
            retry_settings = [
                (nframes_target, MAX_FRAME_PIXELS),
                (16, 224 * 224),
                (12, 196 * 196),
            ]

            outputs = None
            last_exc = None
            for attempt_idx, (nframes, max_frame_pixels) in enumerate(retry_settings, start=1):
                try:
                    # qwen-vl-utils internally computes effective max_pixels from total_pixels and nframes.
                    # For FRAME_FACTOR=2, setting total_pixels = nframes * max_pixels / 2 keeps effective
                    # per-frame cap near max_frame_pixels while avoiding gigantic token counts.
                    total_pixels_budget = int(max(nframes * max_frame_pixels / 2, nframes * MIN_FRAME_PIXELS))
                    min_pixels_for_attempt = int(min(MIN_FRAME_PIXELS, max_frame_pixels))

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": video_path,
                                    "nframes": int(nframes),
                                    "max_pixels": int(max_frame_pixels),
                                    "min_pixels": min_pixels_for_attempt,
                                    "total_pixels": total_pixels_budget,
                                },
                                {"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=prompt)},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        **video_kwargs,
                    )
                    inputs = inputs.to(device)

                    with torch.inference_mode():
                        outputs = model(**inputs)
                    break
                except (RuntimeError, ValueError) as e:
                    last_exc = e
                    is_pixel_error = "max_pixels" in str(e).lower() and "min_pixels" in str(e).lower()
                    if (_is_oom_error(e) or is_pixel_error) and attempt_idx < len(retry_settings):
                        print(
                            f"Retry on {video_filename} (GPU {worker_gpu}): "
                            f"attempt {attempt_idx}/{len(retry_settings)} with nframes={nframes}, "
                            f"max_pixels={max_frame_pixels}."
                        )
                        torch.cuda.empty_cache()
                        continue
                    raise

            if outputs is None and last_exc is not None:
                raise last_exc

            logits = outputs.logits
            scores = [float(logits[0, i].item()) for i in range(logits.shape[-1])]

            # Defensive: keep exactly 5 metrics (vq, tc, dd, t2v, factual)
            if len(scores) >= 5:
                pass_scores.append(scores[:5])
            else:
                pass_scores.append([np.nan, np.nan, np.nan, np.nan, np.nan])

        pass_scores = np.array(pass_scores, dtype=float)
        means = np.nanmean(pass_scores, axis=0)
        stds = np.nanstd(pass_scores, axis=0)

        return {
            "video": video_filename,
            "prompt": prompt,
            "visual_quality_mean": round(means[0], 3) if not np.isnan(means[0]) else np.nan,
            "visual_quality_std": round(stds[0], 3) if not np.isnan(stds[0]) else np.nan,
            "temporal_consistency_mean": round(means[1], 3) if not np.isnan(means[1]) else np.nan,
            "temporal_consistency_std": round(stds[1], 3) if not np.isnan(stds[1]) else np.nan,
            "dynamic_degree_mean": round(means[2], 3) if not np.isnan(means[2]) else np.nan,
            "dynamic_degree_std": round(stds[2], 3) if not np.isnan(stds[2]) else np.nan,
            "text_to_video_alignment_mean": round(means[3], 3) if not np.isnan(means[3]) else np.nan,
            "text_to_video_alignment_std": round(stds[3], 3) if not np.isnan(stds[3]) else np.nan,
            "factual_consistency_mean": round(means[4], 3) if not np.isnan(means[4]) else np.nan,
            "factual_consistency_std": round(stds[4], 3) if not np.isnan(stds[4]) else np.nan,
        }
    except RuntimeError as e:
        if _is_oom_error(e):
            print(f"OOM on {video_filename} (GPU {worker_gpu}) after all retries: {e}")
            torch.cuda.empty_cache()
            return _build_oom_result(video_filename, prompt)
        print(f"Error on {video_filename} (GPU {worker_gpu}): {e}")
        return _build_oom_result(video_filename, prompt)
    except Exception as e:
        print(f"Error on {video_filename} (GPU {worker_gpu}): {e}")
        return _build_oom_result(video_filename, prompt)


def load_prompts():
    df = pd.read_csv(PROMPT_CSV)
    return {row["filename"]: row["prompt"] for _, row in df.iterrows()}


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    prompts_map = load_prompts()

    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not videos:
        print("No videos found in VIDEO_DIR!")

    original_videos = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4")))

    tasks = []
    for v in videos:
        basename = get_base_filename(os.path.basename(v))
        prompt = prompts_map.get(basename, "A high quality video.")
        tasks.append((v, prompt))

    added_originals = 0
    for v in original_videos:
        b = os.path.basename(v)
        prompt = prompts_map.get(b, "A high quality video.")
        tasks.append((v, prompt))
        added_originals += 1

    if added_originals:
        print(f"Added {added_originals} original baseline videos to evaluation.")

    # Resume support
    existing_results = []
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            success_mask = existing_df["visual_quality_mean"].notna()
            completed_videos = set(existing_df.loc[success_mask, "video"].values)
            existing_results = existing_df.loc[success_mask].to_dict("records")
            tasks = [(v, p) for v, p in tasks if os.path.basename(v) not in completed_videos]
            failed_count = int((~success_mask).sum())
            print(
                f"Found {len(completed_videos)} successful videos and {failed_count} failed videos in existing CSV. "
                f"{len(tasks)} videos remaining (failed videos will be retried)."
            )
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    if len(tasks) == 0:
        print("All videos are already evaluated!")
        return

    available_gpus = get_available_gpu_devices(GPU_DEVICES, MIN_FREE_MEMORY_GB)
    if not available_gpus:
        raise RuntimeError(
            f"No GPU has enough free memory. Required >= {MIN_FREE_MEMORY_GB:.1f} GB free per GPU. "
            "Please wait for other jobs to finish or lower MIN_FREE_MEMORY_GB."
        )

    print(
        f"Starting VideoScore-v1(Qwen2-VL) inference (NUM_PASSES={NUM_PASSES}) on {len(tasks)} videos "
        f"across {len(available_gpus)} GPUs: {available_gpus}"
    )

    m = Manager()
    gpu_queue = m.Queue()
    for g in available_gpus:
        gpu_queue.put(g)

    results = existing_results
    with ProcessPoolExecutor(
        max_workers=len(available_gpus),
        initializer=init_worker,
        initargs=(gpu_queue,),
        mp_context=__import__("multiprocessing").get_context("spawn"),
    ) as executor:
        futures = {executor.submit(evaluate_video, v, p): v for v, p in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="VideoScore-v1 Eval (mean/std)"):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
