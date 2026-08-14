import os
import glob
import pandas as pd
import numpy as np
import re
import torch
import av
from PIL import Image
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Manager
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
VIDEOSCORE_PROJECT_PATH = os.environ.get(
    "VIDEOSCORE_PROJECT_PATH",
    str(CODE_ROOT / "src/evaluation/VideoScore"),
)
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification

# ================================
GPU_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
MODEL_NAME = "TIGER-Lab/VideoScore"
MAX_FRAMES = 48
NUM_PASSES = 3
MIN_FREE_MEMORY_GB = float(os.environ.get("MIN_FREE_MEMORY_GB", "30"))

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CODE_ROOT / "data/videos/GenVideos/my_videos")))
VIDEO_DIR = os.environ.get("VIDEO_DIR", str(DATA_ROOT / "alpha_gradients_videos"))
ORIGINAL_DIR = os.environ.get("ORIGINAL_DIR", str(DATA_ROOT / "Videos"))
PROMPT_CSV = os.environ.get("PROMPT_CSV", str(CODE_ROOT / "data_metadata/video_to_prompt_full.csv"))
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    str(CODE_ROOT / "outputs/LargeScaleEval/results_videoscore_v1_official_with_variance.csv"),
)
# ================================

VS1_REGRESSION_QUERY_PROMPT = """Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
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

# Global Process Variables
model = None
processor = None
worker_gpu = None


def preflight_dependencies():
    # Fail fast in main process with a clear message instead of BrokenProcessPool in worker init.
    try:
        import google.protobuf  # noqa: F401
        from sentencepiece import sentencepiece_model_pb2  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing protobuf/sentencepiece runtime in current environment. "
            "Run: pip install protobuf sentencepiece"
        ) from e


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


def init_worker(gpu_queue):
    global model, processor, worker_gpu
    worker_gpu = gpu_queue.get()
    device = f"cuda:{worker_gpu}"
    print(f"Loading VideoScore-v1 (official) on GPU {worker_gpu}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()


def get_base_filename(video_file):
    base = re.sub(r'_alpha_([0-1]\.\d+)\.mp4$', '.mp4', video_file)
    if base == video_file:
        base = video_file.replace('_sora_watermark', '')
    return base


def _read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    indices_list = indices.tolist()
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices_list[-1]:
            break
        if i in indices_list:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def evaluate_video(video_path, prompt):
    global model, processor, worker_gpu
    video_filename = os.path.basename(video_path)
    device = f"cuda:{worker_gpu}"
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames

        pass_scores = []
        for _ in range(NUM_PASSES):
            if total_frames > MAX_FRAMES:
                step = total_frames / MAX_FRAMES
                indices = np.array(
                    [int(np.random.uniform(i * step, min((i + 1) * step, total_frames))) for i in range(MAX_FRAMES)]
                )
            else:
                indices = np.arange(total_frames)

            frames_np = _read_video_pyav(container, indices)
            frames = [Image.fromarray(x) for x in frames_np]

            eval_prompt = VS1_REGRESSION_QUERY_PROMPT.format(text_prompt=prompt)
            num_image_token = eval_prompt.count("<image>")
            if num_image_token < len(frames):
                eval_prompt += "<image> " * (len(frames) - num_image_token)

            inputs = processor(text=eval_prompt, images=frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits[0]

            scores = [s.item() for s in logits]
            pass_scores.append(scores)

            if total_frames <= MAX_FRAMES:
                pass_scores = [scores, scores, scores]
                break

        pass_scores = np.array(pass_scores)
        means = np.mean(pass_scores, axis=0)
        stds = np.std(pass_scores, axis=0)

        return {
            "video": video_filename,
            "prompt": prompt,
            "visual_quality_mean": round(means[0], 3),
            "visual_quality_std": round(stds[0], 3),
            "temporal_consistency_mean": round(means[1], 3),
            "temporal_consistency_std": round(stds[1], 3),
            "dynamic_degree_mean": round(means[2], 3),
            "dynamic_degree_std": round(stds[2], 3),
            "text_to_video_alignment_mean": round(means[3], 3),
            "text_to_video_alignment_std": round(stds[3], 3),
            "factual_consistency_mean": round(means[4], 3),
            "factual_consistency_std": round(stds[4], 3),
        }
    except Exception as e:
        print(f"Error on {video_filename} (GPU {worker_gpu}): {e}")
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


def load_prompts():
    df = pd.read_csv(PROMPT_CSV)
    return {row["filename"]: row["prompt"] for _, row in df.iterrows()}


def main():
    preflight_dependencies()
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
        f"Starting VideoScore-v1 (official) inference (NUM_PASSES={NUM_PASSES}) on {len(tasks)} videos "
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="VideoScore-v1 (official) Eval (mean/std)"):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
