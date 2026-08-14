
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from multiprocessing import Pool, Manager
import torch
import av
from PIL import Image
from pathlib import Path

# --- [START] User Configuration ---

# List of GPU device IDs to use for parallel execution.
GPU_DEVICES = [0, 4, 6, 7]

# Dictionary of VideoScore models to test.
# The script will iterate through this dict, running a full evaluation for each model.
MODELS_TO_TEST = {
    "v1": "TIGER-Lab/VideoScore",
    "v1.1": "TIGER-Lab/VideoScore-v1.1"
}

# --- [END] User Configuration ---

# --- Script Internal Paths ---
# Add the VideoScore project to the path for internal imports
CODE_ROOT = Path(__file__).resolve().parents[3]
VIDEOSCORE_PROJECT_PATH = os.environ.get(
    "VIDEOSCORE_PROJECT_PATH",
    str(CODE_ROOT / "src/evaluation/VideoScore"),
)
import sys
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CODE_ROOT / "data/videos/GenVideos/my_videos")))
VIDEO_DIRECTORIES = {
    "original": str(DATA_ROOT / "Videos"),
    "watermarked": str(DATA_ROOT / "watermarked_videos"),
}
PROMPT_CSV_PATH = os.environ.get(
    "PROMPT_CSV_PATH",
    str(CODE_ROOT / "data_metadata/video_to_prompt_full.csv"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(CODE_ROOT / "outputs/evaluation_results"))

# --- VideoScore v1.1 Evaluation Logic (adapted for workers) ---
try:
    from transformers import AutoProcessor
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as e:
    print(f"FATAL: Could not import VideoScore/Mantis modules: {e}")
    print("Please ensure you have activated the 'VideoScore1_env' and installed the project with 'pip install -e .'")
    sys.exit(1)

VS1_REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality, (2) temporal consistency, (3) dynamic degree, (4) text-to-video alignment, (5) factual consistency
For this video, the text prompt is "{text_prompt}", all the frames of video are as follows:
"""

def _read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index, end_index = indices[0], indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index: break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

def run_videoscore_evaluation(video_path, prompt, model, processor, device, max_frames=48):
    try:
        with av.open(video_path) as container:
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / max_frames).astype(int) if total_frames > max_frames else np.arange(total_frames)
            
            frames = [Image.fromarray(x) for x in _read_video_pyav(container, indices)]
            eval_prompt = VS1_REGRESSION_QUERY_PROMPT.format(text_prompt=prompt) + "<image> " * len(frames)

            inputs = processor(text=eval_prompt, images=[frames], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits[0]
            
            scores = [round(s.item(), 3) for s in logits]
            total_score = round(sum(scores) / len(scores), 3) if scores else 0.0
            return scores + [total_score]
    except Exception:
        return [np.nan] * 6 # 5 scores + 1 total

def process_video_task(task_data):
    """Worker function: loads model, runs evals for one video, and cleans up."""
    video_path, prompt, model_name, gpu_id, video_file = task_data
    device = f"cuda:{gpu_id}"
    model, processor = None, None # Ensure they exist for the finally block

    try:
        # Each process loads its own model instance
        model = Idefics2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        
        scores = run_videoscore_evaluation(video_path, prompt, model, processor, device)
    except Exception:
        scores = [np.nan] * 6
    finally:
        # Crucial for releasing memory
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'video_filename': video_file, 'prompt': prompt,
        'vs1_visual_quality': scores[0], 'vs1_temporal_consistency': scores[1],
        'vs1_dynamic_degree': scores[2], 'vs1_text_alignment': scores[3],
        'vs1_factual_consistency': scores[4], 'vs1_total_score': scores[5],
    }

def main():
    print("--- VideoScore v1/v1.1 Parallel Evaluation ---")
    
    try:
        prompts_df = pd.read_csv(PROMPT_CSV_PATH)
        prompt_map = {row[0]: row[1] for row in prompts_df.itertuples(index=False)}
    except FileNotFoundError:
        print(f"FATAL: Prompt CSV not found at '{PROMPT_CSV_PATH}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_version_key, model_name in MODELS_TO_TEST.items():
        print(f"\n===== Starting Evaluation for Model: {model_version_key} ({model_name}) =====")
        for dir_key, video_dir in VIDEO_DIRECTORIES.items():
            print(f"\n--- Processing directory: {dir_key} ({video_dir}) ---")
            
            tasks = []
            for video_file in sorted(os.listdir(video_dir)):
                if not video_file.endswith('.mp4'): continue
                prompt_val = prompt_map.get(video_file)
                if isinstance(prompt_val, str) and prompt_val.strip():
                    # Pass all necessary info to the worker
                    tasks.append((os.path.join(video_dir, video_file), prompt_val, model_name, -1, video_file))
            
            if not tasks: continue

            # --- Setup and run multiprocessing pool ---
            manager = Manager()
            gpu_queue = manager.Queue()
            for gpu_id in GPU_DEVICES:
                gpu_queue.put(gpu_id)

            # Re-create tasks with assigned GPU IDs
            tasks_with_gpu = []
            for i, task in enumerate(tasks):
                gpu_id = GPU_DEVICES[i % len(GPU_DEVICES)]
                tasks_with_gpu.append(task[:-2] + (gpu_id, task[-1]))

            results = []
            with Pool(processes=len(GPU_DEVICES)) as pool:
                with tqdm(total=len(tasks_with_gpu), desc=f"Evaluating {dir_key}") as pbar:
                    for result in pool.imap_unordered(process_video_task, tasks_with_gpu):
                        results.append(result)
                        pbar.update()

            # --- Save Results ---
            if not results: continue
            
            results_df = pd.DataFrame(results).sort_values(by='video_filename').reset_index(drop=True)
            filename_prefix = f"videoscore1_{model_version_key}"
            output_csv_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_scores_{dir_key}.csv")
            results_df.to_csv(output_csv_path, index=False)
            print(f"\nResults saved to: {output_csv_path}")

            stats_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_statistics_{dir_key}.txt")
            with open(stats_path, 'w') as f:
                f.write(f"--- Stats for {dir_key} (Model: {model_version_key}) ---\\n\\n")
                f.write(results_df.describe().to_string())
            print(f"Statistics saved to: {stats_path}")

    print("\n--- All VideoScore evaluations completed! ---")

if __name__ == "__main__":
    main()
