
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import torch
import av
from PIL import Image
from pathlib import Path

# --- [START] User Configuration ---
GPU_DEVICES = [0, 4, 6, 7]
MODEL_NAME = "TIGER-Lab/VideoScore"
# --- [END] User Configuration ---

# --- Path Definitions ---
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

# --- Evaluation Logic ---
try:
    from transformers import AutoProcessor
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as e:
    print(f"FATAL: Could not import VideoScore/Mantis modules: {e}")
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
    except Exception: return [np.nan] * 6

def process_video_task(task_data):
    video_path, prompt, gpu_id, video_file = task_data
    device = f"cuda:{gpu_id}"
    model, processor = None, None
    try:
        model = Idefics2ForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        scores = run_videoscore_evaluation(video_path, prompt, model, processor, device)
    except Exception: scores = [np.nan] * 6
    finally:
        del model, processor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {
        'video_filename': video_file, 'prompt': prompt,
        'vs1_visual_quality': scores[0], 'vs1_temporal_consistency': scores[1],
        'vs1_dynamic_degree': scores[2], 'vs1_text_alignment': scores[3],
        'vs1_factual_consistency': scores[4], 'vs1_total_score': scores[5],
    }

def main():
    print(f"--- VideoScore v1.0 Final Parallel Evaluation ---")
    try:
        prompts_df = pd.read_csv(PROMPT_CSV_PATH)
        prompt_map = {row[0]: row[1] for row in prompts_df.itertuples(index=False)}
    except FileNotFoundError: print(f"FATAL: Prompt CSV not found at '{PROMPT_CSV_PATH}'"); return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for run_number in range(1, 4):
        print(f"\n--- Starting Run: {run_number}/3 ---")
        for dir_key, video_dir in VIDEO_DIRECTORIES.items():
            print(f"\n--- Processing directory: {dir_key} ({video_dir}) ---")
            tasks, tasks_with_gpu = [], []
            for video_file in sorted(os.listdir(video_dir)):
                if not video_file.endswith('.mp4'): continue
                if isinstance(prompt_map.get(video_file), str) and prompt_map.get(video_file).strip():
                    tasks.append((os.path.join(video_dir, video_file), prompt_map.get(video_file), -1, video_file))
            if not tasks: continue
            for i, task in enumerate(tasks): gpu_id = GPU_DEVICES[i % len(GPU_DEVICES)]; tasks_with_gpu.append(task[:-2] + (gpu_id, task[-1]))
            results = []
            with Pool(processes=len(GPU_DEVICES)) as pool:
                with tqdm(total=len(tasks_with_gpu), desc=f"Evaluating {dir_key} (Run {run_number}) ") as pbar:
                    for result in pool.imap_unordered(process_video_task, tasks_with_gpu):
                        results.append(result); pbar.update()
            if not results: continue
            results_df = pd.DataFrame(results).sort_values(by='video_filename').reset_index(drop=True)
            filename_prefix = "videoscore_v1_final"
            output_csv_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_scores_{dir_key}_run_{run_number}.csv")
            results_df.to_csv(output_csv_path, index=False)
            print(f"\nResults saved to: {output_csv_path}")
            stats_path = os.path.join(OUTPUT_DIR, f"{filename_prefix}_statistics_{dir_key}_run_{run_number}.txt")
            with open(stats_path, 'w') as f: f.write(f"--- Stats for {dir_key} (Model: v1 Final, Run {run_number}) ---\\n\\n"); f.write(results_df.describe().to_string())
            print(f"Statistics saved to: {stats_path}")
    print("\n--- VideoScore v1.0 evaluations completed! ---")

if __name__ == "__main__":
    main()
