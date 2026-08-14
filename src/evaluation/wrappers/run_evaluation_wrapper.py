
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import subprocess
from multiprocessing import Pool, Manager
import sys
from pathlib import Path

# --- [START] User Configuration ---

# List of GPU device IDs to use for parallel execution.
# The number of parallel processes will be the number of GPUs in this list.
# Example: [0, 1, 2, 3] or [0, 4, 6, 7]
GPU_DEVICES = [6, 7]
PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", sys.executable)

# --- [END] User Configuration ---

# --- Script Internal Paths ---
CODE_ROOT = Path(__file__).resolve().parents[3]
VIDEOSCORE2_WORKDIR = os.environ.get(
    "VIDEOSCORE2_WORKDIR",
    str(CODE_ROOT / "third_party/VideoScore2"),
)
VIDEOSCORE2_SCRIPT = os.environ.get(
    "VIDEOSCORE2_SCRIPT",
    str(Path(VIDEOSCORE2_WORKDIR) / "vs2_inference.py"),
)
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

# Global variable to hold the GPU ID for each worker process
worker_gpu_id = -1

def init_worker(gpu_queue):
    """Initializer for each worker process in the pool. Grabs a GPU ID from the queue."""
    global worker_gpu_id
    worker_gpu_id = gpu_queue.get()
    # print(f"[PID:{os.getpid()}] Initialized. Assigned to GPU {worker_gpu_id}")

def parse_vs2_output(output_text):
    """Parses the stdout from vs2_inference.py to extract scores."""
    patterns = {
        'vs2_visual_quality': r"Visual Quality:\s*([\d.]+)",
        'vs2_text_alignment': r"Text-to-Video Alignment:\s*([\d.]+)",
        'vs2_physical_consistency': r"Physical Consistency:\s*([\d.]+)"
    }
    scores = {k: np.nan for k in patterns}
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            try:
                scores[key] = float(match.group(1))
            except (ValueError, IndexError):
                pass # Silently fail on parsing error, will be caught by isnan check
    
    scores_list = list(scores.values())
    total_score = round(np.nanmean(scores_list), 4) if not any(np.isnan(s) for s in scores_list) else np.nan
    return scores_list + [total_score]

def run_single_evaluation(video_path, prompt, gpu_id):
    """Runs vs2_inference.py on a specific GPU and returns the parsed scores."""
    command = [PYTHON_EXECUTABLE, VIDEOSCORE2_SCRIPT, "--video_path", video_path, "--t2v_prompt", prompt]
    
    # Set the environment for the subprocess to use a specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=VIDEOSCORE2_WORKDIR,
            env=env, check=True, timeout=300
        )
        return parse_vs2_output(result.stdout)
    except subprocess.CalledProcessError as e:
        # Write errors to a separate log file for easier debugging
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a") as f:
            f.write(f"--- Subprocess Error for {os.path.basename(video_path)} on GPU {gpu_id} ---\\n")
            f.write(f"Stderr: {e.stderr.strip()[:1000]}\n") # Log first 1000 chars of error
        return [np.nan] * 4
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a") as f:
            f.write(f"--- Unknown Error for {os.path.basename(video_path)} on GPU {gpu_id} ---\\n")
            f.write(f"{e}\n")
        return [np.nan] * 4

def process_video_task(task_data):
    """A wrapper function for the worker process to handle a single video task."""
    video_path, prompt, video_file = task_data
    global worker_gpu_id
    
    all_runs_scores = []
    for _ in range(3):
        scores = run_single_evaluation(video_path, prompt, worker_gpu_id)
        if not any(np.isnan(s) for s in scores):
            all_runs_scores.append(scores)
            
    if not all_runs_scores:
        means, stds = ([np.nan] * 4, [np.nan] * 4)
    else:
        scores_array = np.array(all_runs_scores)
        means = np.mean(scores_array, axis=0).round(4)
        stds = np.std(scores_array, axis=0).round(4)

    return {
        'video_filename': video_file, 'prompt': prompt,
        'vs2_visual_quality_mean': means[0], 'vs2_visual_quality_std': stds[0],
        'vs2_text_alignment_mean': means[1], 'vs2_text_alignment_std': stds[1],
        'vs2_physical_consistency_mean': means[2], 'vs2_physical_consistency_std': stds[2],
        'vs2_total_score_mean': means[3], 'vs2_total_score_std': stds[3],
    }

def main():
    # --- [START] Pre-flight check to cache the model ---
    # This prevents race conditions where multiple processes try to download the model simultaneously.
    try:
        print("--- Caching Hugging Face model locally... ---")
        from transformers import AutoModelForVision2Seq, AutoProcessor
        model_name = "TIGER-Lab/VideoScore2"
        AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("--- Model cached successfully. ---")
    except Exception as e:
        print(f"FATAL ERROR: Could not pre-download the Hugging Face model. Error: {e}")
        print("Please check your internet connection and Hugging Face access.")
        return
    # --- [END] Pre-flight check ---

    print("\n--- VideoScore2 Parallel Evaluation Wrapper ---")
    print(f"Using {len(GPU_DEVICES)} GPUs for parallel execution: {GPU_DEVICES}")

    try:
        prompts_df = pd.read_csv(PROMPT_CSV_PATH)
        prompt_map = {row[0]: row[1] for row in prompts_df.itertuples(index=False)}
    except FileNotFoundError:
        print(f"FATAL ERROR: Prompt CSV file not found at '{PROMPT_CSV_PATH}'")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dir_key, video_dir in VIDEO_DIRECTORIES.items():
        print(f"\n--- Processing directory: {dir_key} ({video_dir}) ---")
        
        tasks = []
        for video_file in sorted(os.listdir(video_dir)):
            if not video_file.endswith('.mp4'): continue
            prompt_val = prompt_map.get(video_file)
            if isinstance(prompt_val, str) and prompt_val.strip():
                tasks.append((os.path.join(video_dir, video_file), prompt_val, video_file))
        
        if not tasks:
            print("No valid videos with prompts found, skipping.")
            continue

        # --- Setup and run multiprocessing pool ---
        manager = Manager()
        gpu_queue = manager.Queue()
        for gpu_id in GPU_DEVICES:
            gpu_queue.put(gpu_id)

        output_csv_path = os.path.join(OUTPUT_DIR, f"videoscore2_scores_{dir_key}.csv")
        # Write header first, ensuring all possible columns are present
        pd.DataFrame([], columns=[
            'video_filename', 'prompt',
            'vs2_visual_quality_mean', 'vs2_visual_quality_std',
            'vs2_text_alignment_mean', 'vs2_text_alignment_std',
            'vs2_physical_consistency_mean', 'vs2_physical_consistency_std',
            'vs2_total_score_mean', 'vs2_total_score_std',
        ]).to_csv(output_csv_path, index=False)

        results = []
        with Pool(processes=len(GPU_DEVICES), initializer=init_worker, initargs=(gpu_queue,)) as pool:
            with tqdm(total=len(tasks), desc=f"Evaluating {dir_key}") as pbar:
                for result in pool.imap_unordered(process_video_task, tasks):
                    if result:
                        results.append(result)
                        # Append the new result to the CSV file immediately
                        pd.DataFrame([result]).to_csv(output_csv_path, mode='a', header=False, index=False)
                    pbar.update()

        # --- Save Results and Statistics ---
        if not results:
            print(f"No videos were successfully processed for {dir_key}.")
            continue

        # The main results are already saved incrementally. We just need to load the full CSV
        # for statistical analysis to ensure it's sorted correctly.
        results_df = pd.read_csv(output_csv_path).sort_values(by='video_filename').reset_index(drop=True)
        print(f"\nResults for {dir_key} saved to: {output_csv_path}")

        stats_path = os.path.join(OUTPUT_DIR, f"videoscore2_statistics_{dir_key}.txt")
        with open(stats_path, 'w') as f:
            f.write(f"--- Statistical Analysis for {dir_key} (based on mean scores) ---\\n\\n")
            mean_cols = [col for col in results_df.columns if 'mean' in col]
            f.write(results_df[mean_cols].describe().to_string())
        print(f"Statistics for {dir_key} saved to: {stats_path}")

    print("\n--- All evaluations completed! ---")

if __name__ == "__main__":
    # Ensure the script can be used with multiprocessing on systems that need it (like Windows)
    main()
