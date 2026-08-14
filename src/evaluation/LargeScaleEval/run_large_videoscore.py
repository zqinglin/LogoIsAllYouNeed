import os
import glob
import pandas as pd
import numpy as np
import re
import torch
import av
from PIL import Image
import sys
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
MODEL_NAME = "TIGER-Lab/VideoScore-v1.1" 
MAX_FRAMES = 48
NUM_PASSES = 3  # 我们定义三次采样取平均

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CODE_ROOT / "data/videos/GenVideos/my_videos")))
VIDEO_DIR = os.environ.get("VIDEO_DIR", str(DATA_ROOT / "alpha_gradients_videos"))
ORIGINAL_DIR = os.environ.get("ORIGINAL_DIR", str(DATA_ROOT / "Videos"))
PROMPT_CSV = os.environ.get("PROMPT_CSV", str(CODE_ROOT / "data_metadata/video_to_prompt_full.csv"))
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    str(CODE_ROOT / "outputs/LargeScaleEval/results_videoscore_with_variance.csv"),
)
# ================================

VS1_1_REGRESSION_QUERY_PROMPT = """Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
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

def init_worker(gpu_queue):
    global model, processor, worker_gpu
    worker_gpu = gpu_queue.get()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_gpu)
    device = f"cuda:{worker_gpu}"
    print(f"Loading VideoScore on GPU {worker_gpu}...")
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
        if i > indices_list[-1]: break
        if i in indices_list: frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

def evaluate_video(video_path, prompt):
    global model, processor, worker_gpu
    video_filename = os.path.basename(video_path)
    device = f"cuda:{worker_gpu}"
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        
        pass_scores = []
        for pass_idx in range(NUM_PASSES):
            if total_frames > MAX_FRAMES:
                # 为了三次结果不同，我们加入随机偏移 (Random variation in uniform sampling)
                # 例如每次采样的步长微调或者在每个区间的范围内随机取一帧
                step = total_frames / MAX_FRAMES
                # 在 [i*step, (i+1)*step) 里随机取一个整数索引作为这一下的抽样帧
                indices = np.array([int(np.random.uniform(i*step, min((i+1)*step, total_frames))) for i in range(MAX_FRAMES)])
            else:
                indices = np.arange(total_frames)

            frames_np = _read_video_pyav(container, indices)
            frames = [Image.fromarray(x) for x in frames_np]
                
            eval_prompt = VS1_1_REGRESSION_QUERY_PROMPT.format(text_prompt=prompt)
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
                # 帧数不足 MAX_FRAMES 时，三次采样的帧如果没做 crop 必然一模一样。如果无需模拟方差直接 break。
                # 但为了保证有3个记录，直接重复添加3次一样的即可
                pass_scores = [scores, scores, scores]
                break
                
        pass_scores = np.array(pass_scores) # Shape: (3, 5)
        means = np.mean(pass_scores, axis=0) # Shape: (5,)
        stds = np.std(pass_scores, axis=0)   # Shape: (5,)
        
        return {
            'video': video_filename,
            'prompt': prompt,
            'visual_quality_mean': round(means[0], 3), 'visual_quality_std': round(stds[0], 3),
            'temporal_consistency_mean': round(means[1], 3), 'temporal_consistency_std': round(stds[1], 3),
            'dynamic_degree_mean': round(means[2], 3), 'dynamic_degree_std': round(stds[2], 3),
            'text_to_video_alignment_mean': round(means[3], 3), 'text_to_video_alignment_std': round(stds[3], 3),
            'factual_consistency_mean': round(means[4], 3), 'factual_consistency_std': round(stds[4], 3)
        }
    except Exception as e:
        print(f"Error on {video_filename} (GPU {worker_gpu}): {e}")
        return {
            'video': video_filename, 'prompt': prompt,
            'visual_quality_mean': np.nan, 'visual_quality_std': np.nan,
            'temporal_consistency_mean': np.nan, 'temporal_consistency_std': np.nan,
            'dynamic_degree_mean': np.nan, 'dynamic_degree_std': np.nan,
            'text_to_video_alignment_mean': np.nan, 'text_to_video_alignment_std': np.nan,
            'factual_consistency_mean': np.nan, 'factual_consistency_std': np.nan,
        }

def load_prompts():
    df = pd.read_csv(PROMPT_CSV)
    return {row["filename"]: row["prompt"] for idx, row in df.iterrows()}

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    prompts_map = load_prompts()
    
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not videos:
        print("No videos found in VIDEO_DIR!")

    # Also include original (non-alpha) videos as baseline if present
    original_videos = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4")))

    # Build a map of already included basenames to avoid duplicates
    included_basenames = set(get_base_filename(os.path.basename(v)) for v in videos)

    tasks = []
    # add watermarked/alpha videos first
    for v in videos:
        basename = get_base_filename(os.path.basename(v))
        prompt = prompts_map.get(basename, "A high quality video.")
        tasks.append((v, prompt))

    # then add originals. We want to evaluate the original videos even if their watermarked counterparts are present!
    added_originals = 0
    for v in original_videos:
        b = os.path.basename(v)
        prompt = prompts_map.get(b, "A high quality video.")
        tasks.append((v, prompt))
        added_originals += 1
    if added_originals:
        print(f"Added {added_originals} original baseline videos to evaluation.")
        
    print(f"Starting VideoScore inference (NUM_PASSES={NUM_PASSES}) on {len(tasks)} videos across {len(GPU_DEVICES)} GPUs...")
    
    # Check for existing results and skip already evaluated videos
    existing_results = []
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            completed_videos = set(existing_df['video'].values)
            existing_results = existing_df.to_dict('records')
            tasks = [(v, p) for v, p in tasks if os.path.basename(v) not in completed_videos]
            print(f"Found {len(completed_videos)} already processed videos. {len(tasks)} videos remaining.")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")
    
    if len(tasks) == 0:
        print("All videos are already evaluated!")
    else:
        m = Manager()
        gpu_queue = m.Queue()
        for g in GPU_DEVICES:
            gpu_queue.put(g)

        results = existing_results
        with ProcessPoolExecutor(max_workers=len(GPU_DEVICES), initializer=init_worker, initargs=(gpu_queue,), mp_context=__import__("multiprocessing").get_context("spawn")) as executor:
            futures = {executor.submit(evaluate_video, v, p): v for v, p in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="VideoScore Eval (mean/std)"):
                results.append(future.result())

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
