import os
import glob
import pandas as pd
import numpy as np
import re
import json
import time
import torch
import av
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Manager
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from huggingface_hub import snapshot_download
from pathlib import Path

# ================================
GPU_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
MODEL_PATH = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
MAX_FRAMES = 16
NUM_PASSES = 3

CODE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CODE_ROOT / "data/videos/GenVideos/my_videos")))
VIDEO_DIR = os.environ.get("VIDEO_DIR", str(DATA_ROOT / "alpha_gradients_videos"))
PROMPT_CSV = os.environ.get("PROMPT_CSV", str(CODE_ROOT / "data_metadata/video_to_prompt_full.csv"))
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    str(CODE_ROOT / "outputs/LargeScaleEval/results_llava_onevision_with_variance.csv"),
)
# ================================
ORIGINAL_DIR = os.environ.get("ORIGINAL_DIR", str(DATA_ROOT / "Videos"))

UNIFIED_VLM_PROMPT = """Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video).

Output STRICTLY JSON exactly like the following block without any other text:
{
    "visual_quality": 3.2,
    "temporal_consistency": 2.7,
    "dynamic_degree": 4.0,
    "text_to_video_alignment": 2.3,
    "factual_consistency": 1.8
}

For this video, the text prompt is "{text_prompt}"
"""

model = None
processor = None
worker_gpu = None


def resolve_model_path(repo_id):
    """Resolve a local snapshot path in the main process to avoid worker-time network calls."""
    # First, prefer strictly local cache to avoid network instability during long runs.
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        pass

    # Fallback: one-time online resolution/download in main process with retries.
    last_err = None
    for _ in range(3):
        try:
            return snapshot_download(repo_id=repo_id, resume_download=True, max_workers=1)
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Failed to resolve model path for {repo_id}: {last_err}")

def init_worker(gpu_queue, model_local_path):
    global model, processor, worker_gpu
    worker_gpu = gpu_queue.get()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_gpu)
    device = f"cuda:{worker_gpu}"
    print(f"Loading LLaVA-OneVision on GPU {worker_gpu}...")
    processor = AutoProcessor.from_pretrained(model_local_path, local_files_only=True, use_fast=False)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_local_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
    ).to(device).eval()
    # Avoid repeated generation warnings by explicitly setting pad token.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

def get_base_filename(video_file):
    base = re.sub(r'_alpha_([0-1]\.\d+)\.mp4$', '.mp4', video_file)
    if base == video_file: base = video_file.replace('_sora_watermark', '')
    return base

def extract_frames(container, max_frames=16, random_sample=False):
    total_frames = container.streams.video[0].frames
    if total_frames > max_frames:
        if random_sample:
            step = total_frames / max_frames
            indices = np.array([int(np.random.uniform(i*step, min((i+1)*step, total_frames))) for i in range(max_frames)])
        else:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]: break
        if i in indices: frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

def safe_float(val):
    try:
        if val is None: return np.nan
        return float(val)
    except:
        return np.nan

def parse_json_output(output_txt):
    try:
        match = re.search(r'\{[^{}]*\}', output_txt, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {"visual_quality": np.nan, "temporal_consistency": np.nan, "dynamic_degree": np.nan, "text_to_video_alignment": np.nan, "factual_consistency": np.nan}

def evaluate_video(video_path, prompt):
    global model, processor, worker_gpu
    video_filename = os.path.basename(video_path)
    device = f"cuda:{worker_gpu}"
    generated_text = ""
    try:
        container = av.open(video_path)
        pass_scores = []
        for pass_idx in range(NUM_PASSES):
            video_frames = extract_frames(container, max_frames=MAX_FRAMES, random_sample=True)
            chat_prompt = [
                {"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": UNIFIED_VLM_PROMPT.replace("{text_prompt}", prompt)}
                ]}
            ]
            formatted_prompt = processor.apply_chat_template(chat_prompt, add_generation_prompt=True)
            inputs = processor(text=formatted_prompt, videos=video_frames, return_tensors="pt").to(device, torch.float16)

            out = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=model.generation_config.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated_text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            scores = parse_json_output(generated_text)
            pass_scores.append([
                safe_float(scores.get("visual_quality", np.nan)),
                safe_float(scores.get("temporal_consistency", np.nan)),
                safe_float(scores.get("dynamic_degree", np.nan)),
                safe_float(scores.get("text_to_video_alignment", np.nan)),
                safe_float(scores.get("factual_consistency", np.nan))
            ])
            
            if container.streams.video[0].frames <= MAX_FRAMES:
                pass_scores = [pass_scores[-1]] * NUM_PASSES
                break
                
        pass_scores = np.array(pass_scores, dtype=float)
        means = np.nanmean(pass_scores, axis=0)
        stds = np.nanstd(pass_scores, axis=0)
        
        return {
            'video': video_filename,
            'prompt': prompt,
            'visual_quality_mean': round(means[0], 3) if not np.isnan(means[0]) else np.nan, 'visual_quality_std': round(stds[0], 3) if not np.isnan(stds[0]) else np.nan,
            'temporal_consistency_mean': round(means[1], 3) if not np.isnan(means[1]) else np.nan, 'temporal_consistency_std': round(stds[1], 3) if not np.isnan(stds[1]) else np.nan,
            'dynamic_degree_mean': round(means[2], 3) if not np.isnan(means[2]) else np.nan, 'dynamic_degree_std': round(stds[2], 3) if not np.isnan(stds[2]) else np.nan,
            'text_to_video_alignment_mean': round(means[3], 3) if not np.isnan(means[3]) else np.nan, 'text_to_video_alignment_std': round(stds[3], 3) if not np.isnan(stds[3]) else np.nan,
            'factual_consistency_mean': round(means[4], 3) if not np.isnan(means[4]) else np.nan, 'factual_consistency_std': round(stds[4], 3) if not np.isnan(stds[4]) else np.nan
        }
    except Exception as e:
        # Provide more debug info for parsing/generation failures
        try:
            print(f"Error on {video_filename} (GPU {worker_gpu}): {repr(e)}")
            print("--- Generated text (first 2000 chars) ---")
            print(generated_text[:2000])
        except Exception:
            pass
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

    # Resolve model snapshot once in main process; workers then load locally only.
    model_local_path = resolve_model_path(MODEL_PATH)
    
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not videos:
        print("No videos found in VIDEO_DIR!")

    # include originals as baseline
    original_videos = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4")))

    tasks = [(v, prompts_map.get(get_base_filename(os.path.basename(v)), "A high quality video.")) for v in videos]
    added_originals = 0
    for v in original_videos:
        b = os.path.basename(v)
        tasks.append((v, prompts_map.get(b, "A high quality video.")))
        added_originals += 1
    if added_originals:
        print(f"Added {added_originals} original baseline videos to evaluation.")

    print(f"Starting LLaVA-OneVision inference on {len(tasks)} videos...")

    # Resume support: skip videos that already exist in OUTPUT_CSV
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
        with ProcessPoolExecutor(max_workers=len(GPU_DEVICES), initializer=init_worker, initargs=(gpu_queue, model_local_path), mp_context=__import__("multiprocessing").get_context("spawn")) as executor:
            futures = {executor.submit(evaluate_video, v, p): v for v, p in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLaVA OneVision Eval (mean/std)"):
                results.append(future.result())

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
