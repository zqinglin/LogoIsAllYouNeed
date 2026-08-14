import os
import glob
import pandas as pd
import re
import json
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import av
import numpy as np
from pathlib import Path

# ===== CONFIG =====
CODE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(CODE_ROOT / "data/videos/GenVideos/my_videos")))
VIDEO_DIR = os.environ.get("VIDEO_DIR", str(DATA_ROOT / "alpha_gradients_videos"))
PROMPT_CSV = os.environ.get("PROMPT_CSV", str(CODE_ROOT / "data_metadata/video_to_prompt_full.csv"))
OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    str(CODE_ROOT / "outputs/LargeScaleEval/results_gpt4o_mini_with_variance.csv"),
)
MAX_WORKERS = 16
MAX_FRAMES = 8
NUM_PASSES = 1

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

def get_base_filename(video_file):
    base = re.sub(r'_alpha_([0-1]\.\d+)\.mp4$', '.mp4', video_file)
    if base == video_file:
        base = video_file.replace('_sora_watermark', '')
    return base

def load_prompts():
    df = pd.read_csv(PROMPT_CSV)
    mapping = {}
    for idx, row in df.iterrows():
        mapping[row["filename"]] = row["prompt"]
    return mapping

def extract_frames(container, max_frames=8, random_sample=False):
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
        if i in indices:
            frames.append(frame.to_image())
            
    return frames

def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def safe_float(val):
    try:
        if val is None: return np.nan
        return float(val)
    except:
        return np.nan

def parse_json_output(output_txt):
    try:
        match = re.search(r'\{[^{}]*\}', output_txt, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return {"visual_quality": np.nan, "temporal_consistency": np.nan, "dynamic_degree": np.nan, "text_to_video_alignment": np.nan, "factual_consistency": np.nan}

def evaluate_video(video_path, prompt, api_key):
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        container = av.open(video_path)
        pass_scores = []
        for pass_idx in range(NUM_PASSES):
            frames = extract_frames(container, max_frames=MAX_FRAMES, random_sample=True)
            content = [{"type": "text", "text": UNIFIED_VLM_PROMPT.replace("{text_prompt}", prompt)}]
            for frame in frames:
                b64 = encode_image_base64(frame)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                temperature=0.0
            )
            scores = parse_json_output(response.choices[0].message.content)
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
            'video': os.path.basename(video_path),
            'prompt': prompt,
            'visual_quality_mean': round(means[0], 3) if not np.isnan(means[0]) else np.nan, 'visual_quality_std': round(stds[0], 3) if not np.isnan(stds[0]) else np.nan,
            'temporal_consistency_mean': round(means[1], 3) if not np.isnan(means[1]) else np.nan, 'temporal_consistency_std': round(stds[1], 3) if not np.isnan(stds[1]) else np.nan,
            'dynamic_degree_mean': round(means[2], 3) if not np.isnan(means[2]) else np.nan, 'dynamic_degree_std': round(stds[2], 3) if not np.isnan(stds[2]) else np.nan,
            'text_to_video_alignment_mean': round(means[3], 3) if not np.isnan(means[3]) else np.nan, 'text_to_video_alignment_std': round(stds[3], 3) if not np.isnan(stds[3]) else np.nan,
            'factual_consistency_mean': round(means[4], 3) if not np.isnan(means[4]) else np.nan, 'factual_consistency_std': round(stds[4], 3) if not np.isnan(stds[4]) else np.nan
        }
    except Exception as e:
        print(f"Error on {os.path.basename(video_path)}: {e}")
        return {
            'video': os.path.basename(video_path), 'prompt': prompt,
            'visual_quality_mean': np.nan, 'visual_quality_std': np.nan,
            'temporal_consistency_mean': np.nan, 'temporal_consistency_std': np.nan,
            'dynamic_degree_mean': np.nan, 'dynamic_degree_std': np.nan,
            'text_to_video_alignment_mean': np.nan, 'text_to_video_alignment_std': np.nan,
            'factual_consistency_mean': np.nan, 'factual_consistency_std': np.nan,
        }

def main():
    prompts = load_prompts()
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY")
        return
        
    tasks = []
    for v in videos:
        basename = get_base_filename(os.path.basename(v))
        prompt = prompts.get(basename, "A high quality AI-generated video.")
        tasks.append((v, prompt))
        
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_video, t[0], t[1], api_key): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="GPT-4o Eval (mean/std)"):
            results.append(future.result())
            
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
