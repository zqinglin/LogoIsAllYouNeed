import os
import pandas as pd
import numpy as np
import torch
import av
from PIL import Image
from tqdm import tqdm
from typing import List
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification
import argparse

# --- [1. 配置区：请核对路径] ---
MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"
ORIG_DIR = "data/videos/GenVideos/my_videos/Videos"
PROMPT_CSV_PATH = "data_metadata/video_to_prompt_full.csv"

MAX_NUM_FRAMES = 48
ROUND_DIGIT = 3

# 使用你单视频脚本中完整的 Prompt
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

# --- [2. 核心函数：完全对齐你的工作代码] ---

def _read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_scores_unified(model, processor, device, video_path, text_prompt):
    # 抽帧逻辑完全照搬你的单视频脚本
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    
    if total_frames > MAX_NUM_FRAMES:
        indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
    else:
        indices = np.arange(total_frames)

    frames_np = _read_video_pyav(container, indices)
    frames = [Image.fromarray(x) for x in frames_np]
    container.close()

    # 构造 Prompt
    eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=text_prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)

    # 准备 inputs (完全对齐你的单视频逻辑)
    flatten_images = []
    for x in [frames]:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)
    
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    num_aspects = logits.shape[-1]
    
    aspect_scores = []
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(), ROUND_DIGIT))
    return aspect_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare original vs. watermarked videos.")
    parser.add_argument("--style", type=str, required=True, choices=['sora', 'gemini', 'kling', 'gray', 'sora_flipped'], help="Watermark style to evaluate.")
    args = parser.parse_args()
    style = args.style

    # --- [动态路径配置] ---
    BASE_VIDEO_DIR = "data/videos/GenVideos/my_videos"
    BASE_EVAL_DIR = "outputs"

    style_config = {
        'sora': {
            'wm_dir': os.path.join(BASE_VIDEO_DIR, 'watermarked_videos'),
            'output_csv': os.path.join(BASE_EVAL_DIR, 'comparison_report.csv') # 保留原始文件名以实现断点续跑
        },
        'gemini': {
            'wm_dir': os.path.join(BASE_VIDEO_DIR, 'gemini_videos'),
            'output_csv': os.path.join(BASE_EVAL_DIR, f'comparison_report_gemini.csv')
        },
        'kling': {
            'wm_dir': os.path.join(BASE_VIDEO_DIR, 'kling_videos'),
            'output_csv': os.path.join(BASE_EVAL_DIR, f'comparison_report_kling.csv')
        },
        'gray': {
            'wm_dir': os.path.join(BASE_VIDEO_DIR, 'gray_videos'),
            'output_csv': os.path.join(BASE_EVAL_DIR, f'comparison_report_gray.csv')
        },
        'sora_flipped': {
            'wm_dir': os.path.join(BASE_VIDEO_DIR, 'sora_flipped_videos'),
            'output_csv': os.path.join(BASE_EVAL_DIR, f'comparison_report_sora_flipped.csv')
        }
    }

    if style not in style_config:
        print(f"Error: Invalid style '{style}'. Please choose from {list(style_config.keys())}")
        return

    WM_DIR = style_config[style]['wm_dir']
    OUTPUT_CSV = style_config[style]['output_csv']
    
    print(f"--- Running evaluation for style: {style} ---")
    print(f"Original videos directory: {ORIG_DIR}")
    print(f"Watermarked videos directory: {WM_DIR}")
    print(f"Output report: {OUTPUT_CSV}")
    print("-" * 50)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} to {device}...")
    
    # 严格使用你单视频代码中的初始化参数
    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).eval()
    model.to(device)

    # 加载映射与文件列表
    mapping_df = pd.read_csv(PROMPT_CSV_PATH)
    prompt_map = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))
    
    orig_files = set([f for f in os.listdir(ORIG_DIR) if f.endswith('.mp4')])
    wm_files = set([f for f in os.listdir(WM_DIR) if f.endswith('.mp4')])
    common_files = sorted(list(orig_files.intersection(wm_files)))

    # 断点续跑逻辑
    if os.path.exists(OUTPUT_CSV):
        try:
            done_df = pd.read_csv(OUTPUT_CSV)
            if 'video_filename' in done_df.columns:
                done = set(done_df['video_filename'].tolist())
                common_files = [f for f in common_files if f not in done]
            else:
                print(f"Warning: 'video_filename' column not found in {OUTPUT_CSV}. Will process all videos.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {OUTPUT_CSV} is empty. Will process all videos.")
            # 文件为空，无需处理
            pass

    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        cols = ["video_filename", "prompt", 
                "orig_v", "orig_t", "orig_d", "orig_a", "orig_f",
                "wm_v", "wm_t", "wm_d", "wm_a", "wm_f",
                "orig_total", "wm_total", "delta"]
        pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)

    print(f"Total pairs to evaluate for style '{style}': {len(common_files)}")

    for fname in tqdm(common_files):
        prompt = prompt_map.get(fname)
        if not prompt: continue

        try:
            # 分别对原视频和加水印视频打分
            s_orig = get_scores_unified(model, processor, device, os.path.join(ORIG_DIR, fname), prompt)
            s_wm = get_scores_unified(model, processor, device, os.path.join(WM_DIR, fname), prompt)

            avg_orig = round(float(np.mean(s_orig)), 3)
            avg_wm = round(float(np.mean(s_wm)), 3)

            res = {
                "video_filename": fname,
                "prompt": prompt,
                "orig_v": s_orig[0], "orig_t": s_orig[1], "orig_d": s_orig[2], "orig_a": s_orig[3], "orig_f": s_orig[4],
                "wm_v": s_wm[0], "wm_t": s_wm[1], "wm_d": s_wm[2], "wm_a": s_wm[3], "wm_f": s_wm[4],
                "orig_total": avg_orig, "wm_total": avg_wm,
                "delta": round(avg_wm - avg_orig, 3)
            }
            pd.DataFrame([res]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error skipping {fname}: {e}")

    # 打印最终统计
    try:
        final_df = pd.read_csv(OUTPUT_CSV)
        # --- [FIX] 增加健壮性判断 ---
        if final_df.empty:
            print("\n" + "="*40)
            print(f"Report for style '{style}' is empty. No statistics to display.")
            print("This might be because all videos were already processed in previous runs.")
            print("="*40)
        elif 'delta' not in final_df.columns:
            print("\n" + "="*40)
            print("ERROR: 'delta' column not found in the report.")
            print("The CSV file might be from an old version of the script.")
            print(f"Consider deleting or renaming {OUTPUT_CSV} and re-running.")
            print("="*40)
        else:
            print("\n" + "="*40)
            print(f"DONE! Processed {len(final_df)} pairs for style '{style}'.")
            print(f"Average Original Score: {final_df['orig_total'].mean():.4f}")
            print(f"Average Watermarked Score: {final_df['wm_total'].mean():.4f}")
            print(f"Mean Score Drift (Delta): {final_df['delta'].mean():.4f}")
            print("="*40)
    except FileNotFoundError:
        print(f"Report file {OUTPUT_CSV} not found. No statistics to display.")
    except pd.errors.EmptyDataError:
        print(f"Report file {OUTPUT_CSV} is empty. No statistics to display.")


if __name__ == "__main__":
    main()