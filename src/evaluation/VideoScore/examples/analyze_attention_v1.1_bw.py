
import os
import torch
import av
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import pandas as pd
import sys
from tqdm import tqdm
import math

# --- Path Definitions ---
# Ensure the script can find the mantis module
VIDEOSCORE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

try:
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as e:
    print(f"FATAL: Could not import from mantis. Make sure the path is correct and the project is installed.")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"
# Use absolute paths to ensure the script can be run from anywhere
VIDEO_PATH = "data/videos/GenVideos/my_videos/watermarked_videos/0000_alpha_357741999.mp4"
PROMPT_CSV_PATH = "data_metadata/video_to_prompt_full.csv"
OUTPUT_DIR = "outputs/attention_analysis_bw"
MAX_FRAMES = 16 # Limit number of frames for faster analysis

# --- Prompt Definition ---
VS1_1_REGRESSION_QUERY_PROMPT = """
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

def get_prompt_for_video(video_filename, csv_path):
    """Fetches the prompt for a given video filename from the CSV."""
    try:
        prompts_df = pd.read_csv(csv_path)
        prompt_row = prompts_df[prompts_df['filename'] == video_filename]
        if not prompt_row.empty:
            return prompt_row.iloc[0]['prompt']
    except FileNotFoundError:
        print(f"Error: Prompt CSV not found at {csv_path}")
    except Exception as e:
        print(f"Error reading or parsing CSV: {e}")
    return "A high-quality video." # Fallback prompt

def _read_video_pyav(container, indices):
    """Reads frames from a video container at specified indices."""
    frames = []
    container.seek(0)
    start_index, end_index = indices[0], indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_image()) # Return PIL Image
    return frames

def analyze_attention():
    """
    Main function to load the model, process a video, and generate attention heatmaps.
    """
    print("--- Starting Attention Analysis for VideoScore v1.1 (B&W Background) ---")
    
    # --- 1. Setup and Device ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Load Model and Processor ---
    print(f"Loading model: {MODEL_NAME}")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
        model = Idefics2ForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
    except Exception as e:
        print(f"FATAL: Could not load model or processor. Error: {e}")
        return

    # --- 3. Prepare Video and Prompt ---
    video_filename = os.path.basename(VIDEO_PATH)
    prompt_text = get_prompt_for_video(video_filename, PROMPT_CSV_PATH)
    print(f"Analyzing video: {video_filename}")
    print(f"With prompt: '{prompt_text}'")

    try:
        with av.open(VIDEO_PATH) as container:
            total_frames = container.streams.video[0].frames
            indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
            video_frames = _read_video_pyav(container, indices)
    except Exception as e:
        print(f"FATAL: Could not read video file. Error: {e}")
        return

    eval_prompt = VS1_1_REGRESSION_QUERY_PROMPT.format(text_prompt=prompt_text)
    # The space after <image> is CRITICAL for the processor to correctly tokenize the images.
    eval_prompt += "<image> " * len(video_frames)

    # --- Final, Correct Implementation for Multiple Frames ---
    print("--- Visualizing Image Self-Attention Saliency (Corrected for Multiple Frames) ---")

    # We use a generic prompt, but now for all frames.
    prompt_text = "Rate the quality of the video."
    eval_prompt = f"User: {prompt_text}" + "<image>" * len(video_frames) + " Assistant:"

    print(f"Using {len(video_frames)} frames and a generic prompt.")

    inputs = processor(
        text=[eval_prompt], 
        images=video_frames, 
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # --- Visualization ---
    last_layer_attention = outputs.attentions[-1]
    avg_attention = last_layer_attention.mean(dim=1).squeeze(0)
    
    # Based on debug, we know there are 14 text tokens for this generic prompt.
    num_text_tokens = 14
    # The rest are image patches.
    image_self_attention = avg_attention[num_text_tokens:, num_text_tokens:]
    
    # The saliency of a patch is the sum of attention it receives.
    saliency_scores = image_self_attention.sum(dim=0)
    
    num_patches_per_image = 64 # 8x8 grid
    patch_grid_size = 8
    cmap = plt.get_cmap('jet')

    for i, frame in enumerate(tqdm(video_frames, desc="Generating Saliency Maps")):
        start_idx = i * num_patches_per_image
        end_idx = (i + 1) * num_patches_per_image

        if end_idx > saliency_scores.shape[0]:
            break

        frame_saliency = saliency_scores[start_idx:end_idx]
        
        heatmap = frame_saliency.reshape(patch_grid_size, patch_grid_size).float().cpu().numpy()

        # Normalize and create the heatmap image
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_pil = Image.fromarray((cmap(heatmap)[:, :, :3] * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize(frame.size, Image.LANCZOS)

        # Create the final overlay
        frame_bw = frame.convert("L").convert("RGB")
        overlay_frame = Image.blend(frame_bw.convert("RGBA"), heatmap_resized.convert("RGBA"), alpha=0.6)

        output_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}_saliency_map_correct.png")
        overlay_frame.convert("RGB").save(output_path)

    print(f"--- Analysis complete! ---")
    print(f"Saved {len(video_frames)} final saliency maps to: {OUTPUT_DIR}")
    print(f"Saved {len(video_frames)} new attention maps to: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_attention()
