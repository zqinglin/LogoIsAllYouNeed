
import os
import torch
import av
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import pandas as pd
import sys

# --- Path Definitions ---
VIDEOSCORE_PROJECT_PATH = "src/evaluation/VideoScore"
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

try:
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as e:
    print(f"FATAL: Could not import from mantis. Make sure {VIDEOSCORE_PROJECT_PATH} is correct and the project is installed.")
    print(f"Import Error: {e}")
    sys.exit(1)
from tqdm import tqdm
import math

# --- Configuration ---
MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"
VIDEO_PATH = "data/videos/GenVideos/my_videos/watermarked_videos/0000_sora_0.mp4"
PROMPT_CSV_PATH = "data_metadata/video_to_prompt_full.csv"
OUTPUT_DIR = "outputs/attention_analysis"
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
    print("--- Starting Attention Analysis for VideoScore v1.1 ---")
    
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
    eval_prompt += "<image> " * len(video_frames)
    
    # --- 4. Model Inference with Attention Output ---
    print("Running model inference to extract attention...")
    inputs = processor(
        text=eval_prompt, 
        images=video_frames, 
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # --- 5. Process and Visualize Attention ---
    print("Processing attention weights and generating heatmaps...")
    
    # For Idefics2, attentions are stored in `outputs.attentions`.
    # This is a tuple of tuples, one for each layer.
    # Each inner tuple contains (self_attn, cross_attn)
    # We are interested in the attention weights of the final layer.
    # The shape of attentions is (layers, batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions[-1] # Last layer
    
    # The shape is (batch_size, num_heads, sequence_length, num_patches)
    # We average across the heads
    attention_weights = attentions.mean(dim=1).squeeze(0) # Shape: [sequence_length, sequence_length]

    # Find the token that corresponds to the start of the image sequence
    # In Idefics2, images are represented by a special token, often `"<image>"`.
    # The processor converts this to a specific token ID.
    # We'll find the last text token before the image tokens start.
    # The attention from the final text token is a good proxy for overall attention to the images.
    
    # The number of patches is the total sequence length minus the number of text tokens.
    num_text_tokens = inputs["input_ids"].shape[1] - 16 * 16 # A rough estimation, needs to be accurate
    # Let's find the exact number of image tokens from the processor
    # This is a bit tricky as the processor doesn't directly expose this
    # However, we know the image patches are appended at the end.
    # The attention matrix is (seq_len, seq_len). We want the attention from text to images.
    num_patches = 1024 # Idefics2 uses a 32x32 grid of patches
    num_text_tokens = attention_weights.shape[0] - num_patches

    image_attention = attention_weights[num_text_tokens-1, num_text_tokens:]
    
    # The patches are arranged in a grid. We need to find the grid size.
    # For Idefics2, it's typically a square grid.
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        print(f"Warning: Number of patches ({num_patches}) is not a perfect square. Heatmap might be inaccurate.")
        # Attempt to find a close rectangular grid
        grid_h = int(num_patches**0.5)
        while num_patches % grid_h != 0:
            grid_h -=1
        grid_w = num_patches // grid_h
    else:
        grid_h, grid_w = grid_size, grid_size

    heatmap = image_attention.reshape(grid_h, grid_w).float().cpu().numpy()

    cmap = plt.get_cmap('viridis')
    
    for i, frame in enumerate(tqdm(video_frames, desc="Generating and saving frames")):
        # Upscale the heatmap to the frame size
        heatmap_pil = Image.fromarray((cmap(heatmap)[:, :, :3] * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize(frame.size, Image.LANCZOS)

        # Overlay heatmap on the frame
        overlay_frame = Image.blend(frame.convert("RGBA"), heatmap_resized.convert("RGBA"), alpha=0.5)

        # Save the frame
        output_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}_attention.png")
        overlay_frame.convert("RGB").save(output_path)

    print(f"--- Analysis complete! ---")
    print(f"Saved {len(video_frames)} attention maps to: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_attention()
