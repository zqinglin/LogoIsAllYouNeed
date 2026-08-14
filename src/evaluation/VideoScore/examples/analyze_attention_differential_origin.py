
import os
import torch
import av
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import sys
from tqdm import tqdm

# --- Path Setup ---
VIDEOSCORE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if VIDEOSCORE_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VIDEOSCORE_PROJECT_PATH)

try:
    from mantis.models.idefics2 import Idefics2ForSequenceClassification
except ImportError as e:
    print(f"FATAL: Could not import from mantis. Path: {VIDEOSCORE_PROJECT_PATH}")
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = "TIGER-Lab/VideoScore-v1.1"
VIDEO_PATH = "data/videos/GenVideos/my_videos/Videos/0000_alpha_357741999.mp4"
OUTPUT_DIR = "outputs/differential_analysis_original"
NUM_FRAMES_TO_ANALYZE = 2  # Number of frames to process from the video

# Perturbation settings - Finer grain
MASK_SIZE = 16  # Smaller mask for more detail
STRIDE = 8      # Smaller step for denser scanning

# --- Helper Functions ---
def _read_video_frames(video_path, max_frames):
    """Reads up to max_frames from a video and returns them as a list of PIL Images."""
    frames = []
    try:
        with av.open(video_path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i >= max_frames:
                    break
                frames.append(frame.to_image())
        return frames
    except Exception as e:
        print(f"FATAL: Could not read video file {video_path}. Error: {e}")
        return None

def get_score(model, processor, image, device):
    """Gets the visual quality score for a single image."""
    prompt = "User: Rate the visual quality of the image.<image> Assistant:"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    return score

# --- Main Analysis Function ---
def differential_analysis():
    """Runs the differential analysis for multiple frames."""
    print("--- Starting Refined Differential Attention Analysis ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model and Processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    
    print(f"Loading {NUM_FRAMES_TO_ANALYZE} frames from: {VIDEO_PATH}")
    video_frames = _read_video_frames(VIDEO_PATH, NUM_FRAMES_TO_ANALYZE)
    if not video_frames: return

    # 2. Process each frame individually
    for i, original_image in enumerate(video_frames):
        print(f"\n--- Processing Frame {i+1}/{len(video_frames)} ---")
        width, height = original_image.size
        influence_heatmap = np.zeros((height, width))

        # Get Benchmark Score for the current frame
        print("Calculating benchmark score for frame...")
        benchmark_score = get_score(model, processor, original_image, device)
        print(f"Frame {i} Benchmark Score: {benchmark_score:.4f}")

        # Iterate, Mask, and Re-evaluate
        print(f"Running perturbation analysis...")
        y_steps = range(0, height - MASK_SIZE + 1, STRIDE)
        x_steps = range(0, width - MASK_SIZE + 1, STRIDE)

        for y in tqdm(y_steps, desc=f"Scanning Frame {i}"):
            for x in x_steps:
                masked_image = original_image.copy()
                draw = ImageDraw.Draw(masked_image)
                draw.rectangle([x, y, x + MASK_SIZE, y + MASK_SIZE], fill='grey')
                perturbed_score = get_score(model, processor, masked_image, device)
                score_diff = abs(benchmark_score - perturbed_score)
                influence_heatmap[y:y+MASK_SIZE, x:x+MASK_SIZE] += score_diff

        # Visualize the Influence Heatmap
        print("Generating influence map for frame...")
        if np.max(influence_heatmap) > 0:
            influence_heatmap = (influence_heatmap - np.min(influence_heatmap)) / (np.max(influence_heatmap) - np.min(influence_heatmap))
        
        cmap = plt.get_cmap('jet')
        heatmap_pil = Image.fromarray((cmap(influence_heatmap)[:, :, :3] * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize(original_image.size, Image.LANCZOS)

        overlay_frame = Image.blend(original_image.convert("RGBA"), heatmap_resized.convert("RGBA"), alpha=0.6)

        output_path = os.path.join(OUTPUT_DIR, f"differential_map_frame_{i:04d}.png")
        overlay_frame.convert("RGB").save(output_path)
        print(f"Saved influence map for frame {i} to: {output_path}")

    print(f"\n--- Analysis Complete! ---")

if __name__ == "__main__":
    differential_analysis()

if __name__ == "__main__":
    differential_analysis()
