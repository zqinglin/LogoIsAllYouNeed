import av
import numpy as np
from typing import List
from PIL import Image
import torch
import sys
import argparse
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification

def _read_video_pyav(frame_paths:List[str], max_frames:int, container, indices):
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

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="TIGER-Lab/VideoScore-v1.1")
    args = parser.parse_args()

    MAX_NUM_FRAMES=48
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model, torch_dtype=torch.bfloat16, use_fast=False)
    model = Idefics2ForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval()
    model.to(device)

    container = av.open(args.video_path)
    total_frames = container.streams.video[0].frames
    if total_frames > MAX_NUM_FRAMES:
        indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
    else:
        indices = np.arange(total_frames)

    frames = [Image.fromarray(x) for x in _read_video_pyav([], MAX_NUM_FRAMES, container, indices)]
    eval_prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=args.prompt)
    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)

    flatten_images = frames
    inputs = processor(text=eval_prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    aspect_scores = [round(logits[0, i].item(), 3) for i in range(logits.shape[-1])]
    
    # print in a clear json format or comma separated format
    print(f"RESULTS:{aspect_scores[0]},{aspect_scores[1]},{aspect_scores[2]},{aspect_scores[3]},{aspect_scores[4]}")

if __name__ == "__main__":
    main()