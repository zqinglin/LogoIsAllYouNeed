#!/bin/bash
# This script adds a static image watermark to the top-right corner of videos.

set -e # Exit immediately if a command fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# The path to the watermark image file (can be overridden via env).
WATERMARK_IMAGE="${WATERMARK_IMAGE:-$CODE_ROOT/data/assets/ai_watermark.png}"

# --- Script Logic ---

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    echo "Example: $0 ./relative/original/videos ./relative/watermarked/videos"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found."
    exit 1
fi

if [ ! -f "$WATERMARK_IMAGE" ]; then
    echo "Error: Watermark image '$WATERMARK_IMAGE' not found."
    echo "Please save your watermark image to that location first."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "--- Starting Image Watermarking Process ---"
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Watermark Image:  $WATERMARK_IMAGE"

# Loop through all .mp4 files in the input directory
for video_file in "$INPUT_DIR"/*.mp4; do
    if [ -f "$video_file" ]; then
        filename=$(basename -- "$video_file")
        output_file="$OUTPUT_DIR/$filename"

        echo "Processing: $filename"

        # Use ffmpeg to overlay the image.
        # -i "$video_file":       Primary input (the video)
        # -i "$WATERMARK_IMAGE":  Secondary input (the watermark image)
        # -filter_complex:        Define the filter graph.
        #   [0:v][1:v]:             Select video streams from the first and second inputs.
        #   overlay=W-w-10:10:      Apply the overlay filter.
        #     W-w-10: X position. Main video width (W) - watermark width (w) - 10px margin.
        #     10:     Y position. 10px from the top.
        # -map "[v]":              Use the output of the filter_complex as the video stream.
        # -map 0:a?:              Copy the audio stream from the first input, if it exists.
        # -c:a copy:              Copy audio without re-encoding.
        # -y:                     Overwrite output file if it exists.

        ffmpeg -i "$video_file" -i "$WATERMARK_IMAGE" \
            -filter_complex "[0:v][1:v] overlay=W-w-10:10" \
            -map "[v]" -map "0:a?" -c:a copy \
            -c:v libx264 -preset medium -crf 23 \
            -y "$output_file" -loglevel error < /dev/null

        if [ $? -eq 0 ]; then
            echo "  -> Success: $output_file"
        else
            echo "  -> ERROR processing $filename"
        fi
    fi
done

echo "--- Watermarking process complete! ---"
