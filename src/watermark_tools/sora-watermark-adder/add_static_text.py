
import os
import subprocess
import argparse
from pathlib import Path

def add_static_text_watermark(input_path, output_path, text_to_draw):
    """
    Adds a static text watermark to a video using ffmpeg's drawtext filter.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output watermarked video.
        text_to_draw (str): The text to be drawn as a watermark.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define drawtext filter parameters for consistency
    # We place it at the bottom-right corner, similar to other watermarks.
    # A semi-transparent black box is added for better visibility.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        print(f"Warning: Font file not found at {font_path}. FFmpeg might use a default font.")
        # On some systems, we might need to escape the colon in the filter
        font_path_escaped = font_path.replace(':', '\\:')
    else:
        font_path_escaped = font_path.replace(':', '\\:')

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', input_path,
        '-vf', (
            f"drawtext=fontfile='{font_path_escaped}':text='{text_to_draw}':"
            "fontsize=24:fontcolor=white:x=w-text_w-10:y=h-text_h-10:"
            "box=1:boxcolor=black@0.5:boxborderw=5"
        ),
        '-c:a', 'copy', # Copy audio stream without re-encoding
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        output_path
    ]

    print("Running ffmpeg command:", ' '.join(ffmpeg_cmd))
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed for {input_path}!")
        print(f"Stderr: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Add a static text watermark to a video or a directory of videos.")
    parser.add_argument("input", help="Input video file or directory.")
    parser.add_argument("-o", "--output", help="Output file or directory.")
    parser.add_argument("--text", required=True, help="The text to draw as a watermark.")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = args.output
        if not output_path:
            # Default output name if not provided
            stem = input_path.stem
            ext = input_path.suffix
            output_path = str(input_path.parent / f"{stem}_text_{args.text}{ext}")
        add_static_text_watermark(str(input_path), output_path, args.text)

    elif input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_text_{args.text}"
        os.makedirs(output_dir, exist_ok=True)

        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        for file in input_path.iterdir():
            if file.suffix.lower() in video_extensions:
                output_file = output_dir / file.name
                add_static_text_watermark(str(file), str(output_file), args.text)

if __name__ == "__main__":
    main()
