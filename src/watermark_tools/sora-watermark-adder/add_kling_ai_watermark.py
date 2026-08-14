
import os
import subprocess
import argparse
from pathlib import Path

def add_kling_ai_watermark(input_path, output_path):
    """Adds a bold 'KLING AI' text watermark to the bottom right of a video."""
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # Use a larger font size and a semi-transparent border to create a 'bolder' effect
    font_size = 30
    border_width = 1.5
    text = "'KLING AI'"

    filter_vf = (
        f"drawtext="
        f"text={text}:"
        f"fontcolor=white:"
        f"fontsize={font_size}:"
        f"borderw={border_width}:"
        f"bordercolor=black@0.4:"
        f"x=w-tw-10:"
        f"y=h-th-10"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', filter_vf,
        '-c:a', 'copy',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        str(output_path)
    ]

    print("Running ffmpeg command:", ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed. Stderr:\n{e.stderr}")
    except FileNotFoundError:
        print("FFmpeg failed. Is ffmpeg installed and in your PATH?")

def main():
    parser = argparse.ArgumentParser(description="Add a 'KLING AI' watermark to a video.")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        output_path = args.output
        if not output_path:
            stem = input_path.stem
            ext = input_path.suffix
            output_path = input_path.parent / f"{stem}_kling_ai{ext}"
        
        add_kling_ai_watermark(input_path, output_path)
        
    elif input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path / "kling_ai_watermarked"
        os.makedirs(output_dir, exist_ok=True)
        
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        for file in input_path.iterdir():
            if file.suffix.lower() in video_extensions:
                output_file = output_dir / file.name
                add_kling_ai_watermark(file, output_file)

if __name__ == "__main__":
    main()
