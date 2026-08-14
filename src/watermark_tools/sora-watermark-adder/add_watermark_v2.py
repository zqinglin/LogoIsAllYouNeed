
import os
import subprocess
import json
import argparse
import tempfile
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def get_video_info(file_path):
    """获取视频时长和尺寸"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    duration = float(data['format']['duration'])

    video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    if not video_stream:
        raise ValueError("No video stream found")

    width = int(video_stream['width'])
    height = int(video_stream['height'])

    return duration, width, height


class Watermark:
    def __init__(self, width, height, duration):
        self.width = width
        self.height = height
        self.duration = duration
        self.cleanup_files = []

    def get_ffmpeg_args(self):
        raise NotImplementedError

    def __del__(self):
        for file_path in self.cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up temp file {file_path}: {e}")

def create_kling_watermark_image(height=24, text="Kling AI"):
    """Creates the Kling AI watermark image with text."""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", height)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = font.getbbox(text)
    image_width = right - left
    image_height = bottom - top

    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.text((-left, -top), text, font=font, fill=(255, 255, 255, 255))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as fp:
        image.save(fp, format="PNG")
        return fp.name

class KlingWatermark(Watermark):
    def __init__(self, width, height, duration):
        super().__init__(width, height, duration)
        self.kling_image_path = create_kling_watermark_image()
        self.cleanup_files.append(self.kling_image_path)

    def get_ffmpeg_args(self):
        return [
            '-i', self.kling_image_path,
            '-filter_complex', f"[0:v][1:v]overlay=x=W-w-10:y=H-h-10[v]",
            '-map', '[v]',
            '-map', '0:a?',
        ]

def create_four_pointed_star_image(size=48, color=(200, 200, 200, 255), num_points=100):
    """Creates a four-pointed star (astroid) image and returns the path to a temporary file."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    center = size / 2
    radius = size / 2

    points = []
    for i in range(num_points + 1):
        t = 2 * math.pi * i / num_points
        x = center + radius * (math.cos(t) ** 3)
        y = center + radius * (math.sin(t) ** 3)
        points.append((x, y))

    draw.polygon(points, fill=color)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as fp:
        image.save(fp, format="PNG")
        return fp.name

class GeminiWatermark(Watermark):
    def __init__(self, width, height, duration):
        super().__init__(width, height, duration)
        self.star_image_path = create_four_pointed_star_image()
        self.cleanup_files.append(self.star_image_path)

    def get_ffmpeg_args(self):
        return [
            '-i', self.star_image_path,
            '-filter_complex', f"[0:v][1:v]overlay=x=W-w-10:y=H-h-10[v]",
            '-map', '[v]',
            '-map', '0:a?',
        ]

class NoiseWatermark(Watermark):
    def get_ffmpeg_args(self):
        return [
            '-filter_complex', "nullsrc=s=100x50[base];[base]noise=alls=100:allf=t+u[n];[0:v][n]overlay=x=W-w-10:y=H-h-10[v]",
            '-map', '[v]',
            '-map', '0:a?'
        ]

def create_gray_block_image(width=100, height=50, color='gray'):
    """Creates a solid gray block image."""
    image = Image.new("RGB", (width, height), color)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as fp:
        image.save(fp, format="PNG")
        return fp.name

class GrayBlockWatermark(Watermark):
    def __init__(self, width, height, duration):
        super().__init__(width, height, duration)
        self.gray_block_path = create_gray_block_image()
        self.cleanup_files.append(self.gray_block_path)

    def get_ffmpeg_args(self):
        return [
            '-i', self.gray_block_path,
            '-filter_complex', f"[0:v][1:v]overlay=x=W-w-10:y=H-h-10[v]",
            '-map', '[v]',
            '-map', '0:a?',
        ]

class SoraWatermark(Watermark):
    def __init__(self, width, height, duration, wm_dir="public/watermarks"):
        super().__init__(width, height, duration)
        self.wm_dir = wm_dir

    def get_watermark_filename(self, is_landscape):
        """Determines which watermark video file to use based on orientation."""
        return "water_横屏.mp4" if is_landscape else "water_竖屏.mp4"

    def get_ffmpeg_args(self):
        is_landscape = self.width >= self.height
        watermark_filename = self.get_watermark_filename(is_landscape)
        watermark_path = os.path.join(self.wm_dir, watermark_filename)

        if not os.path.exists(watermark_path):
            raise FileNotFoundError(f"Watermark file not found: {watermark_path}")

        wm_duration, _, _ = get_video_info(watermark_path)
        loop_count = int(self.duration / wm_duration) + 1

        filter_complex = (
            f"[1:v]scale={self.width}:{self.height}[scaled];"
            f"[scaled]colorkey=0x000000:0.3:0.2[keyed];"
            f"[0:v][keyed]overlay=0:0[v]"
        )

        args = []
        if loop_count > 1:
            args.extend(['-stream_loop', str(loop_count - 1)])

        args.extend([
            '-i', watermark_path,
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', '0:a?',
            '-t', str(self.duration),
        ])
        return args

class SoraFlippedWatermark(SoraWatermark):
    """
    A variation of the Sora watermark that uses horizontally flipped video files.
    This is intended for control experiments to test semantic priors in VLM evaluators.
    """
    def get_watermark_filename(self, is_landscape):
        """Overrides the base method to point to the flipped watermark files."""
        return "water_横屏_flipped.mp4" if is_landscape else "water_竖屏_flipped.mp4"

def create_control_graphic_image(output_path, width=120, height=36, radius=8, text="AI Gen"):
    """Creates the 'AI Gen' graphic with a solid box and hollow text."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), fill="white", radius=radius)

    try:
        font_size = int(height * 0.6)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text_mask = Image.new("L", (width, height), 0)
    text_draw = ImageDraw.Draw(text_mask)

    left, top, right, bottom = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top
    text_x = (width - text_width - left) / 2
    text_y = (height - text_height - top) / 2
    
    text_draw.text((text_x, text_y), text, font=font, fill=255)

    alpha = img.getchannel('A')
    alpha.paste(0, mask=text_mask)
    img.putalpha(alpha)
    
    img.save(output_path)

def generate_animated_watermark_video(graphic_path, output_path, duration=10, size="1920x1080"):
    """Generates an animated watermark video from a static graphic."""
    width, height = map(int, size.split('x'))
    
    filter_complex = (
        f"[0:v]format=rgba[fg];"
        f"color=s={size}:c=black[bg];"
        f"[bg][fg]overlay=x='mod(t*(W+iw)/{duration},W+iw)-iw':y='mod(t*(H+ih)/{duration},H+ih)-ih'[v]"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-loop', '1',
        '-i', graphic_path,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        output_path
    ]
    
    print("Generating animated watermark video:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

class ControlWatermark(SoraWatermark):
    def __init__(self, width, height, duration, wm_dir="public/watermarks"):
        super(SoraWatermark, self).__init__(width, height, duration)
        self.wm_dir = wm_dir
        
        is_landscape = width >= height
        self.watermark_filename = "control_横屏.mp4" if is_landscape else "control_竖屏.mp4"
        self.watermark_path = os.path.join(self.wm_dir, self.watermark_filename)
        
        if not os.path.exists(self.watermark_path):
            print(f"Control watermark video not found. Generating new one at {self.watermark_path}...")
            os.makedirs(self.wm_dir, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as fp:
                graphic_path = fp.name
            
            try:
                create_control_graphic_image(graphic_path)
                video_size = f"{width}x{height}" if is_landscape else f"{height}x{width}"
                generate_animated_watermark_video(graphic_path, self.watermark_path, size=video_size)
                print("Control watermark video generated successfully.")
            finally:
                if os.path.exists(graphic_path):
                    os.remove(graphic_path)

def add_watermark(input_path, output_path, style='kling', wm_dir="public/watermarks"):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    try:
        video_duration, width, height = get_video_info(input_path)
    except Exception as e:
        print(f"Error reading video info: {e}")
        return

    watermark = None
    try:
        if style == 'kling':
            watermark = KlingWatermark(width, height, video_duration)
        elif style == 'gemini':
            watermark = GeminiWatermark(width, height, video_duration)
        elif style == 'sora':
            watermark = SoraWatermark(width, height, video_duration, wm_dir)
        elif style == 'sora_flipped':
            watermark = SoraFlippedWatermark(width, height, video_duration, wm_dir)
        elif style == 'control':
            watermark = ControlWatermark(width, height, video_duration, wm_dir)
        elif style == 'noise':
            watermark = NoiseWatermark(width, height, video_duration)
        elif style == 'gray':
            watermark = GrayBlockWatermark(width, height, video_duration)
        else:
            print(f"Error: Unknown watermark style '{style}'")
            return

        ffmpeg_args = watermark.get_ffmpeg_args()

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
        ]
        cmd.extend(ffmpeg_args)
        cmd.extend([
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ])

        print("Running ffmpeg command:", ' '.join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Successfully saved to: {output_path}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg failed: {e}")
    finally:
        del watermark

def main():
    parser = argparse.ArgumentParser(description="Add various watermarks to video")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--style", default="kling", choices=['kling', 'gemini', 'sora', 'sora_flipped', 'control', 'noise', 'gray'], help="Watermark style")
    parser.add_argument("--wm_dir", default="public/watermarks", help="Directory containing watermark videos for sora and control styles")

    args = parser.parse_args()

    input_path = Path(args.input)
    script_dir = Path(__file__).parent

    if input_path.is_file():
        output_path = args.output
        if not output_path:
            stem = input_path.stem
            ext = input_path.suffix
            output_path = str(input_path.parent / f"{stem}_{args.style}{ext}")

        add_watermark(str(input_path), output_path, args.style, str(script_dir / args.wm_dir))

    elif input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path / f"{args.style}_watermarked"
        os.makedirs(output_dir, exist_ok=True)

        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        for file in input_path.iterdir():
            if file.suffix.lower() in video_extensions:
                output_file = output_dir / file.name
                add_watermark(str(file), str(output_file), args.style, str(script_dir / args.wm_dir))


if __name__ == "__main__":
    main()
