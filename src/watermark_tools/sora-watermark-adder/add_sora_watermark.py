import os
import subprocess
import json
import argparse
from pathlib import Path

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

def add_watermark(input_path, output_path, watermark_dir="watermarks"):
    """
    给视频添加 Sora 水印
    逻辑复刻自 sora-watermark-adder 项目
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # 1. 获取视频信息
    try:
        video_duration, width, height = get_video_info(input_path)
    except Exception as e:
        print(f"Error reading video info: {e}")
        return

    # 2. 选择水印 (横屏/竖屏)
    is_landscape = width >= height
    watermark_filename = "water_横屏.mp4" if is_landscape else "water_竖屏.mp4"
    watermark_path = os.path.join(watermark_dir, watermark_filename)
    
    if not os.path.exists(watermark_path):
        # 尝试备用名称 (原项目中的名称)
        watermark_filename = "water_横屏.mp4" if is_landscape else "water_竖屏.mp4"
        watermark_path = os.path.join(watermark_dir, watermark_filename)
        
    if not os.path.exists(watermark_path):
        print(f"Error: Watermark file not found: {watermark_path}")
        print(f"Please ensure '{watermark_filename}' exists in '{watermark_dir}' folder.")
        return

    # 3. 获取水印时长
    wm_duration, wm_width, wm_height = get_video_info(watermark_path)
    
    # 4. 计算循环次数
    loop_count = int(video_duration / wm_duration) + 1
    
    print(f"Processing: {input_path}")
    print(f"Video: {width}x{height}, {video_duration}s")
    print(f"Watermark: {watermark_path} (Loop {loop_count} times)")

    # 5. 构建 FFmpeg 命令
    # 逻辑：
    # - [1:v]scale=W:H[scaled]: 将水印缩放到视频大小
    # - [scaled]colorkey=0x000000:0.3:0.2[keyed]: 去除黑色背景 (透明度)
    # - [0:v][keyed]overlay=0:0[v]: 叠加水印
    
    filter_complex = (
        f"[1:v]scale={width}:{height}[scaled];"
        f"[scaled]colorkey=0x000000:0.3:0.2[keyed];"
        f"[0:v][keyed]overlay=0:0[v]"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
    ]

    # 如果需要循环水印
    if loop_count > 1:
        cmd.extend(['-stream_loop', str(loop_count - 1)])
    
    cmd.extend([
        '-i', watermark_path,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '0:a?', # 保留原音频（如果有）
        '-c:a', 'copy',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-t', str(video_duration), # 截断到原视频时长
        output_path
    ])

    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Add Sora watermark to video")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--wm_dir", default="watermarks", help="Directory containing watermark videos")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单个文件处理
        output_path = args.output
        if not output_path:
            stem = input_path.stem
            ext = input_path.suffix
            output_path = str(input_path.parent / f"{stem}_sora{ext}")
        
        add_watermark(str(input_path), output_path, args.wm_dir)
        
    elif input_path.is_dir():
        # 批量处理目录
        output_dir = Path(args.output) if args.output else input_path / "sora_watermarked"
        os.makedirs(output_dir, exist_ok=True)
        
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        for file in input_path.iterdir():
            if file.suffix.lower() in video_extensions:
                output_file = output_dir / file.name
                add_watermark(str(file), str(output_file), args.wm_dir)

if __name__ == "__main__":
    main()