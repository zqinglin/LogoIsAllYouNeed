import os
import subprocess
from PIL import Image, ImageDraw
import numpy as np

# 注意：我将默认 duration 改为了 10 秒，因为你前一张截图里 Sora 的视频是 10 秒的
def create_control_watermark_video(width=400, height=100, duration_s=10, fps=24):
    """Creates a STRICTLY TEXTLESS control watermark video."""
    
    temp_frame_dir = "temp_control_frames"
    os.makedirs(temp_frame_dir, exist_ok=True)

    num_frames = duration_s * fps
    
    # --- Generate all frames ---
    for i in range(num_frames):
        # 黑色背景，后续在主脚本里会被 screen 或 colorkey 滤镜过滤掉
        frame = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(frame)

        # 【关键修改 1】：如果你原本的 Sora 水印是静止的，这里也不应该有动态 pan。
        # 如果需要严格控制变量，对照组的位置和运动状态必须和 Sora 水印完全一致！
        # 这里我暂时去掉了 np.sin 的位移，让它保持静止。
        
        # 【关键修改 2】：设置白块的大小。你需要尽量让这个白块的面积，接近原本 "Sora Logo + 字母" 的总面积
        rect_width, rect_height = 180, 50 
        rect_x = (width - rect_width) / 2
        rect_y = (height - rect_height) / 2
        rect_shape = [(rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height)]
        radius = 15

        # 绘制一个实心的白色圆角矩形（模拟原水印的高亮像素，但没有任何文字信息）
        draw.rounded_rectangle(rect_shape, radius=radius, fill="white")

        frame.save(os.path.join(temp_frame_dir, f"frame_{i:04d}.png"))

    print(f"Generated {num_frames} frames.")

    # --- Compile frames into video using ffmpeg ---
    # 输出为 control_water_blank.mp4 以示区分
    output_path = "public/watermarks/control_water_blank.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_frame_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    
    print("Compiling video with ffmpeg...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created purely blank watermark video at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")

    # --- Cleanup ---
    finally:
        for file_name in os.listdir(temp_frame_dir):
            os.remove(os.path.join(temp_frame_dir, file_name))
        os.rmdir(temp_frame_dir)
        print("Cleaned up temporary frames.")

if __name__ == "__main__":
    create_control_watermark_video()