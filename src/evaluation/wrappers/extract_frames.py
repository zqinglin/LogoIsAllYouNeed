import cv2
import os
from pathlib import Path

# --- [1. 视频路径配置] ---
CODE_ROOT = Path(__file__).resolve().parents[3]
video_watermarked = os.environ.get(
    "VIDEO_WATERMARKED",
    str(CODE_ROOT / "data/videos/GenVideos/my_videos/watermarked_videos/0000_hunyuan_1724.mp4"),
)
video_original = os.environ.get(
    "VIDEO_ORIGINAL",
    str(CODE_ROOT / "data/videos/GenVideos/my_videos/Videos/0000_hunyuan_1724.mp4"),
)

# --- [2. 抽帧核心函数] ---
def extract_first_frame(video_path, output_filename):
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频文件，请检查路径: {video_path}")
        return

    # 使用 OpenCV 读取视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    # 读取第一帧 (ret 是布尔值，代表是否成功读取；frame 是图像矩阵)
    ret, frame = cap.read()
    
    if ret:
        # 保存图片
        cv2.imwrite(output_filename, frame)
        print(f"✅ 成功抽取第一帧！已保存为: {output_filename}")
    else:
        print(f"❌ 无法读取该视频的第一帧数据: {video_path}")
        
    # 释放资源
    cap.release()

# --- [3. 主程序] ---
if __name__ == "__main__":
    print("🎬 正在从视频中提取第一帧...")
    
    # 输出的文件名你可以自己改，默认保存在你运行脚本的当前目录下
    extract_first_frame(video_watermarked, "frame_watermarked_hunyuan.jpg")
    extract_first_frame(video_original, "frame_original_hunyuan.jpg")
    
    print("✨ 抽取完成！赶紧下载下来拖进 PPT 里排版吧！")
