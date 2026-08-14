import cv2
import numpy as np
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
default_wm = SCRIPT_DIR / "sora-watermark-adder/public/watermarks/water_横屏.mp4"
default_mask = SCRIPT_DIR / "sora-watermark-adder/public/watermarks/dynamic_mask.mp4"

wm_path = os.environ.get("WM_PATH", str(default_wm))
mask_path = os.environ.get("MASK_PATH", str(default_mask))

print("🔍 正在扫描 Sora 水印的运动轨迹...")

cap = cv2.VideoCapture(wm_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mask_path, fourcc, fps, (w, h))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 提取非黑色区域（即 Logo）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建纯黑背景
    mask = np.zeros_like(frame)
    
    if contours:
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + bw), max(y_max, y + bh)
        
        # 在检测到的坐标画上纯白方块（每帧动态更新坐标）
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        
    out.write(mask)
    frame_count += 1

cap.release()
out.release()
print(f"✅ 动态遮罩已生成 ({frame_count} 帧)! 路径: {mask_path}")
