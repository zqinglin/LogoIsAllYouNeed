import cv2
import numpy as np
import os

def generate_masks(src_path, ref_path, out_dir):
    # 1. 读轨迹
    cap_ref = cv2.VideoCapture(ref_path)
    fps = cap_ref.get(cv2.CAP_PROP_FPS)
    frames_ref = []
    while True:
        ret, frame = cap_ref.read()
        if not ret: break
        frames_ref.append(frame)
    cap_ref.release()

    # 2. 读底片尺寸
    cap_src = cv2.VideoCapture(src_path)
    w, h = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_src.release()

    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, 11):
        alpha = round(i * 0.1, 1)
        alpha_str = "1.0" if alpha == 1.0 else str(alpha)
        out_path = os.path.join(out_dir, f"mask_0002_alpha_{alpha_str}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        last_box = None
        for frame in frames_ref:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            if cnts:
                x, y, ww, hh = cv2.boundingRect(np.concatenate(cnts))
                last_box = (x, y, ww, hh)
            if last_box:
                bx, by, bw, bh = last_box
                cv2.rectangle(mask, (bx, by), (bx+bw, by+bh), (255, 255, 255), -1)
            writer.write(mask)
        writer.release()
        print(f"✅ 已生成: mask_0002_alpha_{alpha_str}.mp4")

if __name__ == '__main__':
    generate_masks('../0002_real_panda_standard.mp4', 'public/watermarks/water_竖屏.mp4', 'public/watermarks/complex_masks')