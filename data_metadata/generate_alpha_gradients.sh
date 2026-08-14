#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_DIR="${SOURCE_DIR:-$CODE_ROOT/data/videos/GenVideos/my_videos/Videos}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_ROOT/data/videos/GenVideos/my_videos/alpha_gradients_videos}"
SORA_WM="${SORA_WM:-$CODE_ROOT/src/watermark_tools/sora-watermark-adder/public/watermarks/water_横屏.mp4}"
DYNAMIC_MASK="${DYNAMIC_MASK:-$CODE_ROOT/src/watermark_tools/sora-watermark-adder/public/watermarks/dynamic_mask.mp4}"

MAX_JOBS=16

mkdir -p "$OUTPUT_DIR"

echo "========================================================="
echo "🚀 开启 100% 动态捕捉 + 物理级时空对齐"
echo "========================================================="

for video in "$SOURCE_DIR"/*.mp4; do
    [ ! -f "$video" ] && continue
    name=$(basename "$video" .mp4)
    
    (
        # 捕捉原片物理属性
        V_INFO=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=s=x:p=0 "$video")
        V_W=$(echo $V_INFO | cut -d'x' -f1); V_H=$(echo $V_INFO | cut -d'x' -f2); V_D=$(echo $V_INFO | cut -d'x' -f3)

        # --- A. 生成真实的 Sora 水印版 ---
        SORA_OUT="$OUTPUT_DIR/${name}_sora_watermark.mp4"
        if [ ! -s "$SORA_OUT" ]; then
            ffmpeg -y -loglevel error -i "$video" -stream_loop -1 -i "$SORA_WM" \
                -filter_complex "[1:v]scale=$V_W:$V_H[wm_scaled];[wm_scaled]colorkey=black:0.1:0.1,format=rgba[wm_alpha];[0:v][wm_alpha]overlay=0:0:shortest=1" \
                -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -t "$V_D" "$SORA_OUT"
        fi

        # --- B. 生成 10 个 Alpha 动态捕捉版 ---
        for i in {1..10}; do
            VAL=$(echo "scale=1; $i/10" | bc); ALPHA=$(printf "%.1f" $VAL)
            OUT_ALPHA="$OUTPUT_DIR/${name}_alpha_${ALPHA}.mp4"
            
            if [ ! -f "$OUT_ALPHA" ]; then
                # 逻辑：把动态白块视频拿进来，缩放对齐后，利用 colorchannelmixer 控制全局透明度
                ffmpeg -y -loglevel error -i "$video" -stream_loop -1 -i "$DYNAMIC_MASK" \
                    -filter_complex "[1:v]scale=$V_W:$V_H[mask_scaled];[mask_scaled]colorkey=black:0.1:0.1,format=rgba,colorchannelmixer=aa=$ALPHA[mask_alpha];[0:v][mask_alpha]overlay=0:0:shortest=1" \
                    -c:v libx264 -crf 23 -preset medium -t "$V_D" -pix_fmt yuv420p "$OUT_ALPHA"
            fi
        done
        echo "✅ $name 组已完成 (动态追踪锁死)"
    ) &

    [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]] && wait -n
done
wait
echo "🎉 完美洗白！"
