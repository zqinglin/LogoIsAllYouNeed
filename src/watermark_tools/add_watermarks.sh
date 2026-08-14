#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 可通过环境变量覆盖
INPUT_DIR="${INPUT_DIR:-$CODE_ROOT/data/videos/GenVideos/my_videos/Videos}"
WATERMARK_DIR="${WATERMARK_DIR:-$SCRIPT_DIR/sora-watermark-adder/public/watermarks}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_ROOT/data/videos/GenVideos/my_videos/watermarked_videos}"

# 横屏和竖屏水印文件的路径
LANDSCAPE_WATERMARK="$WATERMARK_DIR/water_横屏.mp4"
PORTRAIT_WATERMARK="$WATERMARK_DIR/water_竖屏.mp4"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查水印文件是否存在
if [ ! -f "$LANDSCAPE_WATERMARK" ] || [ ! -f "$PORTRAIT_WATERMARK" ]; then
    echo "错误：找不到水印文件。请确保以下文件存在:"
    echo "  - $LANDSCAPE_WATERMARK"
    echo "  - $PORTRAIT_WATERMARK"
    exit 1
fi

# 遍历输入目录中的所有mp4文件
for video_file in "$INPUT_DIR"/*.mp4; do
    if [ -f "$video_file" ]; then
        filename=$(basename -- "$video_file")
        output_file="$OUTPUT_DIR/$filename"

        echo "正在处理: $filename"

        # 获取视频的宽度、高度和时长
        video_info=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of csv=s=x:p=0 "$video_file")
        width=$(echo "$video_info" | cut -d'x' -f1)
        height=$(echo "$video_info" | cut -d'x' -f2)
        duration=$(echo "$video_info" | cut -d'x' -f3)

        # 判断使用横屏还是竖屏水印
        if [ "$width" -ge "$height" ]; then
            watermark_file="$LANDSCAPE_WATERMARK"
        else
            watermark_file="$PORTRAIT_WATERMARK"
        fi
        
        # 获取水印视频的时长
        watermark_duration=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=s=x:p=0 "$watermark_file")

        # 计算水印需要循环的次数
        loop_count=$(awk "BEGIN {print int((${duration:-0} / ${watermark_duration:-1}) + 0.999)}")
        stream_loop_arg=""
        if [ "$loop_count" -gt 1 ]; then
            stream_loop_arg="-stream_loop $(($loop_count - 1))"
        fi

        # 构建ffmpeg命令
        filter_complex="[1:v]scale=${width}:${height}[scaled];[scaled]colorkey=0x000000:0.3:0.2[keyed];[0:v][keyed]overlay=0:0[v]"
        if [ "$loop_count" -eq 1 ]; then
             filter_complex="[1:v]scale=${width}:${height}[scaled];[scaled]colorkey=0x000000:0.3:0.2[keyed];[keyed]trim=end=${duration},setpts=PTS-STARTPTS[wm];[0:v][wm]overlay=0:0[v]"
        fi

        # 执行ffmpeg命令
        ffmpeg -i "$video_file" $stream_loop_arg -i "$watermark_file" \
            -filter_complex "$filter_complex" \
            -map "[v]" -map "0:a?" -c:a copy \
            -c:v libx264 -preset medium -crf 23 \
            -t "$duration" -y "$output_file"

        if [ $? -eq 0 ]; then
            echo "成功生成: $output_file"
        else
            echo "处理失败: $filename"
        fi
        echo "-------------------------"
    fi
done

echo "所有视频处理完成！"
