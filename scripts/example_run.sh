#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Update this if your downloaded videos are elsewhere.
export DATA_ROOT="${DATA_ROOT:-$CODE_ROOT/data/videos/GenVideos/my_videos}"
export OUTPUT_DIR="${OUTPUT_DIR:-$CODE_ROOT/outputs/evaluation_results}"

python "$CODE_ROOT/scripts/repro_check.py"

# 1) prepare watermarked videos
bash "$CODE_ROOT/src/watermark_tools/add_watermarks.sh"

# 2) evaluate with VideoScore v1.1 (parallel)
python "$CODE_ROOT/src/evaluation/wrappers/run_videoscore_v1.1_final.py"

echo "Done."
