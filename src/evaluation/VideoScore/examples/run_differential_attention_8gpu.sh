#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU batch launcher for analyze_attention_differential.py
# Usage:
#   bash run_differential_attention_8gpu.sh
#   bash run_differential_attention_8gpu.sh 0,2,5,7
# Optional env overrides:
#   VIDEO_DIR=... OUTPUT_DIR=... MAX_VIDEOS_PER_WORKER=50 NUM_FRAMES=2 MASK_SIZE=16 STRIDE=8 MAX_IMAGE_EDGE=320 GPU_IDS=0,2,5,7
#   WATERMARK_BBOX=0.72,0.82,0.98,0.98 SUBJECT_TOPK_FRAC=0.01 SAVE_RAW_MAP=1
#   WATERMARK_MOTION_VIDEO=./relative/path/watermark.mp4 MOTION_THRESHOLD=15
#   OVERWRITE=1  # force recompute even if output PNG already exists

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/analyze_attention_differential.py"

VIDEO_DIR="${VIDEO_DIR:-data/videos/GenVideos/my_videos/watermarked_videos}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/differential_analysis}"

# GPU selection priority:
# 1) First positional argument, e.g. "0,2,5,7"
# 2) GPU_IDS env var
# 3) Default "0,1,2,3,4,5,6,7"
GPU_IDS_RAW="${1:-${GPU_IDS:-0,1,2,3,4,5,6,7}}"

# Normalize separators to comma, then split.
GPU_IDS_RAW="${GPU_IDS_RAW// /,}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_RAW}"

# Filter empty items (e.g. trailing comma), preserve order.
GPU_IDS=()
for g in "${GPU_LIST[@]}"; do
  if [[ -n "${g}" ]]; then
    GPU_IDS+=("${g}")
  fi
done

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: No valid GPU ids provided. Use e.g. GPU_IDS=0,2,5,7"
  exit 1
fi

NUM_WORKERS="${#GPU_IDS[@]}"
MAX_VIDEOS_PER_WORKER="${MAX_VIDEOS_PER_WORKER:-0}"
NUM_FRAMES="${NUM_FRAMES:-2}"
MASK_SIZE="${MASK_SIZE:-16}"
STRIDE="${STRIDE:-8}"
MAX_IMAGE_EDGE="${MAX_IMAGE_EDGE:-320}"
WATERMARK_BBOX="${WATERMARK_BBOX:-0.72,0.82,0.98,0.98}"
SUBJECT_TOPK_FRAC="${SUBJECT_TOPK_FRAC:-0.01}"
SAVE_RAW_MAP="${SAVE_RAW_MAP:-0}"
WATERMARK_MOTION_VIDEO="${WATERMARK_MOTION_VIDEO:-}"
MOTION_THRESHOLD="${MOTION_THRESHOLD:-15}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "Launching ${NUM_WORKERS} workers..."
echo "GPU_IDS=${GPU_IDS[*]}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WATERMARK_BBOX=${WATERMARK_BBOX}"
echo "SUBJECT_TOPK_FRAC=${SUBJECT_TOPK_FRAC}"
echo "SAVE_RAW_MAP=${SAVE_RAW_MAP}"
echo "WATERMARK_MOTION_VIDEO=${WATERMARK_MOTION_VIDEO}"
echo "MOTION_THRESHOLD=${MOTION_THRESHOLD}"
echo "OVERWRITE=${OVERWRITE}"

pids=()
for rank in $(seq 0 $((NUM_WORKERS - 1))); do
  log_file="${OUTPUT_DIR}/worker_${rank}.log"
  gpu_id="${GPU_IDS[${rank}]}"

  extra_args=()
  if [[ "${SAVE_RAW_MAP}" == "1" ]]; then
    extra_args+=("--save_raw_map")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    extra_args+=("--overwrite")
  fi
  if [[ -n "${WATERMARK_MOTION_VIDEO}" ]]; then
    extra_args+=("--watermark_motion_video" "${WATERMARK_MOTION_VIDEO}")
    extra_args+=("--motion_threshold" "${MOTION_THRESHOLD}")
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python "${PY_SCRIPT}" \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers "${NUM_WORKERS}" \
    --worker_rank "${rank}" \
    --max_videos "${MAX_VIDEOS_PER_WORKER}" \
    --num_frames "${NUM_FRAMES}" \
    --mask_size "${MASK_SIZE}" \
    --stride "${STRIDE}" \
    --max_image_edge "${MAX_IMAGE_EDGE}" \
    --watermark_bbox "${WATERMARK_BBOX}" \
    --subject_topk_frac "${SUBJECT_TOPK_FRAC}" \
    --device cuda:0 \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &

  pids+=("$!")
  echo "Started worker ${rank} on GPU ${gpu_id}, PID=${pids[-1]}, log=${log_file}"
done

echo "Waiting for workers to finish..."
for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All workers finished."

# Merge worker summaries into one CSV.
python - <<'PY'
import glob
import os
import pandas as pd

out_dir = os.environ.get("OUTPUT_DIR", "") or "outputs/differential_analysis"
paths = sorted(glob.glob(os.path.join(out_dir, "differential_analysis_summary_worker_*.csv")))
if not paths:
    print("No worker summary CSV found; nothing to merge.")
    raise SystemExit(0)

df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
out_path = os.path.join(out_dir, "differential_analysis_summary_all_workers.csv")
df.to_csv(out_path, index=False)
print(f"Merged summary saved to: {out_path}")
print(f"Rows: {len(df)}")
PY

echo "Done."
