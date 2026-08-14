#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ORIG_VIDEO="${CODE_ROOT}/examples/reviewer_demo/videos/original/0000_sora_0.mp4"
WM_VIDEO="${CODE_ROOT}/examples/reviewer_demo/videos/watermarked/0000_sora_0.mp4"
OUT_JSON="${CODE_ROOT}/examples/reviewer_demo/output/demo_result.json"

PROMPT_TEXT="A bustling city marketplace from dawn to dusk, capturing vendors setting up, colorful goods being sold, diverse crowds interacting, lights illuminating as night falls, and the vibrant energy transitioning between day and night."

if [[ ! -f "${ORIG_VIDEO}" ]]; then
  echo "[ERROR] Missing original video: ${ORIG_VIDEO}"
  exit 1
fi
if [[ ! -f "${WM_VIDEO}" ]]; then
  echo "[ERROR] Missing watermarked video: ${WM_VIDEO}"
  exit 1
fi

python "${CODE_ROOT}/examples/reviewer_demo/demo_videoscore_pair.py" \
  --orig-video "${ORIG_VIDEO}" \
  --wm-video "${WM_VIDEO}" \
  --prompt-text "${PROMPT_TEXT}" \
  --output-json "${OUT_JSON}"
