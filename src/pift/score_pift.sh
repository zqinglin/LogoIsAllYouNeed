#!/bin/bash
# After Exp1 training: score trained C+/C- models on SAP clean vs real watermarked videos.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mantis_train
# All paths default to relative locations; override any via env var to point at your data.
ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
BASE_MODEL=$(ls -d ~/.cache/huggingface/hub/models--TIGER-Lab--Mantis-8B-Idefics2/snapshots/*/ | head -1)
# SAP evaluation set locations: point these at your local paths.
CLEAN="${SAP_CLEAN_DIR:-$ROOT/../data/sap_clean}"
WM="${SAP_WM_DIR:-$ROOT/../data/sap_watermarked}"
PROMPTS="${SAP_PROMPT_CSV:-$ROOT/../data/video_to_prompt_full.csv}"
# Path to the TIGER-Lab VideoScore repo (used as a Python module by score_exp2.py).
VIDEOSCORE_ROOT="${VIDEOSCORE_ROOT:-$ROOT/../third_party/VideoScore}"
# Path to score_exp2.py (the shared VideoScore-style scoring driver).
SCORE_DRIVER="${SCORE_DRIVER:-$ROOT/../evaluation/score_exp2.py}"
OUT="${OUT:-$ROOT/exp1_scores}"; mkdir -p "$OUT"
export WANDB_DISABLED=true

for COND in contam pift; do
  CKPT=$ROOT/ckpt_pift_${COND}
  MODELDIR=$CKPT/exp1_${COND}/checkpoint-final
  [ -d "$MODELDIR" ] || MODELDIR=$(ls -d $CKPT/exp1_${COND}/checkpoint-* 2>/dev/null | tail -1)
  echo "=== $COND model dir: $MODELDIR ==="
  # ensure processor files present (copy from base if missing)
  for f in preprocessor_config.json processor_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.json added_tokens.json vocab.json merges.txt; do
    [ -f "$MODELDIR/$f" ] || cp "$BASE_MODEL/$f" "$MODELDIR/" 2>/dev/null || true
  done
  for SPLIT in clean wm; do
    DIR=$CLEAN; [ "$SPLIT" = "wm" ] && DIR=$WM
    echo "--- scoring $COND/$SPLIT ---"
    python "$SCORE_DRIVER" --video-dir "$DIR" --prompt-csv "$PROMPTS" \
      --output-csv "$OUT/results_${COND}_${SPLIT}.csv" --model-name "$MODELDIR" \
      --videoscore-project-path "$VIDEOSCORE_ROOT" --gpus 0,1,2,3 --max-frames 8 --num-passes 1 2>&1 | tail -3
  done
done
echo SCORE_TRAINED_DONE
