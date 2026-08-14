#!/bin/bash
# Exp1 LoRA training of Mantis-Idefics2-8B in regression mode (VideoScore recipe).
# Usage: bash launch_train.sh <cond: cplus|cminus> <gpus e.g. 0,1,2,3>
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mantis_train
export PYTHONNOUSERSITE=1
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29734
export TP_SOCKET_IFNAME=lo
COND=$1
GPUS=${2:-0,1,2,3}   # full FT needs zero3 sharding across all 4 A100s
# ROOT defaults to the directory of this script; override via env var if you keep code and data separate.
ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
MANTIS_TRAIN=$ROOT/Mantis/mantis/train
BASE_MODEL=$ROOT/base_model_idefics2

# data config yaml
CFG=$ROOT/pift_data/dataconfig_${COND}.yaml
cat > "$CFG" <<EOF
data:
  -
    name: "${COND}"
    type: json
    path: "$ROOT/pift_data/train_${COND}.json"
    format: classification
    split: train
EOF
echo "data config: $CFG"

NG=$(echo $GPUS | tr ',' '\n' | grep -c .)
GBS=32
PDBS=1
GA=$(( GBS / (PDBS * NG) ))
OUT=$ROOT/ckpt_pift_${COND}

cd "$MANTIS_TRAIN"
export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$ROOT/Mantis:$PYTHONPATH

accelerate launch --config_file=accelerate_configs/accelerate_config_zero3_offload.yaml \
  --num_processes=$NG --main_process_ip 127.0.0.1 --main_process_port=$((29500 + RANDOM % 1000)) \
  train_idefics2.py \
  --model_name_or_path "$BASE_MODEL" \
  --attn_implementation eager \
  --data_config_file "$CFG" \
  --problem_type regression \
  --num_labels 5 \
  --run_name "exp1_${COND}" \
  --bf16 True \
  --output_dir "$OUT" \
  --num_train_epochs 1 \
  --per_device_train_batch_size $PDBS \
  --gradient_accumulation_steps $GA \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --report_to none \
  --do_train \
  --lora_enabled False \
  --qlora_enabled False \
  --max_seq_len 4096
echo "TRAIN_DONE_${COND}"
