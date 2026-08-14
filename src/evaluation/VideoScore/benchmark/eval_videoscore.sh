#!/bin/bash
# 路径锁定（请确认此路径包含所有 .jpg）
FRAMES_DIR="data/physics-IQ-benchmark/switch-frames/"
RESULT_DIR="./eval_results/physics_iq_official"
mkdir -p $RESULT_DIR

# 四卡并行推理 (GPU 1, 2, 3, 4)
for IDX in {0..3}; do
    GPU_ID=$((IDX + 1))
    echo "正在拉起卡 $GPU_ID 的推理任务..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_videoscore.py         --frames_dir $FRAMES_DIR         --result_file "${RESULT_DIR}/chunk_${IDX}.json"         --num_chunks 4         --chunk_idx $IDX &
done
wait
echo "🎉 Physics-IQ 所有 MLLM 维度指标评测完成！"
