#!/bin/bash

source activate docparse         # 激活环境

IMAGE_DIR="path/to/your/image/dir"  # 图片目录
MERGE_MODEL_PATH="path/to/your/output/dir/merged_qwen2.5_vl_3b"  # 合并的模型权重路径
OUTPUT_BASE_DIR="path/to/your/eval_result"                       # jsonl结果保存目录

LOG_FILE="./log/eval.log"
touch "$LOG_FILE"

# 运行推理脚本
export CUDA_VISIBLE_DEVICES=0
python eval.py \
  --model_path "${MERGE_MODEL_PATH}" \
  --output_base_dir "${OUTPUT_BASE_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  > "${LOG_FILE}" 2>&1