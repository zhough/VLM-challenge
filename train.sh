#!/bin/bash

source activate docparse         # 激活环境

CFG_PATH="path/to/your/qwen2.5vl-3b-lora.yaml"     # 配置文件路径
OUTPUT_PATH="path/to/your/output/dir"              # 模型输出路径

LOG_PATH="./log/train.log"
touch "$LOG_PATH"

MAX_PIXELS=1003520 MIN_PIXELS=200704 \
python -m torch.distributed.launch --nproc_per_node=4 train.py \
  --config "${CFG_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  > "${LOG_PATH}" 2>&1 &

tail -f "${LOG_PATH}"
