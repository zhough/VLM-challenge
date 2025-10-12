# DocParse-Challenge


## 项目简介

`docparse-challenge` 为金山办公2025算法挑战赛——多模态文档解析大赛官方代码，旨在提供一套基于Qwen2.5‑VL‑3B的LoRA 微调、权重合并与推理评测脚本。

- **训练脚本**：`train.py`（LoRA 微调）
- **合并脚本**：`merge_lora.py`
- **推理脚本**：`eval.py`（批量推理生成 `jsonl` 提交文件）

---

## 环境依赖

```bash
conda create -n docparse python=3.10 -y
conda activate docparse

# transformers安装
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 9985d06add07a4cc691dc54a7e34f54205c04d40
pip install .

# 其他库安装
pip install ms-swift==3.4.0 lmdeploy==0.9.1 qwen-vl-utils[decord] peft accelerate beautifulsoup4 bitsandbytes

# flash-attention安装
pip install -U flash-attn --no-build-isolation # 若安装失败，可以参考https://github.com/Dao-AILab/flash-attention/releases，选择v2.7.4.post1本地安装
```

---

## 目录结构

```text
 docparse-challenge/
 ├─ train.sh                  # 训练入口
 ├─ train.py                  # LoRA 微调主脚本
 ├─ eval.sh                   # 推理入口（vLLM + 批量推理）
 ├─ eval.py                   # 执行模型推理并生成 jsonl
 ├─ merge_lora.py             # LoRA 与基座模型合并脚本
 └─ trainer/
     ├─ config/
     │   └─ qwen2.5vl-3b-lora.yaml   # 默认超参 & 数据路径配置
     └─ dataset/
         └─ preprocess.py            # 数据集构建与预处理
```

---

## 快速开始

### 1. 编辑配置

1. **关键字段修改**：
   | 字段                        | 说明                                  |
   | ------------------------- | ----------------------------------- |
   | `model.model_path`        | 基座模型（如Qwen2.5‑VL‑3B‑Instruct）路径 |
   | `dataset.train_data_path`  | 训练集图像数据路径                       |
   | `dataset.train_json_path` | 训练集 `jsonl`                       |
   | `dataset.valid_data_path`  | 验证集集图像数据路径                       |
   | `dataset.valid_json_path` | 验证集 `jsonl`                       |
   | 其余 **LoRA / hparams**     | 按需调整                     |


### 2. 启动训练

1. **修改路径**

   ```bash
   #!/bin/bash
   source activate docparse         # 激活环境

   CFG_PATH="path/to/your/qwen2.5vl-3b-lora.yaml"     # 配置文件路径
   OUTPUT_PATH="path/to/your/output/dir"              # 模型输出路径

   MAX_PIXELS=1003520 MIN_PIXELS=200704 \
   python -m torch.distributed.launch --nproc_per_node=4 train.py \
     --config "${CFG_PATH}" \
     --output_path "${OUTPUT_PATH}" \
   > ./log/train.log 2>&1 &
   tail -f ./log/train.log
   ```

2. **启动训练**
    ```bash
    bash train.sh
    ```

3. **其他参数**
   - `--nproc_per_node=4`：默认使用4块GPU，可自行调整。

训练完成后，`OUTPUT_PATH` 内将包含 **LoRA adapter** 权重。

### 3. 合并 LoRA 权重（可选）
1. **修改变量**：

   ```python
   BASE_MODEL_PATH  = "path/to/your/qwen2.5-vl-3b-instruct"       # 基座模型路径
   ADAPTER_PATH     = "path/to/your/output/dir/checkpoint-XXXX"   # 模型checkpoint路径
   MERGED_MODEL_PATH = "path/to/your/output/dir/merged_qwen2.5_vl_3b"  # 合并模型输出路径
   ```
2. **执行脚本**
   ```bash
   python merge_lora.py
   ```

### 4. 启动推理服务

1. **修改路径**

   ```bash
   source activate path/to/your/conda/env                           # 激活环境

   IMAGE_DIR="path/to/your/image/dir"                               # 图片目录
   MERGE_MODEL_PATH="path/to/your/output/dir/merged_qwen2.5_vl_3b"  # 合并的模型权重路径
   OUTPUT_BASE_DIR="path/to/your/eval_result"                       # jsonl结果保存目录

   export CUDA_VISIBLE_DEVICES=0
   python eval.py \
   --model_path "${MERGE_MODEL_PATH}" \
   --output_base_dir "${OUTPUT_BASE_DIR}" \
   --image_dir "${IMAGE_DIR}" \
   > "${LOG_FILE}" 2>&1
   ```

2. **启动推理**
    ```bash
    bash eval.sh
    ```

推理完成后，确认OUTPUT_BASE_DIR中的predict.jsonl行数与测试集图像数量一致，之后提交至比赛平台即可查看分数。

