from peft import PeftModel
import os
import random
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

def merge_lora(base_model_path, ckpt_root, merged_model_path):
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model_with_adapter = PeftModel.from_pretrained(base_model, ckpt_root)
    merged_model = model_with_adapter.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)

    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(merged_model_path)

    print(f"Merged model saved to {merged_model_path}")


if __name__ == "__main__":
    BASE_MODEL_PATH  = "path/to/your/qwen2.5-vl-3b-instruct"       # 基座模型路径
    ADAPTER_PATH     = "path/to/your/output/dir/checkpoint-XXXX"   # 模型checkpoint路径
    MERGED_MODEL_PATH = "path/to/your/output/dir/merged_qwen2.5_vl_3b"  # 合并模型输出路径
    
    merge_lora(BASE_MODEL_PATH, ADAPTER_PATH, MERGED_MODEL_PATH)