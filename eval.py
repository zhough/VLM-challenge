import os
import re
import json
import time
import yaml
import warnings
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

from lmdeploy import pipeline, GenerationConfig 

warnings.filterwarnings("ignore")

def modify_bboxes(
    text: str,
    image_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    attr_name: str = "data-bbox",
) -> str:
    """
    将 HTML 中的 bbox (x1 y1 x2 y2) 从 resized_size 映射回 image_size
    """
    img_w, img_h = image_size
    res_w, res_h = resized_size

    def _replace(match: re.Match[str]) -> str:
        x1, y1, x2, y2 = map(int, match.group(1).split())
        bbox = np.array([x1, y1, x2, y2], dtype=float)
        scale = np.array([img_w / res_w, img_h / res_h] * 2)
        x1, y1, x2, y2 = (bbox * scale).round().astype(int)
        return f'{attr_name}="{x1} {y1} {x2} {y2}"'

    return re.sub(rf'{attr_name}=\"([^"]+)\"', _replace, text)


def build_messages(pil_img: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    """
    构建LMDeploy VLM的消息格式
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": pil_img}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def worker(
    img_path: Path,
    llm_pipe,
    gen_cfg: GenerationConfig,
    prompt: str = "QwenVL HTML",
) -> Dict[str, Any]:
    """
    对单张图片执行推理并返回结果字典
    """
    img_name = img_path.name
    img_orig = Image.open(img_path)

    try:
        t0 = time.time()
        resp = llm_pipe(build_messages(img_orig, prompt), gen_config=gen_cfg)
        latency = round(time.time() - t0, 3)
        answer = resp.text if hasattr(resp, "text") else str(resp)
        # bbox 映射回原尺寸
        answer = modify_bboxes(answer, img_orig.size, img_orig.size)
    except Exception as exc:
        print(f"[{img_name}] inference error: {exc}")
        return {"image": img_name, "prompt": prompt, "answer": ""}

    return {"image": img_name, "prompt": prompt, "answer": answer, "latency": latency}


def infer(
    image_dir: str,
    output_path: str,
    llm_pipe,
    gen_cfg: GenerationConfig,
    num_threads: int = 8,
) -> None:
    """
    遍历 image_dir 下所有图片，多线程推理并写入 output_path
    """
    img_paths = sorted(
        p
        for p in Path(image_dir).glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    with open(output_path, "w", encoding="utf-8") as fout, ThreadPoolExecutor(
        max_workers=num_threads
    ) as pool:
        futures = [pool.submit(worker, p, llm_pipe, gen_cfg) for p in img_paths]
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"[Saved] {len(img_paths)} records -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LMDeploy VLM batch inference")
    parser.add_argument(
        "--output_base_dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Merged model directory"
    )
    parser.add_argument(
        "--image_dir", required=True, type=str, help="Directory with images"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="For distributed launcher (ignored here)",
    )
    args = parser.parse_args()

    image_dir = args.image_dir

    # 初始化 LMDeploy pipeline
    print(f"[Load model] {args.model_path}")
    llm_pipe = pipeline(args.model_path)
    gen_cfg = GenerationConfig(
        max_new_tokens=4096, temperature=0.1, top_p=0.001, top_k=1
    )

    # 输出文件
    os.makedirs(args.output_base_dir, exist_ok=True)
    output_file = os.path.join(args.output_base_dir, "eval_result.jsonl")
    open(output_file, "w").close()  

    # 推理
    infer(image_dir, output_file, llm_pipe, gen_cfg, num_threads=16)

    llm_pipe.close()


if __name__ == "__main__":
    main()
