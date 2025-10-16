import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """读取 JSONL 文件，返回 JSON 对象列表，处理读取异常"""
    json_list = []
    if not Path(file_path).exists():
        raise FileNotFoundError(f"输入文件不存在：{file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):  # 记录行号，便于定位错误
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                json_obj = json.loads(line)
                json_list.append((json_obj, line_num))  # 保存行号，用于错误提示
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行 JSON 格式错误，已跳过：{str(e)[:50]}")
    return json_list


def clean_json_fields(json_obj: Dict) -> Dict:
    """清理 JSON 对象，仅保留 "image" 和 "answer" 字段"""
    cleaned = {}
    # 必须保留的字段，缺失时提示但仍保留（避免完全丢失数据）
    for field in ["image", "answer"]:
        if field in json_obj:
            cleaned[field] = json_obj[field]
        else:
            cleaned[field] = ""  # 缺失字段用空字符串填充，避免键不存在
            print(f"⚠️  当前 JSON 缺失 '{field}' 字段，已用空字符串填充：{json_obj.get('image', '未知图片名')}")
    return cleaned


def save_cleaned_jsonl(cleaned_json_list: List[Dict], output_path: str) -> None:
    """将清理后的 JSON 列表写入新的 JSONL 文件"""
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for json_obj in cleaned_json_list:
            # 保证 JSON 格式正确，中文不转义
            json.dump(json_obj, f, ensure_ascii=False)
            f.write("\n")  # 每行一个 JSON 对象


def main():
    # 命令行参数配置（支持用户自定义输入/输出路径）
    parser = argparse.ArgumentParser(description="JSONL 文件字段清理：仅保留 image 和 answer 字段")
    parser.add_argument(
        "--input", 
        default='./eval_result.jsonl', 
        type=str, 
        help="输入 JSONL 文件路径（必填），例如：./eval_result.jsonl"
    )
    parser.add_argument(
        "--output", 
        default="./predict.jsonl", 
        type=str, 
        help="输出清理后的 JSONL 文件路径（默认：./predict.jsonl）"
    )
    args = parser.parse_args()

    try:
        # 1. 读取输入文件
        print(f"📥 正在读取输入文件：{args.input}")
        json_with_line_num = load_jsonl(args.input)
        total_lines = len(json_with_line_num)
        if total_lines == 0:
            print("⚠️  输入文件中无有效 JSON 数据，程序退出")
            return
        print(f"✅ 成功读取 {total_lines} 条有效 JSON 数据")

        # 2. 清理字段（仅保留 image 和 answer）
        print("\n🔧 正在清理多余字段...")
        cleaned_json_list = []
        for json_obj, line_num in json_with_line_num:
            cleaned = clean_json_fields(json_obj)
            cleaned_json_list.append(cleaned)
        print(f"✅ 字段清理完成，共处理 {len(cleaned_json_list)} 条数据")

        # 3. 保存输出文件
        print(f"\n📤 正在保存清理后的文件：{args.output}")
        save_cleaned_jsonl(cleaned_json_list, args.output)
        print(f"🎉 全部处理完成！清理后的文件已保存至：{args.output}")

    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")


if __name__ == "__main__":
    main()