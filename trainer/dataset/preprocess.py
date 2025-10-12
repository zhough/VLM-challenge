import io
import os
import re
import json
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from qwen_vl_utils import smart_resize
import bs4
from bs4 import BeautifulSoup
from collections import defaultdict
import yaml
import torch
import torch.distributed as dist

class JSONLDataset(Dataset):
    def __init__(self, 
                data_path: str,
                jsonl_file_path: str, 
                config_path: str, 
                return_image_path = False, 
                format = "normal", 
                attr_name='data-bbox',
    ):  
        self.data_path = data_path
        self.jsonl_file_path = jsonl_file_path
        self.entries = self._load_entries()
        self.return_image_path = return_image_path
        self.format = format
        self.SYSTEM_MESSAGE = """You are a helpful assistant"""
        self.attr_name = attr_name

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _load_entries(self):
        """从jsonl文件中加载数据"""
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_name = entry['image'] 
        image_path = os.path.join(self.data_path, image_name)
        image = Image.open(image_path)  
        
        # 使用QwenVL的smart_resize函数进行图像缩放
        orig_width, orig_height = image.size
        resized_height, resized_width = smart_resize(
            orig_height,
            orig_width,
            factor=28,
            min_pixels=self.config["generate"]["min_pixels"], 
            max_pixels=self.config["generate"]["max_pixels"]
        )
        image = image.resize((resized_width, resized_height))
        text = entry['suffix']
        prompt = entry['prefix']
        
        try:
            # 按照图像相同缩放比例缩放data-bbox
            text = self.modify_bboxes(text, (orig_width, orig_height), (resized_width, resized_height), attr_name=self.attr_name)
        except Exception as e:
            print(f"Failed to modify bboxes: {e}")

        # 组织输出
        data = self.swift_format_data(image, entry, text, prompt) if self.format == "swift" else self.format_data(image, entry, text, prompt)
        return (data, image_path) if self.return_image_path else data
    
    def modify_bboxes(self, input_str, image_size, resized_image_size, attr_name='data-bbox'):
        """将 HTML 中的 bbox 位置按缩放比例调整"""
        img_w, img_h = image_size
        resized_w, resized_h = resized_image_size
        def repl(match):
            # 提取匹配到的四个数字
            x1, y1, x2, y2 = map(int, match.group(1).split())

            bbox = [x1, y1, x2, y2]
            new_bbox = bbox / np.array([img_w, img_h, img_w, img_h]) * np.array([resized_w, resized_h, resized_w, resized_h])
            x1, y1, x2, y2 = new_bbox

            # 返回修改后的字符串
            return f'{attr_name}=\"{int(x1)} {int(y1)} {int(x2)} {int(y2)}\"'

        # 使用正则表达式查找所有符合模式的部分并进行替换
        return re.sub(rf'{attr_name}=\"([^"]+)\"', repl, input_str)
    
    def swift_format_data(self, image, entry, text, prompt):
        return {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>" + prompt
                    },
                    {
                        "role": "assistant",
                        "content": text
                    },
                ],
                "images": [
                    image
                ]
            }
    
    def format_data(self, image, entry, text, prompt):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ]
