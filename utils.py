import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_default_dict():
    """默认的内置词库"""
    return {
        'wo': ['我', '窝', '握'],
        'ni': ['你', '泥', '拟'],
        'ta': ['他', '她', '它'],
        'hao': ['好', '号', '豪'],
        'shi': ['是', '时', '十']
    }

def load_chinese_dict(file_path="chinese_dict.json"):
    """从JSON文件加载中文词库"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"词库文件 {file_path} 未找到，使用内置词库")
        return get_default_dict()  # 返回一个默认的小词库
    except json.JSONDecodeError as e:
        print(f"JSON格式错误: {e}")
        print("使用默认词库")
        return get_default_dict()
    except Exception as e:
        print(f"加载词库失败: {e}，使用内置词库")
        return get_default_dict()




