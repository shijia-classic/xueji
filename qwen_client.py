#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen API客户端
用于数学题目识别
"""

# ==================== 配置区域 ====================
# API Key 通过环境变量 DASHSCOPE_API_KEY 设置
# 获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key
# 
# 设置方法：
# 1. 创建 .env 文件（已添加到 .gitignore，不会被提交）
# 2. 在 .env 文件中添加：DASHSCOPE_API_KEY=your-api-key-here
# 3. 或者直接在终端设置：export DASHSCOPE_API_KEY=your-api-key-here
#
# 区域设置（如果使用北京地域，取消下面的注释并注释掉新加坡地域的URL）
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 北京地域
# BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # 新加坡地域

# 图像优化参数（用于减少API调用时间）
IMAGE_QUALITY = 75  # JPEG压缩质量 (1-100)，平衡速度和识别精度
IMAGE_MAX_SIZE = 1280  # 最大尺寸（像素），720P分辨率，确保框的准确性
# ===================================================

import base64
import io
import json
import os
import re
import time
import cv2
from PIL import Image
from openai import OpenAI

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载 .env 文件
except ImportError:
    # 如果没有安装 python-dotenv，跳过（仍可使用系统环境变量）
    pass

# 尝试从环境变量读取 API Key，如果没有则使用默认值（用于向后兼容）
# 强烈建议使用环境变量，不要在此处硬编码 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


class QwenClient:
    """Qwen API客户端类"""
    
    def _clean_json_string(self, json_str):
        """
        清理和修复格式错误的JSON字符串
        
        Args:
            json_str: 可能有格式错误的JSON字符串
            
        Returns:
            清理后的JSON字符串
        """
        # 修复常见JSON格式错误
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str
    
    def __init__(self, api_key=None, base_url=None, model="qwen3-vl-plus", 
                 image_quality=None, image_max_size=None):
        """
        初始化Qwen API客户端
        
        Args:
            api_key: API密钥，如果为None则使用文件顶部配置的DASHSCOPE_API_KEY
            base_url: API基础URL，如果为None则使用文件顶部配置的BASE_URL
            model: 使用的模型名称，默认为qwen3-vl-plus
            image_quality: JPEG压缩质量 (1-100)，如果为None则使用文件顶部配置的IMAGE_QUALITY
            image_max_size: 最大图像尺寸（像素），如果为None则使用文件顶部配置的IMAGE_MAX_SIZE
        """
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model
        self.image_quality = image_quality if image_quality is not None else IMAGE_QUALITY
        self.image_max_size = image_max_size if image_max_size is not None else IMAGE_MAX_SIZE
        
        if not self.api_key or self.api_key == "your-api-key-here" or self.api_key == "":
            raise ValueError(
                "API Key未设置！\n"
                "请通过以下方式之一设置：\n"
                "1. 创建 .env 文件并添加：DASHSCOPE_API_KEY=your-api-key-here\n"
                "2. 或在终端设置环境变量：export DASHSCOPE_API_KEY=your-api-key-here\n"
                "获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key"
            )
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"[配置] 图像质量: {self.image_quality}, 最大尺寸: {self.image_max_size}px")
    
    def frame_to_base64(self, frame, quality=85, max_size=1280):
        """
        将OpenCV图像转换为base64编码的字符串
        
        Args:
            frame: OpenCV图像（BGR格式）
            quality: JPEG压缩质量 (1-100)，默认85，降低可减少传输时间
            max_size: 最大尺寸，超过会缩放，默认1280，降低可减少传输时间
            
        Returns:
            base64编码的图像字符串（data URI格式）
        """
        # 将BGR转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 如果图像太大，先缩放（保持宽高比）
        h, w = rgb_frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
        
        # 转换为PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # 转换为base64（使用优化的压缩质量）
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_str}"
    
    def detect_math_problems(self, frame):
        """
        识别画面中的数学题目，提取OCR文字内容和边界框
        
        Args:
            frame: OpenCV图像帧
            
        Returns:
            dict: {
                "found": bool,
                "problems": [{"x": float, "y": float, "width": float, "height": float, "text": str}, ...],
            } 或 None
        """
        image_base64 = self.frame_to_base64(frame, 
                                           quality=self.image_quality, 
                                           max_size=self.image_max_size)
        
        prompt = """识别图像中的第一个数学题目，并找到题目下方的作答空白区域。

要求：
1. 识别题目内容（提取前10个字）
2. 找到题目下方的空白作答区域，定位其边界框
3. 检查空白区域是否有作答内容
4. 如果有作答，判断答案是否正确，如果错误需要分析错误原因

只返回第一个找到的题目，不要返回其他题目。

返回格式：
{
    "found": true,
    "problems": [
        {
            "text": "1. 计算：2x + 3",
            "answer_area_bbox": [0.1, 0.4, 0.9, 0.6],
            "answer_text": "作答内容（如果有）",
            "answer_status": "空白" 或 "正确" 或 "错误",
            "error_reason": "错误原因（如果answer_status为'错误'，否则为空字符串）"
        }
    ]
}

未找到题目时：
{
    "found": false,
    "problems": []
}

answer_area_bbox格式：[x_min, y_min, x_max, y_max]，使用归一化坐标0-1，定位题目下方的空白作答区域
answer_text: 作答的具体内容（如果有作答，否则为空字符串）
answer_status: "空白"（如果空白区域没有作答）、"正确"（如果作答正确）、"错误"（如果作答错误）
error_reason: 如果answer_status为"错误"，需要提供简短的错误原因（例如："计算错误"、"符号错误"、"步骤错误"等），否则为空字符串"""

        try:
            # 每次调用都是全新的对话，不保留历史上下文
            # 添加超时设置（60秒）和最大输出tokens限制（加快响应）
            t_api_start = time.time()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                timeout=60.0,  # 60秒超时
                max_tokens=300  # 限制最大输出tokens（包含错误原因，需要更多tokens）
            )
            t_api_end = time.time()
            api_time = (t_api_end - t_api_start) * 1000
            
            response = completion.choices[0].message.content
            
            # 打印API调用时间和原始响应
            print(f"\n[API调用时间] {api_time:.2f}ms")
            print(f"[数学题目识别响应]")
            print(f"{response}")
            print("-" * 50)
            
            # 解析响应
            result = self.parse_math_problems_response(response, frame.shape)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                print("\n⚠️  API Key错误！请检查API Key配置")
            return None
    
    def parse_math_problems_response(self, response, frame_shape):
        """
        解析数学题目识别的响应
        
        Args:
            response: API返回的文本
            frame_shape: 图像尺寸
            
        Returns:
            dict: 包含数学题目位置的字典，或None
        """
        if not response:
            return None
        
        height, width = frame_shape[:2]
        
        try:
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # 清理可能的格式错误
                json_str = self._clean_json_string(json_str)
                
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复常见错误
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    data = json.loads(json_str)
                
                if not data.get("found"):
                    return {"found": False, "problems": []}
                
                problems_list = data.get("problems", [])
                if not isinstance(problems_list, list):
                    return {"found": False, "problems": []}
                
                parsed_problems = []
                for problem in problems_list:
                    if not isinstance(problem, dict):
                        continue
                    
                    # 解析answer_area_bbox格式：[x_min, y_min, x_max, y_max]（作答区域）
                    answer_area_bbox = problem.get("answer_area_bbox", [])
                    if not isinstance(answer_area_bbox, list) or len(answer_area_bbox) < 4:
                        # 兼容旧格式（bbox_2d）
                        bbox_2d = problem.get("bbox_2d", [])
                        if isinstance(bbox_2d, list) and len(bbox_2d) >= 4:
                            answer_area_bbox = bbox_2d
                        else:
                            # 兼容旧格式（x, y, width, height）
                            def get_single_value(value, default=0):
                                """从值中提取单个数字，如果是列表则取第一个"""
                                if isinstance(value, list):
                                    if len(value) > 0:
                                        return float(value[0])
                                    return float(default)
                                return float(value)
                            
                            x_normalized = get_single_value(problem.get("x", 0))
                            y_normalized = get_single_value(problem.get("y", 0))
                            width_normalized = get_single_value(problem.get("width", 0))
                            height_normalized = get_single_value(problem.get("height", 0))
                            
                            x_min = x_normalized
                            y_min = y_normalized
                            x_max = x_normalized + width_normalized
                            y_max = y_normalized + height_normalized
                            answer_area_bbox = [x_min, y_min, x_max, y_max]
                    
                    if isinstance(answer_area_bbox, list) and len(answer_area_bbox) >= 4:
                        # 新格式：answer_area_bbox = [x_min, y_min, x_max, y_max]
                        x_min = float(answer_area_bbox[0])
                        y_min = float(answer_area_bbox[1])
                        x_max = float(answer_area_bbox[2])
                        y_max = float(answer_area_bbox[3])
                    else:
                        continue
                    
                    # 转换为像素坐标（归一化坐标0-1直接乘以宽度/高度）
                    x = int(x_min * width)
                    y = int(y_min * height)
                    x_max_pixel = int(x_max * width)
                    y_max_pixel = int(y_max * height)
                    answer_width = x_max_pixel - x
                    answer_height = y_max_pixel - y
                    
                    # 确保坐标在范围内
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    answer_width = max(20, min(answer_width, width - x))
                    answer_height = max(10, min(answer_height, height - y))
                    
                    # 提取题目文字内容
                    problem_text = problem.get("text", "")
                    
                    # 提取作答状态和内容
                    answer_status = problem.get("answer_status", "空白")
                    answer_text = problem.get("answer_text", "")
                    error_reason = problem.get("error_reason", "")
                    
                    # 确保answer_status是有效值
                    if answer_status not in ["空白", "正确", "错误"]:
                        answer_status = "空白"
                    
                    parsed_problems.append({
                        "x": x,
                        "y": y,
                        "width": answer_width,
                        "height": answer_height,
                        "text": problem_text,
                        "answer_status": answer_status,
                        "answer_text": answer_text,
                        "error_reason": error_reason
                    })
                
                return {
                    "found": len(parsed_problems) > 0,
                    "problems": parsed_problems
                }
                    
        except Exception as e:
            print(f"解析数学题目响应错误: {e}")
            pass
        
        return None
