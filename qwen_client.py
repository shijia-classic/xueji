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
    
    def __init__(self, api_key=None, base_url=None, model="qwen-vl-max", 
                 image_quality=None, image_max_size=None):
        """
        初始化Qwen API客户端
        
        Args:
            api_key: API密钥，如果为None则使用文件顶部配置的DASHSCOPE_API_KEY
            base_url: API基础URL，如果为None则使用文件顶部配置的BASE_URL
            model: 使用的模型名称，默认为qwen-vl-max
            image_quality: JPEG压缩质量 (1-100)，如果为None则使用文件顶部配置的IMAGE_QUALITY
            image_max_size: 最大图像尺寸（像素），如果为None则使用文件顶部配置的IMAGE_MAX_SIZE
        """
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model
        self.image_quality = image_quality if image_quality is not None else IMAGE_QUALITY
        self.image_max_size = image_max_size if image_max_size is not None else IMAGE_MAX_SIZE
        
        if not self.api_key or self.api_key == "your-api-key-here" or self.api_key == "":
            raise ValueError("API Key未设置，请配置DASHSCOPE_API_KEY")
        
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
        
        prompt = """识别图像中的所有数学题目，并检测每个题目的作答情况。

要求：
1. 识别画面中的所有数学题目（从上到下，从左到右）
2. 对每个题目：
   - 识别题目内容（提取前10个字）
   - 找到题目下方的作答空白区域，定位其边界框[x_min, y_min, x_max, y_max]（归一化坐标0-1）
   - 检查空白区域是否有作答内容
   - 如果有作答，判断答案是否正确，如果错误需要分析错误原因

返回格式（必须返回JSON格式）：

{
    "found": true,
    "problems": [
        {
            "text": "1. 计算：2x + 3",
            "question_bbox": [0.1, 0.2, 0.9, 0.35],
            "answer_area_bbox": [0.1, 0.4, 0.9, 0.6],
            "answer_text": "作答内容（如果有）",
            "answer_status": "空白" 或 "正确" 或 "错误",
            "error_reason": "错误原因（如果answer_status为'错误'，15字以内，否则为空字符串）"
        },
        {
            "text": "2. 解方程：x + 5 = 10",
            "question_bbox": [0.1, 0.6, 0.9, 0.75],
            "answer_area_bbox": [0.1, 0.8, 0.9, 0.95],
            "answer_text": "",
            "answer_status": "空白",
            "error_reason": ""
        }
    ]
}

未找到题目时：
{
    "found": false,
    "problems": []
}

说明：
- 必须返回所有找到的题目，不要遗漏
- question_bbox: 题目区域的边界框 [x_min, y_min, x_max, y_max]（归一化坐标0-1）
- answer_area_bbox: 作答区域的边界框 [x_min, y_min, x_max, y_max]（归一化坐标0-1）
- answer_status: "空白"（无作答）、"正确"（作答正确）、"错误"（作答错误）
- error_reason: 仅在answer_status为"错误"时提供，15字以内
- 空白题目也需要返回完整的JSON格式，包含question_bbox和answer_area_bbox"""

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
                max_tokens=1000  # 增加tokens限制以支持多个题目
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
                print("\n⚠️  API Key错误")
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
            # 提取JSON格式（现在所有情况都返回JSON）
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
                    
                    # 解析answer_area_bbox格式
                    # 空白时：[x_min, y_min]（两个数字）
                    # 有作答时：[x_min, y_min, x_max, y_max]（四个数字）
                    answer_area_bbox = problem.get("answer_area_bbox", [])
                    
                    if not isinstance(answer_area_bbox, list) or len(answer_area_bbox) < 2:
                        # 兼容旧格式（bbox_2d）
                        bbox_2d = problem.get("bbox_2d", [])
                        if isinstance(bbox_2d, list) and len(bbox_2d) >= 4:
                            answer_area_bbox = bbox_2d
                        elif isinstance(bbox_2d, list) and len(bbox_2d) >= 2:
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
                            
                            if width_normalized > 0 and height_normalized > 0:
                                # 有宽高，转换为完整bbox
                                answer_area_bbox = [x_normalized, y_normalized, 
                                                   x_normalized + width_normalized, 
                                                   y_normalized + height_normalized]
                            else:
                                # 只有坐标，使用两个数字格式
                                answer_area_bbox = [x_normalized, y_normalized]
                    
                    if isinstance(answer_area_bbox, list) and len(answer_area_bbox) >= 2:
                        x_min = float(answer_area_bbox[0])
                        y_min = float(answer_area_bbox[1])
                        
                        if len(answer_area_bbox) >= 4:
                            # 完整边界框：[x_min, y_min, x_max, y_max]
                            x_max = float(answer_area_bbox[2])
                            y_max = float(answer_area_bbox[3])
                        else:
                            # 只有左上角坐标（空白时），使用默认尺寸
                            # 默认宽度为画面宽度的80%，高度为画面高度的10%
                            x_max = min(x_min + 0.8, 1.0)
                            y_max = min(y_min + 0.1, 1.0)
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
                    
                    # 提取题目区域位置（如果有）
                    question_bbox = problem.get("question_bbox", [])
                    question_x = None
                    question_y = None
                    if isinstance(question_bbox, list) and len(question_bbox) >= 4:
                        q_x_min = float(question_bbox[0])
                        q_y_min = float(question_bbox[1])
                        q_x_max = float(question_bbox[2])
                        q_y_max = float(question_bbox[3])
                        # 题目区域中心下方位置（用于显示问号）
                        question_x = int((q_x_min + q_x_max) / 2 * width)
                        question_y = int(q_y_max * height)
                    
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
                        "error_reason": error_reason,
                        "question_x": question_x,
                        "question_y": question_y
                    })
                
                return {
                    "found": len(parsed_problems) > 0,
                    "problems": parsed_problems
                }
                    
        except Exception as e:
            print(f"解析响应失败")
            pass
        
        return None
    
    def detect_finger_tip(self, frame):
        """
        使用大模型API检测手指指尖位置（多阶段检测）
        
        Args:
            frame: OpenCV图像帧
            
        Returns:
            tuple: (x, y) 手指指尖位置（归一化坐标0-1），如果未检测到返回None
        """
        image_base64 = self.frame_to_base64(frame, 
                                           quality=self.image_quality, 
                                           max_size=self.image_max_size)
        
        # 多阶段prompt：第一阶段快速检测，第二阶段精确定位
        prompt = """检测图像中是否有手指/手部，如果有，定位手指指尖的精确位置。

要求：
1. 快速判断图像中是否有可见的手指或手部
2. 如果检测到手指，定位最靠近画面中心或最突出的手指指尖位置
3. 优先检测食指或中指指尖

返回格式：
- 如果检测到手指指尖，返回归一化坐标（0-1范围）：
[x, y]

- 如果未检测到手指或手部，返回：
null

说明：
- 坐标使用归一化格式，x和y都在0-1之间
- x=0表示画面左边缘，x=1表示画面右边缘
- y=0表示画面上边缘，y=1表示画面下边缘
- 只返回一个坐标点，选择最明显或最靠近中心的指尖
- 如果没有手指，直接返回null，不要返回JSON格式"""

        try:
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
                timeout=30.0,  # 30秒超时（手指检测需要快速响应）
                max_tokens=50  # 只需要返回坐标，token数很少
            )
            
            response = completion.choices[0].message.content.strip()
            
            # 解析响应
            # 情况1：返回坐标 [x, y]
            coord_match = re.search(r'\[?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]?', response)
            if coord_match:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                
                # 确保坐标在0-1范围内
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                
                return (x, y)
            
            # 情况2：返回null或未检测到
            if "null" in response.lower() or "未检测" in response or "没有" in response:
                return None
            
            return None
            
        except Exception as e:
            # 检测失败时返回None，不影响主流程
            return None
