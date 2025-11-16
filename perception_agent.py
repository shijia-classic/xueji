#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知Agent - AI投影学习助手的"感知之眼"
负责分析图像内容，提取学生当前的学习情境数据
"""

import json
import re
import time
from datetime import datetime
from qwen_client import QwenClient


class PerceptionAgent:
    """感知Agent类"""
    
    def __init__(self, qwen_client=None):
        """
        初始化感知Agent
        
        Args:
            qwen_client: QwenClient实例，如果为None则创建新实例（使用qwen3-vl-plus模型）
        """
        if qwen_client is None:
            self.qwen_client = QwenClient(model="qwen3-vl-plus")
        else:
            self.qwen_client = qwen_client
        self.previous_active_question_id = None
    
    def analyze_scene(self, frame, reasoning_feedback=None):
        """
        分析当前场景，提取学习情境数据
        
        Args:
            frame: OpenCV图像帧
            reasoning_feedback: 推理Agent的反馈信息（可选），用于优化感知
            
        Returns:
            dict: 情境报告JSON对象，如果失败返回None
        """
        image_base64 = self.qwen_client.frame_to_base64(
            frame, 
            quality=self.qwen_client.image_quality, 
            max_size=self.qwen_client.image_max_size
        )
        
        # 构建反馈信息（如果有）
        feedback_text = ""
        if reasoning_feedback:
            feedback_json = json.dumps(reasoning_feedback, ensure_ascii=False, indent=2)
            feedback_text = f"""

【推理Agent反馈信息】：
{feedback_json}

请利用以上反馈信息来优化你的感知判断。例如：
- 如果推理Agent刚提供了提示，关注用户是否开始尝试解题
- 如果推理Agent标记了题目状态，根据状态调整你的关注重点
- 如果推理Agent提供了错误分析，验证用户是否已修正错误
"""
        
        # 使用字符串拼接而不是f-string来避免大括号转义问题
        prompt = """你的任务是作为AI投影学习助手的"感知之眼"。你将接收当前作业桌面的图像，并基于以下指令，分析图像内容，提取学生当前的学习情境数据。

**重要原则：你只负责客观观察和提取信息，不做任何答案正确性判断。所有判断（包括答案对错、错误原因等）都由推理Agent负责。**
""" + feedback_text + """

**你的主要目标是：**

1. **识别并追踪书本与页面：** 确定作业本的存在、其打开的页面以及页面上的内容边界。

2. **识别页面上的所有题目：** 精确识别页面上每一道数学题的视觉边界（Bounding Box）。为每个题目生成唯一标识符：使用"第xx题"格式（例如第1题、第2题、第3题等，根据题目在页面上的顺序编号）。

3. **确定用户焦点题目：** 根据用户的凝视方向、书写区域或最近交互区域，判断用户当前最可能在处理的**活跃题目ID**（使用"第xx题"格式）。

4. **识别用户书写行为：** 判断用户是否正在进行书写动作 (`is_writing`)。

5. **提取用户已作答内容：** 对用户在每道题目区域内已写下的手写或打印内容进行OCR识别，并转化为文本。**注意：只提取文本内容，不做任何正确性判断。**

6. **估算题目停留时间：** 追踪用户在**活跃题目**上停留的视觉时间。

7. **判断题目作答完成度：** 尝试评估**活跃题目**是否已完成（例如，所有空白区域被填满，或有明确的答案标记）。**注意：只判断完成度，不做答案正确性判断。**

8. **（可选）识别上次投影的反馈信息：** 如果上次有投影，识别投影内容是否仍在屏幕上或是否已被覆盖。

9. **利用推理Agent的反馈：** 如果提供了`system_internal_state_feedback`，利用这些信息来优化你的感知判断。例如：
   - 如果推理Agent刚提供了L1提示，你可能需要更关注用户是否开始尝试解题
   - 如果推理Agent标记了某个题目为"needs_correction"，你需要特别关注该题目的作答变化
   - 如果推理Agent标记了某个题目为"completed"，你可以减少对该题目的关注

**输出数据格式（你需要以JSON格式返回分析结果）：**

请严格按照以下JSON结构输出，所有字段都必须存在，如果无法识别，请使用`null`或空值。

```json
{
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "current_page_id": "page_A_unique_id",
  "active_question_id": "第1题",
  "previous_active_question_id": "第2题",
  "questions_on_page": [
    {"id": "第1题", "text": "1. 计算：2x + 3", "bbox": [x1,y1,x2,y2]},
    {"id": "第2题", "text": "2. 解方程：x + 5 = 10", "bbox": [x1,y1,x2,y2]},
    {"id": "第3题", "text": "3. 求值：...", "bbox": [x1,y1,x2,y2]}
  ],
  "time_on_active_question_seconds": 75,
  "is_writing": false,
  "user_attempt_content": {
    "第1题": "用户在Q4的答案...",
    "第2题": "3x + 5 = 14\\n3x = 14 - 5\\n3x = 9"
  },
  "is_active_question_completed": false,
  "system_internal_state_feedback": {}
}
```

说明：
- timestamp: 当前时间戳，ISO 8601格式
- current_page_id: 当前识别到的物理页面ID，如果无法识别为"unknown"
- active_question_id: 用户当前焦点所在的题目ID（使用"第xx题"格式，例如"第1题"、"第2题"），如果无焦点则为null
- previous_active_question_id: 上一次识别到的活跃题目ID（使用"第xx题"格式），用于追踪切换
- questions_on_page: 当前页面上所有可识别的题目列表，每个题目包含：
  - id: 题目唯一标识符（使用"第xx题"格式，例如"第1题"、"第2题"，根据题目序号生成）
  - text: 题目文本内容（前10个字，用于识别）
  - bbox: 题目边界框，归一化坐标 [x_min, y_min, x_max, y_max] (0-1范围)
- time_on_active_question_seconds: 用户在active_question_id上的估算停留时间（秒）
- is_writing: 布尔值，用户是否正在书写
- user_attempt_content: 用户对当前页面上所有题目的作答内容（OCR文本），以题目ID为键
- is_active_question_completed: 布尔值，当前活跃题目是否已完成
- system_internal_state_feedback: 从智能交互Agent获取的关于题目状态的反馈信息（可选）"""

        try:
            completion = self.qwen_client.client.chat.completions.create(
                model=self.qwen_client.model,
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
                timeout=60.0,
                max_tokens=2000
            )
            
            response = completion.choices[0].message.content.strip()
            
            # 在终端显示原始响应
            print("\n" + "="*80)
            print("【感知Agent原始响应】")
            print("="*80)
            print(response)
            print("="*80 + "\n")
            
            # 解析JSON响应（传入推理反馈）
            result = self._parse_response(response, frame.shape, reasoning_feedback)
            
            # 更新previous_active_question_id
            if result and result.get("active_question_id"):
                self.previous_active_question_id = result.get("active_question_id")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 感知Agent分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_response(self, response, frame_shape, reasoning_feedback=None):
        """
        解析感知Agent的响应
        
        Args:
            response: API返回的文本
            frame_shape: 图像尺寸
            reasoning_feedback: 推理Agent的反馈信息（可选）
            
        Returns:
            dict: 解析后的情境报告，如果失败返回None
        """
        if not response:
            return None
        
        try:
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # 清理可能的格式错误
                json_str = self.qwen_client._clean_json_string(json_str)
                
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复常见错误
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    data = json.loads(json_str)
                
                # 确保时间戳存在
                if "timestamp" not in data:
                    data["timestamp"] = datetime.now().isoformat() + "Z"
                
                # 设置previous_active_question_id
                if "previous_active_question_id" not in data:
                    data["previous_active_question_id"] = self.previous_active_question_id
                
                # 设置system_internal_state_feedback（如果有推理反馈）
                if reasoning_feedback:
                    data["system_internal_state_feedback"] = reasoning_feedback
                elif "system_internal_state_feedback" not in data:
                    data["system_internal_state_feedback"] = {}
                
                # 转换bbox坐标（归一化坐标转换为像素坐标）
                height, width = frame_shape[:2]
                if "questions_on_page" in data and isinstance(data["questions_on_page"], list):
                    for question in data["questions_on_page"]:
                        if "bbox" in question and isinstance(question["bbox"], list) and len(question["bbox"]) >= 4:
                            # 归一化坐标转像素坐标
                            bbox_norm = question["bbox"]
                            question["bbox_pixel"] = [
                                int(bbox_norm[0] * width),
                                int(bbox_norm[1] * height),
                                int(bbox_norm[2] * width),
                                int(bbox_norm[3] * height)
                            ]
                
                return data
            
            return None
            
        except Exception as e:
            print(f"解析感知Agent响应失败: {e}")
            return None

