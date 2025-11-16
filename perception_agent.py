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
    
    def analyze_scene(self, frame, previous_perception_state=None):
        """
        分析当前场景，提取学习情境数据
        
        Args:
            frame: OpenCV图像帧
            previous_perception_state: 上一次的感知状态（可选），用于比较变化
            
        Returns:
            dict: 情境报告JSON对象（只包含变化的部分），如果失败返回None
        """
        image_base64 = self.qwen_client.frame_to_base64(
            frame, 
            quality=self.qwen_client.image_quality, 
            max_size=self.qwen_client.image_max_size
        )
        
        # 构建上一次感知状态信息（如果有）
        previous_state_text = ""
        if previous_perception_state:
            previous_state_json = json.dumps(previous_perception_state, ensure_ascii=False, indent=2)
            previous_state_text = f"""

【上一次感知状态】（用于比较变化）：
{previous_state_json}

**重要：你需要比较当前图像和上一次感知状态，只输出发生变化的部分。**
"""
        else:
            # 第一次调用，需要输出完整数据
            previous_state_text = """

**重要：这是第一次调用，请输出完整的感知数据，包括所有字段。**
"""
        
        # 使用字符串拼接而不是f-string来避免大括号转义问题
        prompt = """你的任务是作为AI投影学习助手的"感知之眼"。你将接收当前作业桌面的图像，并基于以下指令，分析图像内容，提取学生当前的学习情境数据。

**重要原则：你只负责客观观察和提取信息，不做任何答案正确性判断。所有判断（包括答案对错、错误原因等）都由推理Agent负责。**
""" + previous_state_text + """

**你的主要目标是：**

1. **识别并追踪书本与页面：** 确定作业本的存在、其打开的页面以及页面上的内容边界。

2. **识别页面上的所有题目：** 精确识别页面上每一道数学题的视觉边界（Bounding Box）。为每个题目生成唯一标识符：使用"第xx页-第xx题"格式（例如第1页-第1题、第1页-第2题、第2页-第1题等）。格式说明：
   - 第一部分"第xx页"：使用`current_page_id`对应的页码编号（如果`current_page_id`是"page_242_243"，则使用"第242页"或"第243页"，根据实际情况选择）
   - 第二部分"第xx题"：根据题目在页面上的顺序编号（例如第1题、第2题、第3题等）
   - 示例：如果`current_page_id`是"page_242_243"，第一个题目ID为"第242页-第1题"，第二个题目ID为"第242页-第2题"

3. **确定用户焦点题目：** 根据用户的凝视方向、书写区域或最近交互区域，判断用户当前最可能在处理的**活跃题目ID**（使用"第xx页-第xx题"格式）。

4. **识别用户书写行为：** 判断用户是否正在进行书写动作 (`is_writing`)。**极其重要：`is_writing`的判断必须非常严格和保守。只有当图像中同时满足以下所有条件时，才将`is_writing`设置为true：**
   - **明确看到笔尖或手指正在纸上移动或接触**
   - **能够看到笔或手指在纸上留下痕迹或正在书写**
   - **用户的手部姿势明显是书写姿势（握笔姿势）**
   - **有明显的书写动作（不是静止的）**
   
   **以下情况`is_writing`必须为false：**
   - 用户只是看着题目或思考（手不在纸上）
   - 用户的手在画面中但不在纸上
   - 用户的手在纸上但没有书写动作（只是放在那里）
   - 用户的手在纸上但姿势不是书写姿势（例如只是指着题目）
   - 无法明确判断是否有书写动作
   - 有任何不确定的情况
   
   **保守原则：当不确定时，`is_writing`必须为false。宁愿误判为false，也不要误判为true。**

5. **提取用户已作答内容：** 对用户在每道题目区域内已写下的手写或打印内容进行OCR识别，并转化为文本。**注意：只提取文本内容，不做任何正确性判断。**

6. **估算题目停留时间：** 追踪用户在**活跃题目**上停留的视觉时间。

7. **判断题目作答完成度：** 尝试评估**活跃题目**是否已完成（例如，所有空白区域被填满，或有明确的答案标记）。**注意：只判断完成度，不做答案正确性判断。**


**输出数据格式（你需要以JSON格式返回分析结果）：**

**重要输出规则**：
- 如果是第一次调用（没有上一次感知状态），请输出完整的感知数据，包括所有字段
- 如果不是第一次调用，只输出发生变化的部分。如果某个字段没有变化，可以省略该字段（但timestamp必须始终存在）

请严格按照以下JSON结构输出：

```json
{
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "current_page_id": "page_A_unique_id",
  "active_question_id": "第1页-第1题",
  "questions_on_page": [
    {"id": "第1页-第1题", "text": "1. 计算：2x + 3", "bbox": [x1,y1,x2,y2]},
    {"id": "第1页-第2题", "text": "2. 解方程：x + 5 = 10", "bbox": [x1,y1,x2,y2]},
    {"id": "第1页-第3题", "text": "3. 求值：...", "bbox": [x1,y1,x2,y2]}
  ],
  "time_on_active_question_seconds": 75,
  "is_writing": false,
  "user_attempt_content": {
    "第1页-第1题": "用户在Q4的答案...",
    "第1页-第2题": "3x + 5 = 14\\n3x = 14 - 5\\n3x = 9"
  },
  "is_active_question_completed": false
}
```

说明：
- timestamp: 当前时间戳，ISO 8601格式（必须始终存在）
- current_page_id: 当前识别到的物理页面ID，如果无法识别为"unknown"（仅当页面变化时输出）
- active_question_id: 用户当前焦点所在的题目ID（使用"第xx页-第xx题"格式，例如"第1页-第1题"、"第1页-第2题"），如果无焦点则为null（仅当活跃题目变化时输出）
- questions_on_page: 当前页面上所有可识别的题目列表（仅当题目列表变化时输出，例如新增或删除题目）
  - id: 题目唯一标识符（使用"第xx页-第xx题"格式，例如"第1页-第1题"、"第1页-第2题"，根据页面ID和题目序号生成）
  - text: 题目文本内容（前10个字，用于识别）
  - bbox: 题目边界框，**归一化坐标** [x_min, y_min, x_max, y_max]（0-1范围的小数，例如[0.39, 0.13, 0.68, 0.22]）
- time_on_active_question_seconds: 用户在active_question_id上的估算停留时间（秒）（仅当时间变化时输出）
- is_writing: 布尔值，用户是否正在书写（仅当书写状态变化时输出）
- user_attempt_content: 用户对当前页面上所有题目的作答内容（OCR文本），以题目ID为键（仅当作答内容变化时输出，只包含有变化的题目）。**重要：如果某个题目之前有作答内容，但本次没有变化，则不需要输出该题目，系统会自动保留之前的内容。**
- is_active_question_completed: 布尔值，当前活跃题目是否已完成（仅当完成状态变化时输出）

**重要输出规则**：
- 如果某个字段的值与上一次感知状态相同，则不要输出该字段
- 如果user_attempt_content中的某个题目内容没有变化，则不要包含该题目（系统会自动保留之前的内容）
- **重要：user_attempt_content只需要输出有变化的题目，没有变化的题目不需要输出，系统会从之前的状态中保留**
- 如果questions_on_page中的题目列表没有变化，则不要输出该字段
- 如果所有字段都没有变化，只输出timestamp字段

**关于user_attempt_content的说明**：
- 如果某个题目之前有作答内容，但本次内容没有变化，则不需要在user_attempt_content中包含该题目
- 系统会自动从之前的状态中保留该题目的作答内容
- 只有当题目的作答内容发生变化（新增、修改或删除）时，才需要在user_attempt_content中包含该题目"""

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
            
            # 解析JSON响应
            result = self._parse_response(response, frame.shape, previous_perception_state)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 感知Agent分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_response(self, response, frame_shape, previous_perception_state=None):
        """
        解析感知Agent的响应，并合并到之前的感知状态中
        
        Args:
            response: API返回的文本
            frame_shape: 图像尺寸
            previous_perception_state: 上一次的感知状态（可选），用于合并变化
            
        Returns:
            dict: 解析后的完整情境报告（合并变化后的完整状态），如果失败返回None
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
                
                # 合并到之前的感知状态（如果有）
                if previous_perception_state:
                    # 从之前的状态开始，用新数据覆盖变化的字段
                    merged_data = previous_perception_state.copy()
                    
                    # 特殊处理：先处理user_attempt_content，避免被update覆盖
                    # 保留之前的所有已作答题目
                    if "user_attempt_content" not in merged_data:
                        merged_data["user_attempt_content"] = {}
                    
                    # 如果新数据中有user_attempt_content，更新或添加变化的题目
                    if "user_attempt_content" in data:
                        # 更新或添加新变化的题目内容
                        merged_data["user_attempt_content"].update(data["user_attempt_content"])
                        # 注意：如果data中没有某个题目，说明该题目内容没有变化，保留之前的内容
                    
                    # 特殊处理：如果questions_on_page存在，完全替换（因为题目列表变化需要完整更新）
                    if "questions_on_page" in data:
                        merged_data["questions_on_page"] = data["questions_on_page"]
                    
                    # 更新其他字段（但不覆盖user_attempt_content）
                    for key, value in data.items():
                        if key not in ["user_attempt_content", "questions_on_page"]:
                            merged_data[key] = value
                    
                    data = merged_data
                
                # bbox坐标处理：感知Agent返回归一化坐标（0-1范围），需要转换为像素坐标
                # 如果格式错误，不进行修正，直接返回None
                height, width = frame_shape[:2]
                
                # 验证数据格式是否正确
                has_format_error = False
                
                if "questions_on_page" in data and isinstance(data["questions_on_page"], list):
                    for question in data["questions_on_page"]:
                        if "bbox" in question and isinstance(question["bbox"], list) and len(question["bbox"]) >= 4:
                            bbox_norm = question["bbox"]
                            
                            # 感知Agent返回的bbox应该是归一化坐标（0-1范围）
                            x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3]
                            
                            # 验证是否为归一化坐标（0-1范围）
                            if (0 <= x1_norm <= 1.0 and 0 <= y1_norm <= 1.0 and
                                0 <= x2_norm <= 1.0 and 0 <= y2_norm <= 1.0 and
                                x1_norm < x2_norm and y1_norm < y2_norm):
                                # 是归一化坐标，转换为像素坐标
                                question["bbox_pixel"] = [
                                    int(x1_norm * width),
                                    int(y1_norm * height),
                                    int(x2_norm * width),
                                    int(y2_norm * height)
                                ]
                            else:
                                # 坐标格式异常，标记为格式错误
                                has_format_error = True
                                break
                        else:
                            # bbox字段缺失或格式错误
                            has_format_error = True
                            break
                
                # 如果检测到格式错误，返回None，不更新状态
                if has_format_error:
                    return None
                
                return data
            
            return None
            
        except Exception as e:
            print(f"解析感知Agent响应失败: {e}")
            return None

