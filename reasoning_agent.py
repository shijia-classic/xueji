#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理Agent - 智能学习助手核心
负责智能决策、内容生成和状态管理
"""

import json
import re
from datetime import datetime
from qwen_client import QwenClient


class ReasoningAgent:
    """推理Agent类"""
    
    def __init__(self, qwen_client=None):
        """
        初始化推理Agent
        
        Args:
            qwen_client: QwenClient实例，如果为None则创建新实例（使用qwen3-vl-plus模型）
        """
        if qwen_client is None:
            self.qwen_client = QwenClient(model="qwen3-vl-plus")
        else:
            self.qwen_client = qwen_client
        self.question_states = {}  # 题目学习状态字典（用于状态追踪）
    
    def make_decision(self, perception_report, frame=None):
        """
        根据感知报告和图像做出智能决策
        
        Args:
            perception_report: 来自感知Agent的情境报告
            frame: OpenCV图像帧（可选，用于更准确的答案判断）
            
        Returns:
            dict: 决策结果JSON对象，如果失败返回None
        """
        if not perception_report:
            return None
        
        # 构建当前状态摘要（用于避免重复操作）
        # 只包含题目的关键状态信息，不包含上次决策结果
        state_summary = self._build_state_summary()
        
        # 准备图像（如果有）
        image_content = []
        if frame is not None:
            image_base64 = self.qwen_client.frame_to_base64(
                frame, 
                quality=self.qwen_client.image_quality, 
                max_size=self.qwen_client.image_max_size
            )
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                },
            })
        
        prompt = f"""你是AI投影学习助手的"推理之心"。你的职责是基于感知Agent提供的学习情境数据，做出智能决策，并生成合适的投影内容。

**重要：每次调用都是独立的，不要参考或重复上次的决策结果。只根据当前的感知报告和题目状态做出新的决策。**

你的核心任务：
1. 情境理解：接收并分析来自感知Agent的【情境报告】和当前图像，理解学生在当前活跃题目上的学习状态、进展和潜在需求。
2. 智能决策：根据意图推断，决定是否需要交互，何时交互，以及采取何种交互方式（提供提示、检查答案、清除投影或保持沉默）。
3. 内容生成：生成高质量、分级、针对性强的投影内容。
4. 答案判断：当需要检查答案时，结合图像中的实际作答内容进行准确判断，分析错误原因。

核心原则：
- **非侵入性至上**：当is_writing为true时，绝不允许进行任何投影交互，应立即清除现有投影并保持沉默。这是最高优先级原则。
- **最小化打扰**：默认情况下应该选择"NO_INTERACTION"，只在真正需要时才投影。不要因为"可以投影"就投影，而是因为"必须投影"才投影。
- **适时介入**：仅在以下情况才考虑投影：
  - 学生长时间停留（超过30秒）且没有任何进展，明显遇到困难
  - 学生完成题目后，主动请求检查答案（通过完成标记或长时间停留）
  - 学生明确表示需要帮助（例如长时间停留在同一题目且无作答）
- **分级提示**：从最轻微的暗示开始，逐步深入，避免直接给出答案。提示级别（Level 1-3）根据当前情况决定。
  - Level 1 (关键词/关键点): 圈出题目中的核心信息，不提供具体解法。投影内容必须在10个字以内。
  - Level 2 (精准引导): 基于用户的user_attempt_content，提供引导性问题或下一步的思路启发。投影内容必须在10个字以内。
  - Level 3 (完整解析): 提供完整的解题思路或关键步骤，作为最后手段。投影内容必须在10个字以内。
- **准确判断与反馈**：判对错必须准确，反馈清晰。如果错误，尝试智能分析原因。
- **简洁自然**：生成的投影内容语言应自然、简洁，像一位耐心的老师。**重要：所有投影内容（projection_content）必须在10个字以内，必须精简到最核心的信息。**
- **保守原则**：当不确定是否需要投影时，选择"NO_INTERACTION"。宁愿少投影，也不要过度打扰。

【当前题目学习状态】（仅用于避免重复操作，不要参考上次决策）：
{json.dumps(state_summary, ensure_ascii=False, indent=2)}

【情境报告】（来自感知Agent）：
{json.dumps(perception_report, ensure_ascii=False, indent=2)}

**重要：所有question_id格式说明**
- 感知报告中的active_question_id和questions_on_page中的id都使用"第xx页-第xx题"格式（例如"第1页-第1题"、"第1页-第2题"、"第2页-第1题"）
- 当需要指定target_question_id时，使用感知报告中active_question_id的值（已经是"第xx页-第xx题"格式）
- 在updated_question_states中，键名也使用"第xx页-第xx题"格式
- 例如：如果题目是第1页的第1题，则question_id为"第1页-第1题"
- 如果题目是第1页的第2题，则question_id为"第1页-第2题"
- 如果题目是第2页的第1题，则question_id为"第2页-第1题"

【当前图像】：
你已接收到当前作业桌面的图像。你可以：
- 查看图像中的实际题目内容
- 识别用户的手写作答内容
- 验证感知Agent提取的OCR文本是否准确
- 进行更准确的答案判断和错误分析

请根据以上信息（包括图像），做出智能决策并返回JSON格式的决策结果。

输出JSON格式：
{{
  "decision_type": "PROJECT_HINT" | "NO_INTERACTION" | "CHECK_ANSWER" | "CLEAR_PROJECTION",
  "target_question_id": "目标题目标识（仅当PROJECT_HINT时，使用'第xx页-第xx题'格式，例如：'第1页-第1题'、'第1页-第2题'等）",
  "hint_level": 1-3（仅当PROJECT_HINT时）,
  "projection_content": "投影内容（仅当PROJECT_HINT时，必须在10个字以内，精简到最核心信息）",
  "reason": "原因说明（仅当NO_INTERACTION或CLEAR_PROJECTION时）",
  "updated_question_states": {{
    "第xx页-第xx题": {{
      "hint_level": 0-3,
      "last_action_type": "hint" | "check_correct" | "check_incorrect",
      "last_action_time": "YYYY-MM-DDTHH:MM:SSZ",
      "status": "in_progress" | "needs_correction" | "completed",
      "error_log": null或错误记录,
      "is_correct": true/false（仅当last_action_type为check_correct或check_incorrect时）,
      "error_analysis": "错误分析（仅当is_correct为false时，15字以内，将作为投影内容显示）"
    }}
  }}（仅包含有变化的状态，如果状态没有变化则为空对象{{}}）
}}

重要决策规则（按优先级排序）：
1. **如果is_writing为true**：必须返回decision_type为"CLEAR_PROJECTION"，reason为"用户正在书写"（最高优先级）。**注意：只有当感知报告明确且确定地显示`is_writing`为true时才返回CLEAR_PROJECTION。如果`is_writing`为false、未提供、或不确定，则不要返回CLEAR_PROJECTION，应该继续其他决策流程。**
2. **优先检查已作答题目（高优先级）**：如果感知报告中`user_attempt_content`中有任何题目的作答内容（非空字符串），且该题目尚未被检查过（通过查看当前题目学习状态，`last_action_type`不是`check_correct`或`check_incorrect`），**必须立即返回"CHECK_ANSWER"**，不要等待`is_active_question_completed`、`time_on_active_question_seconds`等其他条件
3. **如果active_question_id为null**：必须返回"NO_INTERACTION"，reason为"无活跃题目"
4. **如果time_on_active_question_seconds < 30秒**：通常返回"NO_INTERACTION"，reason为"停留时间较短，等待用户继续"（但如果已有作答内容需要检查，则优先检查）
5. **CHECK_ANSWER决策**：**优先且及时检查已作答题目**。当感知报告中`user_attempt_content`中有任何题目的作答内容时，应该立即返回"CHECK_ANSWER"进行判题。判断规则：
   - **只要`user_attempt_content`中有题目的作答内容（非空字符串），就应该立即检查该题目**
   - **不需要等待`is_active_question_completed`为true，只要检测到有作答内容就应该检查**
   - **不需要等待`time_on_active_question_seconds >= 30秒`，发现作答内容就应该立即检查**
   - 如果题目的`last_action_type`已经是`check_correct`或`check_incorrect`，则**绝对不要**再次检查该题目
   - **只检查那些`last_action_type`为`null`、`hint`或不存在，且`status`不是`completed`的题目**
   - 如果所有已作答的题目都已经被检查过，则返回`updated_question_states`为空对象`{{}}`，不要重复输出已检查题目的状态
6. **如果需要提供提示**：只有在time_on_active_question_seconds >= 30秒且用户没有任何进展时才考虑"PROJECT_HINT"。**重要：避免重复提供相同级别的提示**（通过查看当前题目学习状态中的hint_level，如果已经提供过L1提示，下次应该提供L2或L3）。
7. **默认选择**：当不确定时，优先选择"NO_INTERACTION"，保持沉默，不打扰用户
8. **所有时间戳使用ISO 8601格式**
9. **target_question_id格式**：必须使用"第xx页-第xx题"格式，例如第1页的第1题，则target_question_id为"第1页-第1题"；第1页的第2题则为"第1页-第2题"；第2页的第1题则为"第2页-第1题"
10. **CHECK_ANSWER时的状态更新**：当decision_type为"CHECK_ANSWER"时，**必须立即检查感知报告中user_attempt_content中所有有内容的题目**。**重要：只要user_attempt_content中有任何题目的作答内容，就应该检查该题目，不要等待其他条件**。判断规则：
   - **优先检查：只要user_attempt_content中有题目的作答内容（非空字符串），就应该检查该题目**
   - **绝对禁止：如果题目的`last_action_type`已经是`check_correct`或`check_incorrect`，则绝对不要再次检查该题目，也绝对不要在`updated_question_states`中包含该题目**
   - **绝对禁止：如果题目的`status`已经是`completed`，则绝对不要再次检查该题目，也绝对不要在`updated_question_states`中包含该题目**
   - **只检查那些`last_action_type`为`null`、`hint`或不存在，且`status`不是`completed`的题目**
   - **如果所有已作答的题目都已经被检查过（`last_action_type`已经是`check_correct`或`check_incorrect`），则`updated_question_states`必须为空对象`{{}}`，不要包含任何已检查的题目**
   - 对于每个检查的题目，在updated_question_states中更新该题目的状态，包括：
     - last_action_type: "check_correct"或"check_incorrect"
     - is_correct: true/false
     - error_analysis: 如果is_correct为false，提供错误分析（15字以内）
     - status: 根据检查结果更新为"completed"或"needs_correction"
     - last_action_time: 当前时间戳
11. **状态更新（精简输出 - 非常重要）**：
    - **必须比较当前题目学习状态和本次决策后的新状态**
    - **绝对禁止输出已检查题目的状态**：
      * **如果题目状态已经是"completed"且last_action_type已经是"check_correct"或"check_incorrect"，本次绝对不要再次检查该题目，也绝对不要在`updated_question_states`中包含该题目**
      * **如果题目的last_action_type已经是"check_correct"或"check_incorrect"，即使status不是"completed"，也绝对不要再次检查该题目，也绝对不要在`updated_question_states`中包含该题目**
      * **在输出`updated_question_states`之前，必须检查每个题目的`last_action_type`，如果已经是`check_correct`或`check_incorrect`，则绝对不要包含该题目**
    - **只有当状态真正发生变化时才输出**，例如：
      * 如果hint_level没有变化，且没有新的操作，则不要输出
      * 如果本次决策是"NO_INTERACTION"或"CLEAR_PROJECTION"，通常不会有状态变化，updated_question_states应该为空对象{{}}
    - **只在以下情况输出updated_question_states中的某个题目**：
      * 本次决策对某个题目提供了提示（PROJECT_HINT），且hint_level发生了变化
      * 本次决策检查了某个题目的答案（CHECK_ANSWER），**且该题目的last_action_type不是"check_correct"或"check_incorrect"**，且is_correct或status发生了变化
      * 题目的status从"in_progress"变为"completed"或"needs_correction"
    - **如果本次决策没有对任何题目进行操作，或者所有题目的状态都没有变化，updated_question_states必须为空对象{{}}**
    - **绝对不要输出未变化的状态，即使题目在checked_questions中出现，如果状态没有变化也不要输出**
    - **绝对不要重复检查已经检查过的题目（last_action_type已经是check_correct或check_incorrect的题目），也绝对不要在updated_question_states中包含这些题目**

记住：
1. 你的目标是帮助用户，而不是打扰用户。只有在用户真正需要帮助时才投影。
2. **所有投影内容（projection_content）必须在10个字以内**，必须精简到最核心的信息，不要冗长。
3. 错误分析（error_analysis）必须在15字以内。"""

        try:
            # 构建消息内容（包含图像和文本）
            # 注意：每次调用都是独立的，不保留历史上下文
            content = image_content + [{"type": "text", "text": prompt}]
            
            completion = self.qwen_client.client.chat.completions.create(
                model=self.qwen_client.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                timeout=60.0,
                max_tokens=2000
            )
            # 注意：OpenAI API默认是无状态的，每次调用都是独立的，不需要手动清空上下文
            
            response = completion.choices[0].message.content.strip()
            
            # 在终端显示原始响应
            print("\n" + "="*80)
            print("【推理Agent原始响应】")
            print("="*80)
            print(response)
            print("="*80 + "\n")
            
            # 解析JSON响应
            result = self._parse_response(response)
            
            # 过滤掉未变化的状态和已检查的题目（只保留真正变化的状态）
            if result and "updated_question_states" in result:
                updated_states = result["updated_question_states"]
                filtered_states = {}
                
                for question_id, new_state in updated_states.items():
                    old_state = self.question_states.get(question_id)
                    
                    # 如果旧状态不存在，说明是新题目，需要添加
                    if old_state is None:
                        filtered_states[question_id] = new_state
                    else:
                        # 重要：如果题目已经被检查过（last_action_type是check_correct或check_incorrect），
                        # 且新状态也是检查操作，则比较检查结果是否变化
                        old_action_type = old_state.get("last_action_type")
                        new_action_type = new_state.get("last_action_type")
                        
                        # 如果旧状态和新状态都是检查操作，需要比较检查结果是否变化
                        if (old_action_type in ["check_correct", "check_incorrect"] and 
                            new_action_type in ["check_correct", "check_incorrect"]):
                            # 比较检查结果是否变化（例如用户修改了答案，从正确变为错误）
                            old_is_correct = old_state.get("is_correct")
                            new_is_correct = new_state.get("is_correct")
                            
                            # 如果检查结果发生变化（例如从正确变为错误，或从错误变为正确），则允许更新
                            if old_is_correct != new_is_correct:
                                filtered_states[question_id] = new_state
                            # 如果检查结果没有变化，则跳过（避免重复检查）
                            else:
                                continue
                        # 如果旧状态不是检查操作，或新状态不是检查操作，则按正常流程处理
                        else:
                            # 比较状态是否真正发生变化
                            has_changed = False
                            
                            # 检查关键字段是否变化
                            if (new_state.get("hint_level") != old_state.get("hint_level") or
                                new_state.get("last_action_type") != old_state.get("last_action_type") or
                                new_state.get("status") != old_state.get("status") or
                                new_state.get("error_log") != old_state.get("error_log")):
                                has_changed = True
                            
                            if has_changed:
                                filtered_states[question_id] = new_state
                
                # 只更新真正变化的状态
                result["updated_question_states"] = filtered_states
                
                # 更新内部状态（只更新过滤后的状态）
                for question_id, state in filtered_states.items():
                    self.question_states[question_id] = state
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 推理Agent决策失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_state_summary(self):
        """
        构建当前状态摘要（仅包含题目的关键状态信息，用于避免重复操作）
        不包含决策结果，只包含题目的学习状态
        """
        # 只返回题目的关键状态信息，用于避免重复操作
        # 不包含决策相关的信息（如decision_type、reason等）
        return self.question_states.copy()
    
    def _parse_response(self, response):
        """
        解析推理Agent的响应
        
        Args:
            response: API返回的文本
            
        Returns:
            dict: 解析后的决策结果，如果失败返回None
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
                
                return data
            
            return None
            
        except Exception as e:
            print(f"解析推理Agent响应失败: {e}")
            return None

