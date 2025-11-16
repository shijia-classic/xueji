#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI投影学习助手 - 双Agent架构
使用感知Agent和推理Agent实现智能学习辅助
"""

import cv2
import numpy as np
import threading
import time
import traceback
from PIL import Image, ImageDraw, ImageFont
from qwen_client import QwenClient
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent


class AIProjectionLearningAssistant:
    """AI投影学习助手主类"""
    
    def __init__(self, camera_index=0):
        """初始化AI投影学习助手"""
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        
        # 初始化双Agent（都使用qwen3-vl-plus模型）
        self.perception_agent = PerceptionAgent()  # 使用qwen3-vl-plus
        self.reasoning_agent = ReasoningAgent()  # 使用qwen3-vl-plus
        
        # 当前感知报告（用于绘制投影时获取最新感知数据）
        self.current_perception_report = None
        self.data_lock = threading.Lock()
        
        # 题目学习状态（从推理Agent获取并维护）
        self.question_states = {}  # 格式：{"第1页-第1题": {"hint_level": 0, "status": "in_progress", ...}, ...}
        
        # 感知状态（从感知Agent获取并维护，用于比较变化）
        self.perception_states = {}  # 格式：{"current_page_id": "...", "questions_on_page": [...], "user_attempt_content": {...}, ...}
        
        # 推理决策状态（从推理Agent获取并维护，用于投影显示）
        self.decision_states = {}  # 格式：{"decision_type": "...", "target_question_id": "...", "projection_content": "...", "updated_question_states": {...}, ...}
        
        # 检测状态
        self.is_analyzing = False
        self.analysis_lock = threading.Lock()
        
        # API调用间隔控制（500ms，从上次检测完成开始计算）
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # 500ms间隔
        
        
        # 最新帧缓存（确保检测使用最新画面）
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        
        # 720P显示尺寸
        self.display_width = 1280
        self.display_height = 720
        
        # 全屏标志
        self.fullscreen = False
    
    def init_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"无法打开摄像头 {self.camera_index}")
        
        # 设置摄像头为720P
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {actual_width}x{actual_height}")
        
        cv2.namedWindow('AI投影学习助手', cv2.WINDOW_NORMAL)
    
    def draw_projection(self, frame, perception_report, decision):
        """
        绘制投影内容到画布
        
        Args:
            frame: 原始视频帧
            perception_report: 感知报告
            decision: 推理决策结果（全局维护的决策状态）
            
        Returns:
            np.ndarray: 绘制后的画布
        """
        try:
            h, w = frame.shape[:2]
            # 创建全黑画布
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 在右侧绘制竖线分隔状态显示区域（尽量靠右，90%位置）
            divider_x = int(w * 0.9)
            cv2.line(canvas, (divider_x, 0), (divider_x, h), (64, 64, 64), 1)
            
            if not perception_report:
                return canvas
            
            # 不再绘制题目边界框，只保留投影内容显示
            questions = perception_report.get("questions_on_page", [])
            active_question_id = perception_report.get("active_question_id")
            
            # 根据决策显示投影内容
            if decision:
                decision_type = decision.get("decision_type")
                target_question_id = decision.get("target_question_id")
                projection_content = decision.get("projection_content")
            
            # 显示所有已检查题目的结果（无论decision_type是什么，只要题目有检查结果就显示）
            # 这样可以保留对号和错误标记的投影
            with self.data_lock:
                all_question_states = self.question_states.copy()
            
            # 为每个已检查的题目显示结果（只显示last_action_type为check_correct或check_incorrect的）
            for question_id, state in all_question_states.items():
                last_action_type = state.get("last_action_type")
                
                # 只处理检查答案的操作
                if last_action_type in ["check_correct", "check_incorrect"]:
                    is_correct = state.get("is_correct", True)
                    error_analysis = state.get("error_analysis", "")
                    
                    # 找到题目的位置
                    target_question = None
                    for question in questions:
                        if question.get("id") == question_id:
                            target_question = question
                            break
                    
                    if target_question:
                        # 优先使用bbox_pixel，如果没有则从bbox转换
                        if target_question.get("bbox_pixel"):
                            bbox = target_question["bbox_pixel"]
                            x1, y1, x2, y2 = bbox
                        elif target_question.get("bbox"):
                            # 如果只有归一化坐标，需要转换
                            bbox_norm = target_question["bbox"]
                            h_frame, w_frame = frame.shape[:2]
                            x1 = int(bbox_norm[0] * w_frame)
                            y1 = int(bbox_norm[1] * h_frame)
                            x2 = int(bbox_norm[2] * w_frame)
                            y2 = int(bbox_norm[3] * h_frame)
                        else:
                            continue
                        
                        # 验证坐标是否合理（应该在图像范围内）
                        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                            # 坐标异常，跳过绘制
                            continue
                        
                        if is_correct:
                            # 答对了，绘制对号图形
                            # 投影位置：题目左下方，x坐标不变
                            checkmark_x = x1  # x坐标不变，使用题目左边界
                            checkmark_y = y2 - 10  # 题目底部向上10像素
                            # 确保坐标在图像范围内
                            checkmark_x = max(0, min(checkmark_x, w - 50))
                            checkmark_y = max(0, min(checkmark_y, h - 50))
                            canvas = self.draw_checkmark(canvas, (checkmark_x, checkmark_y), size=30, color=(0, 255, 0))
                        else:
                            # 答错了，显示错误分析文字
                            if error_analysis:
                                text_color = (255, 255, 255)  # 白色
                                # 投影位置：题目左下方，x坐标不变
                                text_x = x1  # x坐标不变，使用题目左边界
                                text_y = y2 - 5  # 题目底部向上5像素
                                # 确保坐标在图像范围内
                                text_x = max(0, min(text_x, w - 200))
                                text_y = max(0, min(text_y, h - 1))
                                canvas = self.put_text(canvas, error_analysis, (text_x, text_y),
                                             font_size=18, color=text_color)
            
            # 根据决策显示其他投影内容（提示等）
            if decision:
                decision_type = decision.get("decision_type")
                target_question_id = decision.get("target_question_id")
                projection_content = decision.get("projection_content")
                
                # CHECK_ANSWER的投影已经在上面统一处理了，这里不需要重复处理
                
                # 显示投影内容（如果有）
                if projection_content:
                    if decision_type == "PROJECT_HINT" and target_question_id:
                        # 找到目标题目的位置（通过"第xx页-第xx题"格式匹配）
                        target_question = None
                        for question in questions:
                            # 直接通过id匹配（id格式为"第xx页-第xx题"）
                            question_id = question.get("id", "")
                            if question_id == target_question_id:
                                target_question = question
                                break
                        
                        if target_question and target_question.get("bbox_pixel"):
                            bbox = target_question["bbox_pixel"]
                            x1, y1, x2, y2 = bbox
                            # 在题目右侧显示提示内容，避免覆盖作答区域
                            text_x = x2 + 20  # 题目右侧20像素
                            text_y = y1 + 15  # 题目顶部稍微下移
                            # 确保坐标在图像范围内
                            text_x = max(0, min(text_x, w - 200))
                            text_y = max(0, min(text_y, h - 1))
                            canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                         font_size=18, color=(0, 255, 255))  # 青色
                        else:
                            # 如果找不到题目位置，在画面中央显示
                            text_x = w // 2 - 100
                            text_y = h // 2
                            canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                         font_size=18, color=(0, 255, 255))  # 青色
                    
                    elif decision_type == "CLEAR_PROJECTION":
                        # 清除投影：不显示投影内容，reason会在右侧显示
                        pass
                    
                    elif decision_type == "NO_INTERACTION":
                        # 不交互：不显示投影内容，reason会在右侧显示
                        pass
                    
                    else:
                        # 其他类型的决策，如果有投影内容，在画面中央显示
                        text_x = w // 2 - 100
                        text_y = h // 2
                        canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                     font_size=18, color=(255, 255, 255))  # 白色
            
            # 在右侧侧栏显示状态信息（用于debug，精简显示）
            status_y = 30  # 起始Y位置（距离顶部30像素）
            status_x = divider_x + 10  # 竖线右侧10像素
            line_height = 22  # 每行高度
            text_color = (0, 255, 0)  # 绿色，更清晰
            font_size = 13  # 稍大一点的字体
            
            # 显示决策类型和reason（精简）
            if decision:
                decision_type = decision.get("decision_type")
                reason = decision.get("reason", "")
                
                # 显示决策类型（精简）
                canvas = self.put_text(canvas, decision_type, (status_x, status_y),
                             font_size=font_size, color=text_color)
                status_y += line_height
                
                # 显示reason（如果有，精简显示）
                if reason:
                    # 如果reason太长，截断
                    display_reason = reason[:20] + "..." if len(reason) > 20 else reason
                    canvas = self.put_text(canvas, display_reason, (status_x, status_y),
                                 font_size=font_size, color=text_color)
                    status_y += line_height
            
            # 显示感知报告的关键信息（精简）
            if perception_report:
                is_writing = perception_report.get("is_writing", False)
                time_on_question = perception_report.get("time_on_active_question_seconds", 0)
                
                # 只显示关键状态
                writing_text = "书写中" if is_writing else "空闲"
                canvas = self.put_text(canvas, writing_text, (status_x, status_y),
                             font_size=font_size, color=text_color)
                status_y += line_height
                
                if time_on_question > 0:
                    canvas = self.put_text(canvas, f"{int(time_on_question)}s", (status_x, status_y),
                                 font_size=font_size, color=text_color)
            
            return canvas
        except Exception as e:
            traceback.print_exc()
            # 返回原始帧作为fallback
            return frame.copy() if frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    
    def put_text(self, frame, text, position, font_size=20, color=(255, 255, 255)):
        """
        在图像上绘制中文文字
        
        Args:
            frame: OpenCV图像（BGR格式）
            text: 要绘制的文字
            position: 文字位置 (x, y)
            font_size: 字体大小
            color: 文字颜色（BGR格式）
            
        Returns:
            np.ndarray: 修改后的图像
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            except:
                font = ImageFont.load_default()
        
        # 将BGR颜色转换为RGB（PIL使用RGB）
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def draw_checkmark(self, frame, position, size=30, color=(0, 255, 0), thickness=3):
        """
        在图像上绘制对号图形
        
        Args:
            frame: OpenCV图像（BGR格式）
            position: 对号左下角位置 (x, y)
            size: 对号大小（像素）
            color: 对号颜色（BGR格式，默认绿色）
            thickness: 线条粗细
            
        Returns:
            np.ndarray: 修改后的图像
        """
        x, y = position
        # 对号的路径：从左上到中间，然后到右下
        # 标准的对号形状
        h = size  # 高度
        w = size  # 宽度
        
        # 对号的三个关键点（从左上到中间转折，再到右下）
        point1 = (x, y - h // 3)  # 左上起点
        point2 = (x + w // 3, y)  # 中间转折点
        point3 = (x + w, y - h)  # 右下终点
        
        # 绘制对号的两条线段
        cv2.line(frame, point1, point2, color, thickness)
        cv2.line(frame, point2, point3, color, thickness)
        
        return frame
    
    def run(self):
        """运行主程序"""
        try:
            self.init_camera()
            self.running = True
            
            print("\n=== AI投影学习助手 ===")
            print("按 'q' 键退出\n")
            
            analysis_thread = None
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 更新最新帧
                with self.latest_frame_lock:
                    self.latest_frame = frame.copy()
                
                # 检查是否到了分析时间
                current_time = time.time()
                time_since_last_analysis = current_time - self.last_analysis_time
                
                # 异步分析（如果当前没有分析任务，且距离上次分析完成已超过间隔时间）
                with self.analysis_lock:
                    can_analyze = not self.is_analyzing
                
                if can_analyze and time_since_last_analysis >= self.analysis_interval:
                    def analyze_scene():
                        # 使用最新的帧进行分析
                        with self.latest_frame_lock:
                            process_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                        
                        if process_frame is None:
                            with self.analysis_lock:
                                self.is_analyzing = False
                            return
                        
                        with self.analysis_lock:
                            self.is_analyzing = True
                        
                        try:
                            # 步骤1：感知Agent分析场景
                            # 获取上一次的感知状态
                            with self.data_lock:
                                previous_perception_state = self.perception_states.copy() if self.perception_states else None
                            
                            perception_report = self.perception_agent.analyze_scene(process_frame, previous_perception_state)
                            if perception_report:
                                # 验证感知报告格式是否正确
                                if not isinstance(perception_report, dict):
                                    perception_report = None
                                elif "questions_on_page" in perception_report:
                                    # 验证questions_on_page格式
                                    questions = perception_report.get("questions_on_page", [])
                                    if not isinstance(questions, list):
                                        perception_report = None
                                    else:
                                        # 验证每个题目是否有bbox_pixel
                                        for q in questions:
                                            if not isinstance(q, dict) or "bbox_pixel" not in q or q.get("bbox_pixel") is None:
                                                perception_report = None
                                                break
                                
                                if perception_report:
                                    # 更新全局感知状态（只有格式正确时才更新）
                                    with self.data_lock:
                                        old_state_keys = set(self.perception_states.keys()) if self.perception_states else set()
                                        self.perception_states = perception_report.copy()
                                        new_state_keys = set(self.perception_states.keys())
                                        changed_keys = new_state_keys - old_state_keys if old_state_keys else new_state_keys
                                        
                                        print(f"[LOG] ========== 更新全局感知状态 ==========")
                                        print(f"[LOG] 感知状态字段: {sorted(self.perception_states.keys())}")
                                        if changed_keys:
                                            print(f"[LOG] 新增/变化的字段: {sorted(changed_keys)}")
                                        
                                        # 详细输出感知状态内容
                                        print(f"[LOG] --- 感知状态详情 ---")
                                        print(f"[LOG] timestamp: {self.perception_states.get('timestamp')}")
                                        print(f"[LOG] current_page_id: {self.perception_states.get('current_page_id')}")
                                        print(f"[LOG] active_question_id: {self.perception_states.get('active_question_id')}")
                                        print(f"[LOG] is_writing: {self.perception_states.get('is_writing')}")
                                        print(f"[LOG] is_active_question_completed: {self.perception_states.get('is_active_question_completed')}")
                                        print(f"[LOG] time_on_active_question_seconds: {self.perception_states.get('time_on_active_question_seconds')}")
                                        
                                        if "questions_on_page" in self.perception_states:
                                            questions = self.perception_states['questions_on_page']
                                            print(f"[LOG] 题目数量: {len(questions)}")
                                            for q in questions:
                                                q_id = q.get('id', 'unknown')
                                                q_text = q.get('text', '')[:30] + '...' if len(q.get('text', '')) > 30 else q.get('text', '')
                                                q_bbox = q.get('bbox', [])
                                                q_bbox_pixel = q.get('bbox_pixel', [])
                                                print(f"[LOG]   - {q_id}: text='{q_text}', bbox={q_bbox}, bbox_pixel={q_bbox_pixel}")
                                        
                                        if "user_attempt_content" in self.perception_states:
                                            attempt_content = self.perception_states['user_attempt_content']
                                            print(f"[LOG] 已作答题目: {list(attempt_content.keys())}")
                                            for q_id, content in attempt_content.items():
                                                content_preview = content[:50] + '...' if len(content) > 50 else content
                                                print(f"[LOG]   - {q_id}: {content_preview}")
                                        
                                        print(f"[LOG] ======================================")
                            
                            # 步骤2：推理Agent做出决策（传入图像以便更准确的判断）
                            decision = None
                            if perception_report:
                                # 将main中维护的状态传递给推理Agent
                                with self.data_lock:
                                    # 将main中维护的状态同步到推理Agent
                                    self.reasoning_agent.question_states = self.question_states.copy()
                                
                                decision = self.reasoning_agent.make_decision(perception_report, process_frame)
                                if decision:
                                    # 检查决策是否有变化
                                    decision_type = decision.get("decision_type")
                                    has_changed = False
                                    updated_states = decision.get("updated_question_states", {})
                                    
                                    with self.data_lock:
                                        # 比较决策是否发生变化
                                        old_decision_type = self.decision_states.get("decision_type")
                                        old_target_question_id = self.decision_states.get("target_question_id")
                                        old_projection_content = self.decision_states.get("projection_content")
                                        
                                        new_target_question_id = decision.get("target_question_id")
                                        new_projection_content = decision.get("projection_content")
                                        
                                        # 检查决策类型、目标题目或投影内容是否变化
                                        if (decision_type != old_decision_type or 
                                            new_target_question_id != old_target_question_id or
                                            new_projection_content != old_projection_content):
                                            has_changed = True
                                        
                                        # 检查updated_question_states是否有变化
                                        if updated_states:
                                            # 比较状态变化（检查是否有新增或修改的状态）
                                            # 应该与self.question_states比较，而不是decision_states中的updated_question_states
                                            for question_id, new_state in updated_states.items():
                                                old_state = self.question_states.get(question_id)
                                                # 比较关键字段
                                                if (old_state is None or
                                                    old_state.get("last_action_type") != new_state.get("last_action_type") or
                                                    old_state.get("status") != new_state.get("status") or
                                                    old_state.get("is_correct") != new_state.get("is_correct")):
                                                    has_changed = True
                                                    break
                                        
                                        # 如果决策有变化，更新全局决策状态
                                        if has_changed:
                                            old_decision_type_for_log = self.decision_states.get("decision_type") if self.decision_states else None
                                            old_question_count = len(self.question_states)
                                            
                                            self.decision_states = decision.copy()
                                            
                                            # 更新题目状态（从推理Agent返回的updated_question_states）
                                            if updated_states:
                                                # 合并更新状态，保留未更新的题目状态
                                                for question_id, new_state in updated_states.items():
                                                    old_state = self.question_states.get(question_id)
                                                    
                                                    # 合并状态：保留旧状态中不在新状态中的字段，用新状态覆盖
                                                    if old_state:
                                                        merged_state = old_state.copy()
                                                        merged_state.update(new_state)
                                                        self.question_states[question_id] = merged_state
                                                    else:
                                                        self.question_states[question_id] = new_state.copy()
                                        
                                        # 无论是否有变化，都打印全局推理决策状态信息
                                        print(f"[LOG] ========== 推理Agent返回 - 全局推理决策状态 ==========")
                                        print(f"[LOG] 决策类型: {old_decision_type} -> {decision_type}")
                                        print(f"[LOG] --- 本次决策详情 ---")
                                        print(f"[LOG] decision_type: {decision_type}")
                                        print(f"[LOG] target_question_id: {decision.get('target_question_id')}")
                                        print(f"[LOG] projection_content: {decision.get('projection_content')}")
                                        print(f"[LOG] hint_level: {decision.get('hint_level')}")
                                        print(f"[LOG] reason: {decision.get('reason')}")
                                        
                                        # 显示updated_question_states（如果有）
                                        if updated_states:
                                            print(f"[LOG] --- 本次返回的updated_question_states ---")
                                            for question_id, new_state in updated_states.items():
                                                old_state = self.question_states.get(question_id)
                                                print(f"[LOG]   {question_id}:")
                                                print(f"[LOG]     - last_action_type: {old_state.get('last_action_type') if old_state else None} -> {new_state.get('last_action_type')}")
                                                print(f"[LOG]     - status: {old_state.get('status') if old_state else None} -> {new_state.get('status')}")
                                                print(f"[LOG]     - hint_level: {old_state.get('hint_level') if old_state else None} -> {new_state.get('hint_level')}")
                                                if new_state.get('last_action_time'):
                                                    print(f"[LOG]     - last_action_time: {new_state.get('last_action_time')}")
                                                if new_state.get('is_correct') is not None:
                                                    print(f"[LOG]     - is_correct: {new_state.get('is_correct')}")
                                                if new_state.get('error_analysis'):
                                                    print(f"[LOG]     - error_analysis: {new_state.get('error_analysis')}")
                                                if new_state.get('error_log'):
                                                    print(f"[LOG]     - error_log: {new_state.get('error_log')}")
                                        
                                        # 输出当前全局决策状态
                                        print(f"[LOG] --- 当前全局决策状态 (decision_states) ---")
                                        if self.decision_states:
                                            print(f"[LOG] decision_type: {self.decision_states.get('decision_type')}")
                                            print(f"[LOG] target_question_id: {self.decision_states.get('target_question_id')}")
                                            print(f"[LOG] projection_content: {self.decision_states.get('projection_content')}")
                                            print(f"[LOG] hint_level: {self.decision_states.get('hint_level')}")
                                            print(f"[LOG] reason: {self.decision_states.get('reason')}")
                                        else:
                                            print(f"[LOG] (空)")
                                        
                                        # 输出当前所有题目的全局状态
                                        print(f"[LOG] --- 当前全局所有题目状态 (question_states) ---")
                                        if self.question_states:
                                            for q_id, q_state in self.question_states.items():
                                                print(f"[LOG]   {q_id}:")
                                                print(f"[LOG]     - hint_level: {q_state.get('hint_level')}")
                                                print(f"[LOG]     - last_action_type: {q_state.get('last_action_type')}")
                                                print(f"[LOG]     - status: {q_state.get('status')}")
                                                print(f"[LOG]     - last_action_time: {q_state.get('last_action_time')}")
                                                if q_state.get('is_correct') is not None:
                                                    print(f"[LOG]     - is_correct: {q_state.get('is_correct')}")
                                                if q_state.get('error_analysis'):
                                                    print(f"[LOG]     - error_analysis: {q_state.get('error_analysis')}")
                                        else:
                                            print(f"[LOG] (空)")
                                        
                                        print(f"[LOG] ==========================================")
                            
                            # 更新当前感知报告（用于绘制投影时获取最新感知数据）
                            with self.data_lock:
                                self.current_perception_report = perception_report
                                # 注意：decision_states已经在上面更新了，这里不需要再更新current_decision
                            
                        except Exception as e:
                            traceback.print_exc()
                        finally:
                            self.last_analysis_time = time.time()
                            with self.analysis_lock:
                                self.is_analyzing = False
                    
                    analysis_thread = threading.Thread(target=analyze_scene, daemon=True)
                    analysis_thread.start()
                
                # 获取当前数据（使用全局维护的决策状态）
                try:
                    with self.data_lock:
                        perception_report = self.current_perception_report
                        # 使用全局维护的决策状态，而不是临时的decision
                        decision = self.decision_states.copy() if self.decision_states else None
                    
                    # 绘制投影（基于全局维护的决策状态）
                    display_frame = self.draw_projection(frame, perception_report, decision)
                except Exception as e:
                    traceback.print_exc()
                    display_frame = frame.copy()
                
                # 调整到720P显示
                h, w = display_frame.shape[:2]
                if h != self.display_height or w != self.display_width:
                    display_frame = cv2.resize(display_frame, (self.display_width, self.display_height))
                
                # 显示图像
                cv2.imshow('AI投影学习助手', display_frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('f'):
                    # 切换全屏模式
                    try:
                        current_prop = cv2.getWindowProperty('AI投影学习助手', cv2.WND_PROP_FULLSCREEN)
                        if current_prop == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty('AI投影学习助手', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            self.fullscreen = False
                        else:
                            cv2.setWindowProperty('AI投影学习助手', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            self.fullscreen = True
                    except:
                        if self.fullscreen:
                            cv2.resizeWindow('AI投影学习助手', self.display_width, self.display_height)
                            self.fullscreen = False
                        else:
                            cv2.resizeWindow('AI投影学习助手', self.display_width, self.display_height)
                            self.fullscreen = True
            
        except Exception as e:
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    try:
        assistant = AIProjectionLearningAssistant(camera_index=0)
        assistant.run()
    except ValueError as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()

