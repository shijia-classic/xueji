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
        
        # 当前感知报告和决策结果
        self.current_perception_report = None
        self.current_decision = None
        self.data_lock = threading.Lock()
        
        # 推理Agent的反馈（传递给下一次感知）
        self.reasoning_feedback = None
        
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
            decision: 推理决策结果
            
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
                
                # 对于CHECK_ANSWER的情况，处理投影内容
                if decision_type == "CHECK_ANSWER":
                    is_correct = decision.get("is_correct", True)
                    if is_correct:
                        # 答对了，使用特殊标记表示需要绘制对号
                        projection_content = "__DRAW_CHECKMARK__"
                    else:
                        # 答错了，使用error_analysis作为投影内容
                        error_analysis = decision.get("error_analysis", "")
                        if error_analysis:
                            projection_content = error_analysis
                
                # 显示投影内容（如果有）
                if projection_content:
                    if decision_type == "PROJECT_HINT" and target_question_id:
                        # 找到目标题目的位置（通过"第xx题"格式匹配）
                        target_question = None
                        for question in questions:
                            # 直接通过id匹配（id格式为"第xx题"）
                            question_id = question.get("id", "")
                            if question_id == target_question_id:
                                target_question = question
                                break
                        
                        if target_question and target_question.get("bbox_pixel"):
                            bbox = target_question["bbox_pixel"]
                            x1, y1, x2, y2 = bbox
                            # 在题目下方显示提示内容
                            text_x = x1
                            text_y = y2 + 20
                            canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                         font_size=18, color=(0, 255, 255))  # 青色
                        else:
                            # 如果找不到题目位置，在画面中央显示
                            text_x = w // 2 - 100
                            text_y = h // 2
                            canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                         font_size=18, color=(0, 255, 255))  # 青色
                    
                    elif decision_type == "CHECK_ANSWER" and target_question_id:
                        # 找到目标题目的位置（通过"第xx题"格式匹配）
                        target_question = None
                        for question in questions:
                            # 直接通过id匹配（id格式为"第xx题"）
                            question_id = question.get("id", "")
                            if question_id == target_question_id:
                                target_question = question
                                break
                        
                        if target_question and target_question.get("bbox_pixel"):
                            bbox = target_question["bbox_pixel"]
                            x1, y1, x2, y2 = bbox
                            
                            # 判断对错，显示不同内容
                            is_correct = decision.get("is_correct")
                            if is_correct:
                                # 答对了，绘制对号图形
                                checkmark_x = x1
                                checkmark_y = y2 + 30
                                canvas = self.draw_checkmark(canvas, (checkmark_x, checkmark_y), size=30, color=(0, 255, 0))
                            else:
                                # 答错了，显示错误分析文字
                                text_color = (255, 255, 255)  # 白色
                                text_x = x1
                                text_y = y2 + 20
                                canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                             font_size=18, color=text_color)
                        else:
                            # 如果找不到题目位置，在画面中央显示
                            is_correct = decision.get("is_correct")
                            if is_correct:
                                # 答对了，绘制对号图形
                                checkmark_x = w // 2 - 15
                                checkmark_y = h // 2
                                canvas = self.draw_checkmark(canvas, (checkmark_x, checkmark_y), size=30, color=(0, 255, 0))
                            else:
                                # 答错了，显示错误分析文字
                                text_color = (255, 255, 255)  # 白色
                                text_x = w // 2 - 100
                                text_y = h // 2
                                canvas = self.put_text(canvas, projection_content, (text_x, text_y),
                                         font_size=18, color=text_color)
                    
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
            print(f"[ERROR] draw_projection出错: {e}")
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
                            print(f"[LOG] 开始分析场景，时间: {time.time()}")
                            
                            # 获取上一次的推理反馈
                            with self.data_lock:
                                last_feedback = self.reasoning_feedback
                            print(f"[LOG] 获取推理反馈: {last_feedback is not None}")
                            
                            # 步骤1：感知Agent分析场景（传入推理Agent的反馈）
                            print("[LOG] 调用感知Agent...")
                            perception_report = self.perception_agent.analyze_scene(process_frame, last_feedback)
                            print(f"[LOG] 感知Agent返回: {perception_report is not None}")
                            if perception_report:
                                print(f"[LOG] 感知报告 - 活跃题目: {perception_report.get('active_question_id')}, 书写中: {perception_report.get('is_writing')}")
                            
                            # 步骤2：推理Agent做出决策（传入图像以便更准确的判断）
                            decision = None
                            if perception_report:
                                print("[LOG] 调用推理Agent...")
                                decision = self.reasoning_agent.make_decision(perception_report, process_frame)
                                print(f"[LOG] 推理Agent返回: {decision is not None}")
                                if decision:
                                    print(f"[LOG] 决策类型: {decision.get('decision_type')}")
                                
                                # 提取推理Agent的反馈信息（用于下一次感知）
                                if decision and "feedback_to_perception" in decision:
                                    feedback = decision.get("feedback_to_perception")
                                    if feedback:
                                        # 构建反馈信息，包含题目状态
                                        updated_states = decision.get("updated_question_states", {})
                                        reasoning_feedback = {
                                            "feedback_to_perception": feedback,
                                            "updated_question_states": updated_states,
                                            "last_decision_type": decision.get("decision_type"),
                                            "last_target_question_id": decision.get("target_question_id")
                                        }
                                        print(f"[LOG] 构建推理反馈成功")
                                    else:
                                        reasoning_feedback = None
                                else:
                                    reasoning_feedback = None
                            else:
                                reasoning_feedback = None
                                print("[LOG] 感知报告为空，跳过推理Agent")
                            
                            # 更新当前数据
                            print("[LOG] 更新当前数据...")
                            with self.data_lock:
                                self.current_perception_report = perception_report
                                self.current_decision = decision
                                if reasoning_feedback:
                                    self.reasoning_feedback = reasoning_feedback
                            print("[LOG] 分析完成")
                            
                        except Exception as e:
                            print(f"[ERROR] 分析过程出错: {e}")
                            print(f"[ERROR] 错误详情:")
                            traceback.print_exc()
                        finally:
                            self.last_analysis_time = time.time()
                            with self.analysis_lock:
                                self.is_analyzing = False
                    
                    analysis_thread = threading.Thread(target=analyze_scene, daemon=True)
                    analysis_thread.start()
                
                # 获取当前数据
                try:
                    with self.data_lock:
                        perception_report = self.current_perception_report
                        decision = self.current_decision
                    
                    # 绘制投影
                    display_frame = self.draw_projection(frame, perception_report, decision)
                except Exception as e:
                    print(f"[ERROR] 绘制投影出错: {e}")
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
            print(f"[ERROR] 运行错误: {e}")
            traceback.print_exc()
        finally:
            print("[LOG] 清理资源...")
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

