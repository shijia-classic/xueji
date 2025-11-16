#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学题目识别工具
自动识别画面中的数学题目并绘制边界框
"""

import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageDraw, ImageFont
from qwen_client import QwenClient


class MathProblemDetector:
    def __init__(self, camera_index=0):
        """初始化数学题目识别工具"""
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        
        # 初始化Qwen API客户端
        self.qwen_client = QwenClient()
        
        # 当前数学题目识别信息
        self.current_math_problems = None
        self.math_problems_lock = threading.Lock()
        
        # 检测状态
        self.is_detecting_math = False
        self.math_detection_lock = threading.Lock()
        
        # API调用间隔控制（500ms，从上次检测完成开始计算）
        self.last_api_call_time = 0
        self.api_call_interval = 0.5  # 500ms间隔
        
        # 手指检测间隔控制（200ms，降低API调用频率）
        self.last_finger_detection_time = 0
        self.finger_detection_interval = 0.2  # 200ms间隔
        self.finger_detection_lock = threading.Lock()
        self.is_detecting_finger = False
        
        # 最新帧缓存（确保检测使用最新画面）
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        
        # 720P显示尺寸
        self.display_width = 1280
        self.display_height = 720
        
        # 全屏标志
        self.fullscreen = False
        
        # 手指检测状态（保留用于未来扩展）
        self.finger_position = None  # 当前手指位置 (x, y)
    
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
        
        cv2.namedWindow('数学题目识别', cv2.WINDOW_NORMAL)
    
    def detect_finger_tip(self, frame):
        """
        检测手指指尖位置（使用大模型API）
        
        Args:
            frame: OpenCV图像帧
            
        Returns:
            tuple: (x, y) 手指位置（像素坐标），如果未检测到返回None
        """
        # 使用Qwen API检测手指指尖（返回归一化坐标）
        normalized_pos = self.qwen_client.detect_finger_tip(frame)
        
        if normalized_pos is None:
            return None
        
        # 转换为像素坐标
        h, w = frame.shape[:2]
        x = int(normalized_pos[0] * w)
        y = int(normalized_pos[1] * h)
        
        return (x, y)
    
    def is_finger_near_question_mark(self, finger_pos, question_mark_pos, radius=50):
        """
        检查手指是否在问号附近
        
        Args:
            finger_pos: 手指位置 (x, y)
            question_mark_pos: 问号位置 (x, y, radius)
            radius: 检测半径（像素）
            
        Returns:
            bool: 手指是否在问号附近
        """
        if finger_pos is None or question_mark_pos is None:
            return False
        
        fx, fy = finger_pos
        qx, qy, qr = question_mark_pos
        
        # 计算距离
        distance = np.sqrt((fx - qx) ** 2 + (fy - qy) ** 2)
        return distance <= radius
    
    def draw_math_problems(self, frame, math_problems_info):
        """在全黑背景上绘制数学题目框"""
        # 创建全黑画布
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 绘制竖线分界线（在宽度10%处）
        divider_x = int(w * 0.1)
        cv2.line(canvas, (divider_x, 0), (divider_x, h), (128, 128, 128), 2)
        
        # 绘制右侧边界竖线
        right_edge = w - 1
        cv2.line(canvas, (right_edge, 0), (right_edge, h), (128, 128, 128), 2)
        
        if not math_problems_info or not math_problems_info.get("found"):
            return canvas
        
        problems = math_problems_info.get("problems", [])
        if not problems:
            return canvas
        
        # 当前时间
        current_time = time.time()
        
        # 检测手指位置（异步，避免阻塞）
        # 如果距离上次检测已超过间隔时间，且当前没有检测任务，则启动新的检测
        time_since_last_finger_detection = current_time - self.last_finger_detection_time
        with self.finger_detection_lock:
            can_detect_finger = not self.is_detecting_finger
        
        if can_detect_finger and time_since_last_finger_detection >= self.finger_detection_interval:
            # 使用最新帧进行手指检测（异步）
            def detect_finger_async():
                with self.latest_frame_lock:
                    process_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                
                if process_frame is not None:
                    with self.finger_detection_lock:
                        self.is_detecting_finger = True
                    
                    finger_pos_normalized = self.qwen_client.detect_finger_tip(process_frame)
                    
                    if finger_pos_normalized is not None:
                        # 转换为像素坐标
                        h, w = process_frame.shape[:2]
                        x = int(finger_pos_normalized[0] * w)
                        y = int(finger_pos_normalized[1] * h)
                        self.finger_position = (x, y)
                    else:
                        self.finger_position = None
                    
                    self.last_finger_detection_time = time.time()
                    
                    with self.finger_detection_lock:
                        self.is_detecting_finger = False
            
            finger_detection_thread = threading.Thread(target=detect_finger_async, daemon=True)
            finger_detection_thread.start()
        
        # 绘制每个题目的作答区域框
        for i, problem in enumerate(problems):
            x = max(0, min(problem.get("x", 0), w - 1))
            y = max(0, min(problem.get("y", 0), h - 1))
            width = max(10, min(problem.get("width", 0), w - x))
            height = max(10, min(problem.get("height", 0), h - y))
            
            # 获取作答状态和错误原因
            answer_status = problem.get("answer_status", "空白")
            error_reason = problem.get("error_reason", "")
            
            # 根据作答状态显示不同内容
            if answer_status == "空白":
                # 空白状态：显示"请作答"（白色）
                display_text = "请作答"
                text_color = (255, 255, 255)  # 白色
            elif answer_status == "正确":
                # 已作答且正确：显示"正确"（绿色）
                display_text = "正确"
                text_color = (0, 255, 0)  # 绿色 (BGR格式)
            elif answer_status == "错误":
                # 已作答但错误：显示"错误"和错误原因（白色）
                if error_reason:
                    error_reason_short = error_reason[:15] if len(error_reason) > 15 else error_reason
                    display_text = f"错误：{error_reason_short}"
                else:
                    display_text = "错误"
                text_color = (255, 255, 255)  # 白色
            else:
                # 默认情况
                display_text = "请作答"
                text_color = (255, 255, 255)  # 白色
            
            # 在作答区域显示文本
            text_size = 20
            text_x = x
            text_y = y
            canvas = self.put_text(canvas, display_text, (text_x, text_y), 
                                 font_size=text_size, color=text_color)
        
        return canvas
    
    def put_text(self, frame, text, position, font_size=20, color=(255, 255, 255)):
        """在图像上绘制中文文字
        Args:
            frame: OpenCV图像（BGR格式）
            text: 要绘制的文字
            position: 文字位置 (x, y)
            font_size: 字体大小
            color: 文字颜色（BGR格式，如 (0, 255, 0) 表示绿色）
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
    
    def run(self):
        """运行主程序"""
        try:
            self.init_camera()
            self.running = True
            
            print("\n=== 数学题目识别工具 ===")
            print("按 'q' 键退出\n")
            
            math_detection_thread = None
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 更新最新帧（确保检测使用最新画面）
                with self.latest_frame_lock:
                    self.latest_frame = frame.copy()
                
                # 检查是否到了API调用时间（从上次检测完成开始计算，每隔500ms）
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                
                # 实时检测（如果当前没有检测任务，且距离上次检测完成已超过500ms）
                with self.math_detection_lock:
                    can_detect = not self.is_detecting_math  # 如果不在检测中，可以检测
                
                if (can_detect or math_detection_thread is None or not math_detection_thread.is_alive()) and \
                   time_since_last_call >= self.api_call_interval:
                    
                    def detect_math():
                        # 使用最新的帧进行检测（确保使用最新画面）
                        with self.latest_frame_lock:
                            process_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                        
                        if process_frame is None:
                            with self.math_detection_lock:
                                self.is_detecting_math = False  # 检测完成
                            return
                        
                        math_info = self.qwen_client.detect_math_problems(process_frame)
                        
                        with self.math_problems_lock:
                            self.current_math_problems = math_info if math_info else {"found": False, "problems": []}
                        
                        # 检测完成后，更新API调用时间（从检测完成开始计算500ms间隔）
                        self.last_api_call_time = time.time()
                        
                        with self.math_detection_lock:
                            self.is_detecting_math = False  # 检测完成，可以开始下一次检测
                    
                    with self.math_detection_lock:
                        self.is_detecting_math = True  # 标记为正在检测
                    
                    math_detection_thread = threading.Thread(target=detect_math, daemon=True)
                    math_detection_thread.start()
                
                # 获取当前检测结果
                with self.math_problems_lock:
                    math_problems_info = self.current_math_problems
                
                # 绘制题目框
                display_frame = self.draw_math_problems(frame, math_problems_info)
                
                # 调整到720P显示
                h, w = display_frame.shape[:2]
                if h != self.display_height or w != self.display_width:
                    display_frame = cv2.resize(display_frame, (self.display_width, self.display_height))
                
                # 显示状态（在左侧10%区域内）
                status_x = 20
                status_y = self.display_height - 30
                
                if not self.is_detecting_math:
                    status = "冷却中..."
                else:
                    status = "AI分析中..."
                
                display_frame = self.put_text(display_frame, status, (status_x, status_y),
                                             font_size=16, color=(0, 255, 0))
                
                # 显示图像
                cv2.imshow('数学题目识别', display_frame)
                
                # 按键控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('f'):
                    # 切换全屏模式
                    try:
                        current_prop = cv2.getWindowProperty('数学题目识别', cv2.WND_PROP_FULLSCREEN)
                        if current_prop == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty('数学题目识别', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            self.fullscreen = False
                        else:
                            cv2.setWindowProperty('数学题目识别', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            self.fullscreen = True
                    except:
                        # 如果切换失败，手动调整窗口大小
                        if self.fullscreen:
                            cv2.resizeWindow('数学题目识别', self.display_width, self.display_height)
                            self.fullscreen = False
                        else:
                            cv2.resizeWindow('数学题目识别', self.display_width, self.display_height)
                            self.fullscreen = True
            
        except Exception as e:
            print(f"运行错误")
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
        detector = MathProblemDetector(camera_index=0)
        detector.run()
    except ValueError as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
