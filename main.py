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
        
        # API调用间隔控制（3秒，从上次检测完成开始计算）
        self.last_api_call_time = 0
        self.api_call_interval = 3.0  # 3秒间隔
        
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
        
        cv2.namedWindow('数学题目识别', cv2.WINDOW_NORMAL)
    
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
        
        # 绘制每个题目的作答区域框
        for i, problem in enumerate(problems):
            x = max(0, min(problem.get("x", 0), w - 1))
            y = max(0, min(problem.get("y", 0), h - 1))
            width = max(10, min(problem.get("width", 0), w - x))
            height = max(10, min(problem.get("height", 0), h - y))
            
            # 获取作答状态和错误原因
            answer_status = problem.get("answer_status", "空白")
            error_reason = problem.get("error_reason", "")
            
            # 文字颜色统一使用最亮的白色
            text_color = (255, 255, 255)  # 最亮的白色
            
            # 在作答区域显示文字（左上角显示）
            # 构建显示文本：始终显示 answer_status，如果错误则加上错误原因
            if answer_status == "错误" and error_reason:
                # 如果错误，显示状态和错误原因
                display_text = f"{answer_status}：{error_reason}"
            elif answer_status == "空白":
                # 如果空白，显示"请开始作答"
                display_text = "请开始作答"
            else:
                # 如果正确，显示状态
                display_text = answer_status
            
            # 计算文字位置（左上角）
            text_size = 20  # 增大字体，确保可见
            text_x = x  # 水平位置：x
            text_y = y  # 垂直位置：y
            # put_text 返回修改后的画布，需要接收返回值
            canvas = self.put_text(canvas, display_text, (text_x, text_y), font_size=text_size, color=text_color)
        
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
                
                # 检查是否到了API调用时间（从上次检测完成开始计算，每隔3秒）
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                
                # 实时检测（如果当前没有检测任务，且距离上次检测完成已超过3秒）
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
                        
                        # 检测完成后，更新API调用时间（从检测完成开始计算3秒间隔）
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
            print(f"运行错误: {e}")
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
        print("请在 qwen_client.py 文件顶部设置 DASHSCOPE_API_KEY")


if __name__ == "__main__":
    main()
