#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试环境变量配置脚本
用于验证 API Key 是否正确配置
"""

import os
import sys

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ 成功加载 .env 文件")
except ImportError:
    print("⚠️  未安装 python-dotenv，将只使用系统环境变量")
    print("   建议运行: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  加载 .env 文件时出错: {e}")

# 检查 API Key
api_key = os.getenv("DASHSCOPE_API_KEY", "")

print("\n" + "=" * 50)
print("环境变量检查")
print("=" * 50)

if api_key:
    # 只显示前8个字符和后4个字符，中间用*代替
    masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else "*" * len(api_key)
    print(f"✓ DASHSCOPE_API_KEY: {masked_key}")
    print(f"  长度: {len(api_key)} 字符")
    
    # 验证格式（通常以 sk- 开头）
    if api_key.startswith("sk-"):
        print("✓ API Key 格式正确（以 'sk-' 开头）")
    else:
        print("⚠️  API Key 格式可能不正确（通常以 'sk-' 开头）")
    
    print("\n✓ 配置检查通过！可以运行 main.py")
    sys.exit(0)
else:
    print("✗ DASHSCOPE_API_KEY: 未设置")
    print("\n请通过以下方式之一设置 API Key：")
    print("\n方法一：创建 .env 文件")
    print("  1. 在项目根目录创建 .env 文件")
    print("  2. 添加内容：DASHSCOPE_API_KEY=your-api-key-here")
    print("\n方法二：设置系统环境变量")
    print("  macOS/Linux: export DASHSCOPE_API_KEY=your-api-key-here")
    print("  Windows:     set DASHSCOPE_API_KEY=your-api-key-here")
    print("\n获取API Key: https://www.alibabacloud.com/help/zh/model-studio/get-api-key")
    sys.exit(1)

