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
    print("⚠️  未安装 python-dotenv")
except Exception as e:
    print(f"⚠️  加载 .env 失败: {e}")

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
    print("✗ API Key未设置")
    print("\n请创建 .env 文件并添加：DASHSCOPE_API_KEY=your-api-key-here")
    print("或设置环境变量：export DASHSCOPE_API_KEY=your-api-key-here")
    sys.exit(1)

