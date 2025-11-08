# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 配置 API Key

创建 `.env` 文件（在项目根目录），添加：

```bash
DASHSCOPE_API_KEY=你的API Key
```

**重要**：请使用你的真实 API Key 替换 `你的API Key`

获取 API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key

## 3. 验证配置

运行测试脚本检查配置：

```bash
python test_env.py
```

如果看到 `✓ 配置检查通过！可以运行 main.py`，说明配置成功。

## 4. 运行程序

```bash
python main.py
```

## 常见问题

### Q: 测试脚本显示 "未安装 python-dotenv"
A: 运行 `pip install python-dotenv` 安装依赖

### Q: 测试脚本显示 "DASHSCOPE_API_KEY: 未设置"
A: 检查 `.env` 文件是否存在，格式是否正确（不要有引号，不要有多余空格）

### Q: 运行 main.py 时提示 API Key 错误
A: 
1. 检查 `.env` 文件中的 API Key 是否正确
2. 确认 API Key 是否有效（可能已过期或被撤销）
3. 运行 `python test_env.py` 验证环境变量是否正确加载

## 安全提示

- ✅ `.env` 文件已添加到 `.gitignore`，不会被提交到 Git
- ✅ 不要在代码中硬编码 API Key
- ✅ 如果 API Key 已暴露，立即在控制台撤销并创建新的

