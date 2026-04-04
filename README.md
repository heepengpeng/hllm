# hllm - 轻量级 LLM 推理框架

在 CPU 上运行的简化推理框架

## 快速开始

```python
from hllm import HLLM

# 初始化模型
model = HLLM(model_path="microsoft/Phi-3-mini-4k-instruct", device="cpu")

# 生成文本
result = model.generate("Write a short story about a robot.")
print(result)
```

## 目录结构

```
hllm/
├── hllm/              # 核心模块
│   ├── __init__.py
│   ├── model.py       # 模型加载与推理
│   ├── tokenizer.py   # 分词器封装
│   └── generate.py    # 生成逻辑
├── tests/             # 测试
└── examples/          # 示例
```