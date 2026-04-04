# light-llm-hp - 轻量级 LLM 推理框架

[![PyPI version](https://badge.fury.io/py/light-llm-hp.svg)](https://badge.fury.io/py/light-llm-hp)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://pypi.org/project/light-llm-hp/)
[![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](docs/coverage_report.md)
[![CI](https://github.com/heepengpeng/hllm/actions/workflows/ci.yml/badge.svg)](https://github.com/heepengpeng/hllm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

在 CPU 上运行的简化推理框架，支持 REST API 服务。

**🚀 Apple Silicon 优化**: 支持 MLX 后端，比 PyTorch MPS 快 2-5 倍

## 快速开始

```python
from hllm import HLLM

# 自动选择最佳后端 (Apple Silicon 上自动使用 MLX)
model = HLLM(model_path="microsoft/Phi-3-mini-4k-instruct")

# 生成文本
result = model.generate("Write a short story about a robot.")
print(result)
```

### Apple Silicon 优化 (MLX)

在 M1/M2/M3 Mac 上，使用 MLX 后端可获得最佳性能：

```bash
# 安装 MLX 支持
pip install light-llm-hp[mlx]
```

```python
from hllm import HLLM

# 显式使用 MLX 后端 (推荐)
model = HLLM(model_path="mlx-community/Llama-3.2-1B-Instruct-4bit", backend="mlx")

# 或使用 PyTorch MPS
model = HLLM(model_path="microsoft/Phi-3-mini-4k-instruct", backend="pytorch", device="mps")

# 查看后端信息
print(model.get_info())
# {'name': 'mlx', 'device': 'mlx', ...}
```

### 性能对比

在 M1 MacBook Pro 上的典型性能 (Llama-3.2-1B, 100 tokens):

| 后端 | 首 token 延迟 | 吞吐量 | 内存占用 |
|------|--------------|--------|----------|
| MLX | ~50ms | ~45 tok/s | ~800MB |
| PyTorch MPS | ~150ms | ~15 tok/s | ~1200MB |
| PyTorch CPU | ~500ms | ~5 tok/s | ~1200MB |

运行基准测试：
```bash
python examples/benchmark.py
```

## REST API 服务 (OpenAI 兼容)

### 安装 API 依赖

```bash
pip install light-llm-hp[api]
```

### 启动服务

```bash
python -m hllm.server --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000
```

### 使用 OpenAI 官方客户端

```python
import httpx
from openai import OpenAI

# 禁用代理避免 502 错误
http_client = httpx.Client(trust_env=False)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    http_client=http_client
)

# 对话
response = client.chat.completions.create(
    model="hllm-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

完整示例：[examples/test_openai_client.py](examples/test_openai_client.py)

### OpenAI 兼容端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | 对话补全 (支持流式) |
| `/v1/completions` | POST | 文本补全 (支持流式) |

详细 API 文档见 [docs/api.md](docs/api.md)。

## 目录结构

```
hllm/
├── hllm/              # 核心模块
│   ├── __init__.py
│   ├── model.py       # 模型加载与推理
│   ├── tokenizer.py   # 分词器封装
│   ├── generate.py    # 生成逻辑
│   ├── server.py      # REST API 服务端
│   └── client.py      # REST API 客户端
├── tests/             # 测试
├── examples/          # 示例
└── docs/              # 文档
```