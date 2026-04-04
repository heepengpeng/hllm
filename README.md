# light-llm-hp - 轻量级 LLM 推理框架

在 CPU 上运行的简化推理框架，支持 REST API 服务。

## 快速开始

```python
from hllm import HLLM

# 初始化模型
model = HLLM(model_path="microsoft/Phi-3-mini-4k-instruct", device="cpu")

# 生成文本
result = model.generate("Write a short story about a robot.")
print(result)
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