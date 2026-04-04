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

## REST API 服务

### 安装 API 依赖

```bash
pip install light-llm-hp[api]
```

### 启动服务

```python
from hllm import HLLM
from hllm.server import Server

model = HLLM(model_path="./TinyLlama-1.1B-Chat-v1.0", device="cpu")
server = Server(model, host="0.0.0.0", port=8000)
server.start()
```

或通过命令行：

```bash
python -m hllm.server --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000
```

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/models` | GET | 模型信息 |
| `/generate` | POST | 文本生成 |
| `/chat` | POST | 对话生成 |
| `/generate/stream` | POST | 流式生成 (SSE) |

### 使用客户端

```python
from hllm.client import HLLMClient

client = HLLMClient("http://localhost:8000")

# 生成文本
response = client.generate("Hello, how are you?")
print(response.text)

# 对话
response = client.chat([
    {"role": "user", "content": "What is AI?"}
])
print(response.message.content)

# 流式生成
for chunk in client.generate_stream("Tell me a story"):
    print(chunk.token, end="", flush=True)
```

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