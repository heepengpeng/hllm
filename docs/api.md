# HLLM REST API 设计文档

## 概述

HLLM REST API 提供基于 HTTP 的 LLM 推理服务，支持文本生成、模型管理等功能。

## 服务端启动

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

## API 端点

### 1. 健康检查

**GET** `/health`

检查服务是否正常运行。

**响应：**
```json
{
  "status": "ok",
  "model": "TinyLlama-1.1B-Chat-v1.0",
  "device": "cpu"
}
```

### 2. 文本生成

**POST** `/generate`

生成文本内容。

**请求体：**
```json
{
  "prompt": "Hello, how are you?",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stream": false
}
```

**参数说明：**
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| prompt | string | 是 | - | 输入提示文本 |
| max_new_tokens | int | 否 | 100 | 最大生成 token 数 |
| temperature | float | 否 | 0.7 | 温度系数 |
| top_p | float | 否 | 0.9 | Top-p 采样 |
| top_k | int | 否 | 50 | Top-k 采样 |
| stream | bool | 否 | false | 是否流式返回 |

**响应（非流式）：**
```json
{
  "text": "I'm doing well, thank you! How can I help you today?",
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 12,
    "total_tokens": 17
  }
}
```

### 3. 对话生成

**POST** `/chat`

支持多轮对话的聊天接口。

**请求体：**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
  ],
  "max_new_tokens": 200,
  "temperature": 0.7,
  "stream": false
}
```

**响应：**
```json
{
  "message": {
    "role": "assistant",
    "content": "Python is a high-level, interpreted programming language..."
  },
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 45,
    "total_tokens": 60
  }
}
```

### 4. 模型信息

**GET** `/models`

获取当前加载的模型信息。

**响应：**
```json
{
  "model": "TinyLlama-1.1B-Chat-v1.0",
  "device": "cpu",
  "max_length": 2048,
  "vocab_size": 32000
}
```

### 5. 流式生成

**POST** `/generate/stream`

SSE (Server-Sent Events) 流式生成。

**请求体：**
```json
{
  "prompt": "Tell me a story",
  "max_new_tokens": 500
}
```

**响应格式：**
```
data: {"token": "Once", "index": 0}

data: {"token": " upon", "index": 1}

data: {"token": " a", "index": 2}

data: {"token": " time", "index": 3}

data: [DONE]
```

## 错误处理

所有错误返回 JSON 格式：

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: prompt"
  }
}
```

**错误码：**
| 状态码 | 错误码 | 说明 |
|--------|--------|------|
| 400 | INVALID_REQUEST | 请求参数错误 |
| 404 | MODEL_NOT_FOUND | 模型未找到 |
| 500 | INTERNAL_ERROR | 服务器内部错误 |
| 503 | SERVICE_UNAVAILABLE | 服务暂不可用 |

## Python 客户端使用

```python
from hllm.client import HLLMClient

client = HLLMClient(base_url="http://localhost:8000")

# 生成文本
response = client.generate(
    prompt="Hello, how are you?",
    max_new_tokens=100
)
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

## curl 示例

```bash
# 健康检查
curl http://localhost:8000/health

# 生成文本
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 50}'

# 对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_new_tokens": 100
  }'
```
