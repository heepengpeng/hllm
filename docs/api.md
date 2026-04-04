# HLLM OpenAI Compatible API 设计文档

## 概述

HLLM 提供与 OpenAI API 兼容的 REST 接口，支持使用 OpenAI 客户端或任意兼容工具访问。

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

## OpenAI 兼容 API 端点

### 1. 模型列表

**GET** `/v1/models`

获取可用模型列表。

**响应（OpenAI 格式）：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "hllm-model",
      "object": "model",
      "created": 1677610602,
      "owned_by": "hllm"
    }
  ]
}
```

### 2. 对话补全 (Chat Completions)

**POST** `/v1/chat/completions`

与 OpenAI 的 chat.completions.create 兼容。

**请求体：**
```json
{
  "model": "hllm-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**响应（非流式）：**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "hllm-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

**流式响应 (SSE)：**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"hllm-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"hllm-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"hllm-model","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"hllm-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 3. 文本补全 (Completions)

**POST** `/v1/completions`

与 OpenAI 的 completions.create 兼容（传统接口）。

**请求体：**
```json
{
  "model": "hllm-model",
  "prompt": "Once upon a time",
  "max_tokens": 50,
  "temperature": 0.7,
  "stream": false
}
```

**响应：**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1677652288,
  "model": "hllm-model",
  "choices": [
    {
      "text": ", there was a brave knight who...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 12,
    "total_tokens": 16
  }
}
```

### 4. 嵌入 (Embeddings)

**POST** `/v1/embeddings`

**请求体：**
```json
{
  "model": "hllm-model",
  "input": "The food was delicious and the waiter..."
}
```

**响应：**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023064255, -0.009327292, ...],
      "index": 0
    }
  ],
  "model": "hllm-model",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

## 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| model | string | 是 | - | 模型 ID（固定为 "hllm-model"） |
| messages | array | 是* | - | 对话消息列表（chat completions） |
| prompt | string/array | 是* | - | 提示文本（completions） |
| max_tokens | int | 否 | 100 | 最大生成 token 数 |
| temperature | float | 否 | 0.7 | 温度系数 (0-2) |
| top_p | float | 否 | 0.9 | Top-p 采样 |
| top_k | int | 否 | 50 | Top-k 采样 |
| stream | bool | 否 | false | 是否流式返回 |
| stop | string/array | 否 | null | 停止序列 |

*chat.completions 用 messages，completions 用 prompt

## 使用 OpenAI 官方客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # HLLM 不验证 API key
)

# 对话
response = client.chat.completions.create(
    model="hllm-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    max_tokens=100
)
print(response.choices[0].message.content)

# 流式
for chunk in client.chat.completions.create(
    model="hllm-model",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 使用 HLLM 内置客户端

```python
from hllm.client import HLLMClient

client = HLLMClient(base_url="http://localhost:8000")

# 对话（OpenAI 兼容格式）
response = client.chat.completions.create(
    model="hllm-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# 文本补全
response = client.completions.create(
    model="hllm-model",
    prompt="Once upon a time",
    max_tokens=50
)
print(response.choices[0].text)
```

## curl 示例

```bash
# 模型列表
curl http://localhost:8000/v1/models

# 对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hllm-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# 流式对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hllm-model",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true
  }'
```

## 错误处理

OpenAI 兼容错误格式：

```json
{
  "error": {
    "message": "Invalid request",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

**HTTP 状态码：**
| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未授权（API key 无效） |
| 404 | 模型未找到 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |
