"""
HLLM OpenAI Compatible REST API Client

与 OpenAI 官方客户端 API 兼容的 Python 客户端。
可以配合 HLLM 服务使用，也可以替代 OpenAI 官方客户端。
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Generator, Literal

import requests


@dataclass
class ChatMessage:
    """聊天消息"""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatCompletionChoice:
    """对话补全选项"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionResponse:
    """对话补全响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


@dataclass
class CompletionChoice:
    """文本补全选项"""
    text: str
    index: int
    finish_reason: Optional[str] = None


@dataclass
class CompletionResponse:
    """文本补全响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


@dataclass
class StreamDelta:
    """流式响应增量"""
    role: Optional[str] = None
    content: Optional[str] = None


@dataclass
class StreamChoice:
    """流式选项"""
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionStreamResponse:
    """流式对话补全响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]


class ChatCompletionsAPI:
    """对话补全 API（OpenAI 兼容）"""

    def __init__(self, client: "HLLMClient"):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionStreamResponse, None, None]]:
        """
        创建对话补全（与 OpenAI 兼容）

        Args:
            model: 模型 ID
            messages: 消息列表
            max_tokens: 最大生成 token 数
            temperature: 温度系数
            top_p: Top-p 采样
            stream: 是否流式返回

        Returns:
            对话补全响应或流式生成器
        """
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        data.update(kwargs)

        if stream:
            return self._stream_create(data)

        response = self._client._post("/v1/chat/completions", data)
        return self._parse_chat_response(response)

    def _stream_create(
        self,
        data: Dict[str, Any]
    ) -> Generator[ChatCompletionStreamResponse, None, None]:
        """流式创建"""
        response = self._client._post_stream("/v1/chat/completions", data)

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield self._parse_stream_response(chunk_data)
                    except json.JSONDecodeError:
                        continue

    def _parse_chat_response(self, data: Dict) -> ChatCompletionResponse:
        """解析对话响应"""
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(ChatCompletionChoice(
                index=c.get("index", 0),
                message=ChatMessage(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content", "")
                ),
                finish_reason=c.get("finish_reason")
            ))

        return ChatCompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=data.get("usage", {})
        )

    def _parse_stream_response(self, data: Dict) -> ChatCompletionStreamResponse:
        """解析流式响应"""
        choices = []
        for c in data.get("choices", []):
            delta = c.get("delta", {})
            choices.append(StreamChoice(
                index=c.get("index", 0),
                delta=StreamDelta(
                    role=delta.get("role"),
                    content=delta.get("content")
                ),
                finish_reason=c.get("finish_reason")
            ))

        return ChatCompletionStreamResponse(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices
        )


class CompletionsAPI:
    """文本补全 API（OpenAI 兼容）"""

    def __init__(self, client: "HLLMClient"):
        self._client = client

    def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Union[CompletionResponse, Generator[ChatCompletionStreamResponse, None, None]]:
        """
        创建文本补全（与 OpenAI 兼容）

        Args:
            model: 模型 ID
            prompt: 提示文本
            max_tokens: 最大生成 token 数
            temperature: 温度系数
            top_p: Top-p 采样
            stream: 是否流式返回

        Returns:
            文本补全响应或流式生成器
        """
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        data.update(kwargs)

        if stream:
            return self._stream_create(data)

        response = self._client._post("/v1/completions", data)
        return self._parse_completion_response(response)

    def _stream_create(
        self,
        data: Dict[str, Any]
    ) -> Generator[ChatCompletionStreamResponse, None, None]:
        """流式创建"""
        response = self._client._post_stream("/v1/completions", data)

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield self._client.chat.completions._parse_stream_response(chunk_data)
                    except json.JSONDecodeError:
                        continue

    def _parse_completion_response(self, data: Dict) -> CompletionResponse:
        """解析补全响应"""
        choices = []
        for c in data.get("choices", []):
            choices.append(CompletionChoice(
                index=c.get("index", 0),
                text=c.get("text", ""),
                finish_reason=c.get("finish_reason")
            ))

        return CompletionResponse(
            id=data.get("id", ""),
            object=data.get("object", "text_completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=data.get("usage", {})
        )


class ModelsAPI:
    """模型 API"""

    def __init__(self, client: "HLLMClient"):
        self._client = client

    def list(self) -> Dict[str, Any]:
        """获取模型列表"""
        return self._client._get("/v1/models")


class HLLMClient:
    """
    HLLM OpenAI 兼容 REST API 客户端

    与 OpenAI 官方客户端 API 完全兼容。

    示例：
        >>> client = HLLMClient("http://localhost:8000")
        >>> response = client.chat.completions.create(
        ...     model="hllm-model",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0):
        """
        初始化客户端

        Args:
            base_url: API 基础 URL (如 http://localhost:8000)
            api_key: API 密钥（HLLM 不需要，但兼容 OpenAI 格式）
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        # API 子模块（OpenAI 兼容）
        self.chat = type('Chat', (), {'completions': ChatCompletionsAPI(self)})()
        self.completions = CompletionsAPI(self)
        self.models = ModelsAPI(self)

    def _get(self, endpoint: str) -> Dict:
        """GET 请求"""
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: Dict) -> Dict:
        """POST 请求"""
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _post_stream(self, endpoint: str, data: Dict):
        """流式 POST 请求"""
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=data,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response

    def close(self) -> None:
        """关闭客户端会话"""
        self.session.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False
