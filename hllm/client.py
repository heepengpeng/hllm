"""
HLLM REST API Client

用于与 HLLM REST API 服务通信的 Python 客户端。
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Generator
import json

import requests


@dataclass
class GenerateResponse:
    """文本生成响应"""
    text: str
    usage: Dict[str, int]


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str


@dataclass
class ChatResponse:
    """对话响应"""
    message: ChatMessage
    usage: Dict[str, int]


@dataclass
class StreamChunk:
    """流式生成块"""
    token: str
    index: int


class HLLMClient:
    """
    HLLM REST API 客户端

    示例：
        >>> client = HLLMClient("http://localhost:8000")
        >>> response = client.generate("Hello, how are you?")
        >>> print(response.text)
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        初始化客户端

        Args:
            base_url: API 基础 URL (如 http://localhost:8000)
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            服务状态信息
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_models(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型配置信息
        """
        response = self.session.get(
            f"{self.base_url}/models",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False
    ) -> GenerateResponse:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数
            top_p: Top-p 采样
            top_k: Top-k 采样
            stream: 是否流式返回（客户端流式请使用 generate_stream）

        Returns:
            生成响应
        """
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream
        }

        response = self.session.post(
            f"{self.base_url}/generate",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        return GenerateResponse(
            text=result["text"],
            usage=result["usage"]
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Generator[StreamChunk, None, None]:
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数
            top_p: Top-p 采样
            top_k: Top-k 采样

        Yields:
            生成的 token 块
        """
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        response = self.session.post(
            f"{self.base_url}/generate/stream",
            json=data,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield StreamChunk(
                            token=chunk_data["token"],
                            index=chunk_data["index"]
                        )
                    except json.JSONDecodeError:
                        continue

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> ChatResponse:
        """
        对话生成

        Args:
            messages: 消息列表，每个消息包含 role 和 content
            max_new_tokens: 最大生成 token 数
            temperature: 温度系数
            top_p: Top-p 采样
            top_k: Top-k 采样

        Returns:
            对话响应
        """
        data = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        response = self.session.post(
            f"{self.base_url}/chat",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        message_data = result["message"]
        return ChatResponse(
            message=ChatMessage(
                role=message_data["role"],
                content=message_data["content"]
            ),
            usage=result["usage"]
        )

    def chat_simple(
        self,
        message: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        简化的单轮对话

        Args:
            message: 用户消息
            system: 系统提示（可选）
            **kwargs: 其他参数传递给 chat()

        Returns:
            助手的回复内容
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        response = self.chat(messages, **kwargs)
        return response.message.content

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
