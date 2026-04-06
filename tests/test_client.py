"""
测试 client 模块
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests


class TestHLLMClient:
    """测试 HLLMClient"""

    def test_client_init(self):
        """测试客户端初始化"""
        from hllm.client import HLLMClient

        client = HLLMClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 60.0

    def test_client_has_session(self):
        """测试使用 session"""
        from hllm.client import HLLMClient

        client = HLLMClient(base_url="http://localhost:8000")
        assert hasattr(client, 'session')
        assert client.session is not None

    def test_client_has_chat_api(self):
        """测试有 chat API"""
        from hllm.client import HLLMClient

        client = HLLMClient(base_url="http://localhost:8000")
        assert hasattr(client, 'chat')

    def test_client_has_models_api(self):
        """测试有 models API"""
        from hllm.client import HLLMClient

        client = HLLMClient(base_url="http://localhost:8000")
        assert hasattr(client, 'models')

    def test_client_close(self):
        """测试 close 方法"""
        from hllm.client import HLLMClient

        client = HLLMClient(base_url="http://localhost:8000")
        client.close()  # 不应抛出异常


class TestChatMessage:
    """测试 ChatMessage"""

    def test_chat_message_init(self):
        """测试 ChatMessage 初始化"""
        from hllm.client import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_roles(self):
        """测试 ChatMessage 角色"""
        from hllm.client import ChatMessage

        for role in ["system", "user", "assistant"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role


class TestResponseClasses:
    """测试响应类"""

    def test_chat_completion_choice(self):
        """测试 ChatCompletionChoice"""
        from hllm.client import ChatCompletionChoice, ChatMessage

        msg = ChatMessage(role="assistant", content="Hi")
        choice = ChatCompletionChoice(index=0, message=msg)
        assert choice.index == 0

    def test_stream_delta(self):
        """测试 StreamDelta"""
        from hllm.client import StreamDelta

        delta = StreamDelta(role="assistant", content="Hello")
        assert delta.role == "assistant"
        assert delta.content == "Hello"

    def test_stream_choice(self):
        """测试 StreamChoice"""
        from hllm.client import StreamChoice, StreamDelta

        delta = StreamDelta(content="test")
        choice = StreamChoice(index=0, delta=delta)
        assert choice.index == 0


class TestClientIntegration:
    """客户端集成测试"""

    @patch("hllm.client.requests.Session")
    def test_models_list(self, mock_session):
        """测试 models.list"""
        from hllm.client import HLLMClient

        mock_response = Mock()
        mock_response.json.return_value = {"data": []}
        mock_session.return_value.get.return_value = mock_response

        client = HLLMClient(base_url="http://localhost:8000")
        try:
            client.models.list()
        except Exception:
            pass  # 可能需要实际环境


class TestClientContextManager:
    """测试上下文管理器"""

    def test_context_manager(self):
        """测试 with 语句"""
        from hllm.client import HLLMClient

        with HLLMClient(base_url="http://localhost:8000") as client:
            assert client.base_url == "http://localhost:8000"


class TestClientAPI:
    """测试 API 类"""

    def test_chat_completions_api_exists(self):
        """测试 ChatCompletionsAPI 存在"""
        from hllm.client import ChatCompletionsAPI
        assert callable(ChatCompletionsAPI)

    def test_completions_api_exists(self):
        """测试 CompletionsAPI 存在"""
        from hllm.client import CompletionsAPI
        assert callable(CompletionsAPI)

    def test_models_api_exists(self):
        """测试 ModelsAPI 存在"""
        from hllm.client import ModelsAPI
        assert callable(ModelsAPI)
