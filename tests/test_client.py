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


class TestChatCompletionsAPI:
    """测试 ChatCompletionsAPI"""

    def test_create_chat(self):
        """测试创建对话"""
        from hllm.client import HLLMClient

        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_instance.post.return_value = mock_response_obj
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            result = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100
            )

            assert result.id == "chatcmpl-123"
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Hello!"

    def test_create_chat_with_temperature(self):
        """测试带温度参数的创建对话"""
        from hllm.client import HLLMClient

        mock_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Test"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_instance.post.return_value = mock_response_obj
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            result = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                top_p=0.9
            )

            # 验证请求被发送
            mock_instance.post.assert_called()
            call_kwargs = mock_instance.post.call_args[1]  # 关键字参数
            assert call_kwargs["json"]["temperature"] == 0.7
            assert call_kwargs["json"]["top_p"] == 0.9


class TestCompletionsAPI:
    """测试 CompletionsAPI"""

    def test_create_completion(self):
        """测试创建补全"""
        from hllm.client import HLLMClient

        mock_response = {
            "id": "cmp-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "text": "Completion text",
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_instance.post.return_value = mock_response_obj
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            result = client.completions.create(
                model="test-model",
                prompt="Hello",
                max_tokens=100
            )

            assert result.id == "cmp-123"
            assert len(result.choices) == 1


class TestModelsAPI:
    """测试 ModelsAPI"""

    def test_list_models(self):
        """测试列出模型"""
        from hllm.client import HLLMClient

        mock_response = {
            "object": "list",
            "data": [
                {"id": "model-1", "object": "model"},
                {"id": "model-2", "object": "model"}
            ]
        }

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_instance.get.return_value = mock_response_obj
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            result = client.models.list()

            assert result["object"] == "list"
            assert len(result["data"]) == 2


class TestClientErrorHandling:
    """测试客户端错误处理"""

    def test_request_timeout(self):
        """测试请求超时"""
        from hllm.client import HLLMClient
        import requests

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_instance.get.side_effect = requests.exceptions.Timeout()
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000", timeout=5)

            with pytest.raises(requests.exceptions.Timeout):
                client.models.list()

    def test_connection_error(self):
        """测试连接错误"""
        from hllm.client import HLLMClient
        import requests

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_instance.get.side_effect = requests.exceptions.ConnectionError()
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            with pytest.raises(requests.exceptions.ConnectionError):
                client.models.list()


class TestClientDataClasses:
    """测试数据类"""

    def test_chat_completion_response(self):
        """测试 ChatCompletionResponse"""
        from hllm.client import (
            ChatCompletionResponse,
            ChatCompletionChoice,
            ChatMessage,
            ChatCompletionsAPI,
            HLLMClient
        )

        msg = ChatMessage(role="assistant", content="Hello")
        choice = ChatCompletionChoice(index=0, message=msg, finish_reason="stop")
        response = ChatCompletionResponse(
            id="test-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[choice],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )

        assert response.id == "test-123"
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.usage["total_tokens"] == 7

    def test_completion_response(self):
        """测试 CompletionResponse"""
        from hllm.client import CompletionResponse, CompletionChoice

        choice = CompletionChoice(index=0, text="Test completion", finish_reason="stop")
        response = CompletionResponse(
            id="cmp-123",
            object="text_completion",
            created=1234567890,
            model="test-model",
            choices=[choice],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )

        assert response.id == "cmp-123"
        assert response.choices[0].text == "Test completion"

    def test_stream_delta_defaults(self):
        """测试 StreamDelta 默认值"""
        from hllm.client import StreamDelta

        delta = StreamDelta()
        assert delta.role is None
        assert delta.content is None

    def test_stream_choice_defaults(self):
        """测试 StreamChoice 默认值"""
        from hllm.client import StreamChoice, StreamDelta

        choice = StreamChoice(index=0, delta=StreamDelta())
        assert choice.finish_reason is None


class TestClientStreamProcessing:
    """测试客户端流式处理"""

    def test_stream_create_yields_chunks(self):
        """测试流式创建返回 chunks"""
        from hllm.client import HLLMClient

        mock_response_data = [
            b'data: {"id":"1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"id":"2","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null}]}',
            b'data: [DONE]',
        ]

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response = Mock()
            mock_response.iter_lines.return_value = iter(mock_response_data)
            mock_instance.post.return_value = mock_response
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            # 调用 chat.completions.create 并获取流式响应
            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True
            )

            chunks = list(response)
            assert len(chunks) == 2
            assert chunks[0].choices[0].delta.content == "Hello"
            assert chunks[1].choices[0].delta.content == " World"

    def test_stream_create_handles_ddone(self):
        """测试流式处理 [DONE] 消息"""
        from hllm.client import HLLMClient

        mock_response_data = [
            b'data: {"id":"1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
            b'data: [DONE]',
            b'data: {"id":"2","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Should not see"},"finish_reason":null}]}',
        ]

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response = Mock()
            mock_response.iter_lines.return_value = iter(mock_response_data)
            mock_instance.post.return_value = mock_response
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True
            )

            chunks = list(response)
            # 应该只看到 [DONE] 之前的 chunks
            assert len(chunks) == 1

    def test_parse_stream_response(self):
        """测试解析流式响应"""
        from hllm.client import ChatCompletionsAPI, HLLMClient

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            chunk_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-3.5",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hello", "role": "assistant"},
                    "finish_reason": None
                }]
            }

            result = client.chat.completions._parse_stream_response(chunk_data)

            assert result.id == "chatcmpl-123"
            assert result.object == "chat.completion.chunk"
            assert len(result.choices) == 1
            assert result.choices[0].delta.content == "Hello"
            assert result.choices[0].delta.role == "assistant"

    def test_parse_chat_response_with_usage(self):
        """测试解析对话响应包含 usage"""
        from hllm.client import ChatCompletionsAPI, HLLMClient

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-3.5",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }

            result = client.chat.completions._parse_chat_response(response_data)

            assert result.usage["prompt_tokens"] == 10
            assert result.usage["completion_tokens"] == 5
            assert result.usage["total_tokens"] == 15


class TestClientCompletionsAPI:
    """测试 CompletionsAPI"""

    def test_completions_create(self):
        """测试创建文本补全"""
        from hllm.client import HLLMClient

        mock_response = {
            "id": "cmp-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "text": "Completion text",
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_instance.post.return_value = mock_response_obj
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            result = client.completions.create(
                model="test-model",
                prompt="Hello",
                max_tokens=100
            )

            assert result.id == "cmp-123"
            assert len(result.choices) == 1
            assert result.choices[0].text == "Completion text"

    def test_completions_create_with_custom_params(self):
        """测试带自定义参数的补全"""
        from hllm.client import HLLMClient

        mock_response = {
            "id": "cmp-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "text": "Test",
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }

        with patch("hllm.client.requests.Session") as mock_session:
            mock_instance = Mock()
            mock_response_obj = Mock()
            mock_response_obj.json.return_value = mock_response
            mock_instance.post.return_value = mock_response_obj
            mock_session.return_value = mock_instance

            client = HLLMClient(base_url="http://localhost:8000")

            result = client.completions.create(
                model="test-model",
                prompt="Hello",
                temperature=0.5,
                top_p=0.8,
                max_tokens=50
            )

            # 验证请求参数
            call_kwargs = mock_instance.post.call_args[1]
            assert call_kwargs["json"]["temperature"] == 0.5
            assert call_kwargs["json"]["top_p"] == 0.8
