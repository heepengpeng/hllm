"""
HLLM OpenAI Compatible REST API Client Test
"""

import json
import unittest
from unittest.mock import Mock, patch

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@unittest.skipUnless(HAS_REQUESTS, "requests not installed")
class TestHLLMClient(unittest.TestCase):
    """测试 OpenAI 兼容客户端"""

    def setUp(self):
        """设置测试环境"""
        from hllm.client import HLLMClient
        self.client = HLLMClient("http://localhost:8000")

    def tearDown(self):
        """清理测试环境"""
        self.client.close()

    @patch("hllm.client.requests.Session.get")
    def test_models_list(self, mock_get):
        """测试模型列表"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"id": "hllm-model", "object": "model"}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.client.models.list()

        self.assertEqual(result["object"], "list")
        self.assertEqual(len(result["data"]), 1)
        self.assertEqual(result["data"][0]["id"], "hllm-model")

    @patch("hllm.client.requests.Session.post")
    def test_chat_completions_create(self, mock_post):
        """测试对话补全创建"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "hllm-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.client.chat.completions.create(
            model="hllm-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=50
        )

        self.assertEqual(response.id, "chatcmpl-123")
        self.assertEqual(response.object, "chat.completion")
        self.assertEqual(response.model, "hllm-model")
        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertEqual(response.choices[0].message.content, "Hello!")
        self.assertEqual(response.usage["total_tokens"], 7)

    @patch("hllm.client.requests.Session.post")
    def test_completions_create(self, mock_post):
        """测试文本补全创建"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1677652288,
            "model": "hllm-model",
            "choices": [{
                "text": " generated text",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.client.completions.create(
            model="hllm-model",
            prompt="Once upon a time",
            max_tokens=50
        )

        self.assertEqual(response.id, "cmpl-123")
        self.assertEqual(response.object, "text_completion")
        self.assertEqual(response.choices[0].text, " generated text")

    @patch("hllm.client.requests.Session.post")
    def test_chat_completions_stream(self, mock_post):
        """测试流式对话补全"""
        # 模拟 SSE 响应
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"hllm-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"hllm-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: [DONE]'
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(self.client.chat.completions.create(
            model="hllm-model",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True
        ))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].object, "chat.completion.chunk")
        self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
        self.assertEqual(chunks[1].choices[0].delta.content, "Hello")

    def test_headers(self):
        """测试请求头包含 Authorization"""
        self.assertEqual(
            self.client.session.headers["Authorization"],
            "Bearer not-needed"
        )
        self.assertEqual(
            self.client.session.headers["Content-Type"],
            "application/json"
        )


@unittest.skipUnless(HAS_REQUESTS, "requests not installed")
class TestHLLMClientContextManager(unittest.TestCase):
    """测试客户端上下文管理器"""

    @patch("hllm.client.requests.Session.close")
    def test_context_manager(self, mock_close):
        """测试上下文管理器正确关闭会话"""
        from hllm.client import HLLMClient

        with HLLMClient("http://localhost:8000") as client:
            self.assertIsNotNone(client.session)

        mock_close.assert_called_once()


@unittest.skipUnless(HAS_REQUESTS, "requests not installed")
class TestDataClasses(unittest.TestCase):
    """测试数据类"""

    def test_chat_message(self):
        """测试 ChatMessage"""
        from hllm.client import ChatMessage

        message = ChatMessage(role="user", content="Hello")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello")

    def test_chat_completion_response(self):
        """测试 ChatCompletionResponse"""
        from hllm.client import ChatCompletionResponse, ChatCompletionChoice, ChatMessage

        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hi"),
            finish_reason="stop"
        )
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=123456,
            model="hllm-model",
            choices=[choice],
            usage={"total_tokens": 10}
        )
        self.assertEqual(response.id, "chatcmpl-123")
        self.assertEqual(response.choices[0].message.content, "Hi")

    def test_completion_response(self):
        """测试 CompletionResponse"""
        from hllm.client import CompletionResponse, CompletionChoice

        choice = CompletionChoice(text="Hello", index=0, finish_reason="stop")
        response = CompletionResponse(
            id="cmpl-123",
            object="text_completion",
            created=123456,
            model="hllm-model",
            choices=[choice],
            usage={}
        )
        self.assertEqual(response.choices[0].text, "Hello")

    def test_stream_response(self):
        """测试流式响应"""
        from hllm.client import ChatCompletionStreamResponse, StreamChoice, StreamDelta

        choice = StreamChoice(
            index=0,
            delta=StreamDelta(content="Hello"),
            finish_reason=None
        )
        response = ChatCompletionStreamResponse(
            id="chatcmpl-123",
            object="chat.completion.chunk",
            created=123456,
            model="hllm-model",
            choices=[choice]
        )
        self.assertEqual(response.object, "chat.completion.chunk")
        self.assertEqual(response.choices[0].delta.content, "Hello")


if __name__ == "__main__":
    unittest.main()
