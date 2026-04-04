"""
HLLM REST API 客户端测试
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@unittest.skipUnless(HAS_REQUESTS, "requests not installed")
class TestHLLMClient(unittest.TestCase):
    """测试 REST API 客户端"""

    def setUp(self):
        """设置测试环境"""
        from hllm.client import HLLMClient
        self.client = HLLMClient("http://localhost:8000")

    def tearDown(self):
        """清理测试环境"""
        self.client.close()

    @patch("hllm.client.requests.Session.get")
    def test_health(self, mock_get):
        """测试健康检查"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok", "model": "test"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.client.health()

        self.assertEqual(result["status"], "ok")
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            timeout=60.0
        )

    @patch("hllm.client.requests.Session.get")
    def test_get_models(self, mock_get):
        """测试获取模型信息"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "test-model",
            "device": "cpu"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.client.get_models()

        self.assertEqual(result["model"], "test-model")

    @patch("hllm.client.requests.Session.post")
    def test_generate(self, mock_post):
        """测试生成文本"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "text": "Generated text",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = self.client.generate(
            prompt="Hello",
            max_new_tokens=50,
            temperature=0.8
        )

        self.assertEqual(result.text, "Generated text")
        self.assertEqual(result.usage["total_tokens"], 15)

        # 验证请求参数
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "http://localhost:8000/generate")
        self.assertEqual(call_args[1]["json"]["prompt"], "Hello")
        self.assertEqual(call_args[1]["json"]["temperature"], 0.8)

    @patch("hllm.client.requests.Session.post")
    def test_chat(self, mock_post):
        """测试对话"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "I'm doing well!"
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "How are you?"}]
        result = self.client.chat(messages)

        self.assertEqual(result.message.role, "assistant")
        self.assertEqual(result.message.content, "I'm doing well!")

    @patch("hllm.client.requests.Session.post")
    def test_chat_simple(self, mock_post):
        """测试简化对话"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "usage": {"total_tokens": 10}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = self.client.chat_simple(
            "Hi",
            system="You are a helpful assistant"
        )

        self.assertEqual(result, "Hello!")

    @patch("hllm.client.requests.Session.post")
    def test_generate_stream(self, mock_post):
        """测试流式生成"""
        # 模拟 SSE 响应
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b"data: {\"token\": \"Hello\", \"index\": 0}",
            b"data: {\"token\": \" world\", \"index\": 1}",
            b"data: [DONE]"
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(self.client.generate_stream("Hello"))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].token, "Hello")
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[1].token, " world")


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

    def test_generate_response(self):
        """测试 GenerateResponse"""
        from hllm.client import GenerateResponse

        response = GenerateResponse(
            text="Hello",
            usage={"total_tokens": 10}
        )
        self.assertEqual(response.text, "Hello")
        self.assertEqual(response.usage["total_tokens"], 10)

    def test_chat_message(self):
        """测试 ChatMessage"""
        from hllm.client import ChatMessage

        message = ChatMessage(role="user", content="Hello")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello")

    def test_chat_response(self):
        """测试 ChatResponse"""
        from hllm.client import ChatMessage, ChatResponse

        message = ChatMessage(role="assistant", content="Hi")
        response = ChatResponse(message=message, usage={})
        self.assertEqual(response.message.content, "Hi")

    def test_stream_chunk(self):
        """测试 StreamChunk"""
        from hllm.client import StreamChunk

        chunk = StreamChunk(token="Hello", index=0)
        self.assertEqual(chunk.token, "Hello")
        self.assertEqual(chunk.index, 0)


if __name__ == "__main__":
    unittest.main()
