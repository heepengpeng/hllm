"""
测试 Server 简单功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestServerImport(unittest.TestCase):
    """测试 Server 导入"""

    def test_server_import(self):
        """测试 Server 导入"""
        try:
            from hllm.server import Server
            self.assertIsNotNone(Server)
        except ImportError:
            self.skipTest("FastAPI not installed")

    def test_main_import(self):
        """测试 main 函数导入"""
        try:
            from hllm.server import main
            self.assertTrue(callable(main))
        except ImportError:
            self.skipTest("FastAPI not installed")


class TestServerClass(unittest.TestCase):
    """测试 Server 类"""

    def test_server_init(self):
        """测试 Server 初始化"""
        try:
            from hllm.server import Server
        except ImportError:
            self.skipTest("FastAPI not installed")

        mock_model = Mock()
        mock_model.model_path = "test-model"

        server = Server(mock_model, host="127.0.0.1", port=9000)

        self.assertEqual(server.host, "127.0.0.1")
        self.assertEqual(server.port, 9000)


class TestPydanticModels(unittest.TestCase):
    """测试 Pydantic 模型"""

    def test_chat_message(self):
        """测试 ChatMessage"""
        try:
            from hllm.server import ChatMessage
        except ImportError:
            self.skipTest("FastAPI not installed")

        msg = ChatMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

    def test_chat_completion_request(self):
        """测试 ChatCompletionRequest"""
        try:
            from hllm.server import ChatCompletionRequest, ChatMessage
        except ImportError:
            self.skipTest("FastAPI not installed")

        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=50,
            temperature=0.7
        )
        self.assertEqual(req.model, "test-model")
        self.assertEqual(req.max_tokens, 50)
        self.assertEqual(req.temperature, 0.7)


if __name__ == "__main__":
    unittest.main()
