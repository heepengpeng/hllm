"""
测试 client 模块
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestClientClasses(unittest.TestCase):
    """测试客户端类"""

    def test_client_class_exists(self):
        """测试 HLLMClient 类存在"""
        from hllm.client import HLLMClient
        self.assertTrue(callable(HLLMClient))

    def test_chat_message_class_exists(self):
        """测试 ChatMessage 类存在"""
        from hllm.client import ChatMessage
        self.assertTrue(callable(ChatMessage))

    def test_chat_completion_choice_class_exists(self):
        """测试 ChatCompletionChoice 类存在"""
        from hllm.client import ChatCompletionChoice
        self.assertTrue(callable(ChatCompletionChoice))


class TestClientMethods(unittest.TestCase):
    """测试客户端方法"""

    def test_client_has_methods(self):
        """测试客户端有基本方法"""
        from hllm.client import HLLMClient
        # 检查基本属性
        self.assertTrue(hasattr(HLLMClient, '__init__'))


class TestChatMessage(unittest.TestCase):
    """测试 ChatMessage"""

    def test_chat_message_init(self):
        """测试 ChatMessage 初始化"""
        from hllm.client import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")


class TestClientIntegration(unittest.TestCase):
    """客户端集成测试"""

    def test_client_can_be_instantiated(self):
        """测试客户端可以实例化"""
        from hllm.client import HLLMClient

        client = HLLMClient(base_url="http://localhost:8000")
        self.assertIsNotNone(client)
        self.assertTrue(hasattr(client, 'base_url'))


if __name__ == "__main__":
    unittest.main()
