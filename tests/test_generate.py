"""
测试生成模块
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestGenerate(unittest.TestCase):
    """测试生成函数"""

    def test_import(self):
        """测试导入"""
        from hllm.generate import generate, stream_generate
        self.assertIsNotNone(generate)
        self.assertIsNotNone(stream_generate)


class TestGenerateFunctions(unittest.TestCase):
    """测试生成功能"""

    def setUp(self):
        """设置测试环境"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

    def test_generate_function_exists(self):
        """测试 generate 函数存在"""
        from hllm import generate
        self.assertTrue(callable(generate))

    def test_stream_generate_function_exists(self):
        """测试 stream_generate 函数存在"""
        from hllm.generate import stream_generate
        self.assertTrue(callable(stream_generate))


if __name__ == "__main__":
    unittest.main()
