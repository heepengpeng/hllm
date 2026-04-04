"""
测试生成模块的简单功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestGenerateSimple(unittest.TestCase):
    """测试生成功能"""

    def test_generate_function_exists(self):
        """测试 generate 函数存在"""
        from hllm.generate import generate
        self.assertTrue(callable(generate))

    def test_stream_generate_function_exists(self):
        """测试 stream_generate 函数存在"""
        from hllm.generate import stream_generate
        self.assertTrue(callable(stream_generate))


class TestGenerateParams(unittest.TestCase):
    """测试生成参数"""

    def test_default_params(self):
        """测试默认参数"""
        import inspect
        from hllm.generate import generate

        sig = inspect.signature(generate)
        params = sig.parameters

        # 检查默认参数
        self.assertEqual(params['max_new_tokens'].default, 128)
        self.assertEqual(params['temperature'].default, 1.0)
        self.assertEqual(params['top_p'].default, 1.0)
        self.assertEqual(params['top_k'].default, 50)


if __name__ == "__main__":
    unittest.main()
