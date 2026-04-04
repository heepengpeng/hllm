"""
更多 generate 模块测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestGenerateHelpers(unittest.TestCase):
    """测试生成辅助函数"""

    def test_imports(self):
        """测试导入"""
        from hllm.generate import generate, stream_generate
        self.assertTrue(callable(generate))
        self.assertTrue(callable(stream_generate))


class TestGenerateDefaults(unittest.TestCase):
    """测试生成默认值"""

    def test_default_max_tokens(self):
        """测试默认最大 token 数"""
        import inspect
        from hllm.generate import generate

        sig = inspect.signature(generate)
        self.assertEqual(sig.parameters['max_new_tokens'].default, 128)

    def test_default_temperature(self):
        """测试默认温度"""
        import inspect
        from hllm.generate import generate

        sig = inspect.signature(generate)
        self.assertEqual(sig.parameters['temperature'].default, 1.0)

    def test_default_top_p(self):
        """测试默认 top_p"""
        import inspect
        from hllm.generate import generate

        sig = inspect.signature(generate)
        self.assertEqual(sig.parameters['top_p'].default, 1.0)

    def test_default_top_k(self):
        """测试默认 top_k"""
        import inspect
        from hllm.generate import generate

        sig = inspect.signature(generate)
        self.assertEqual(sig.parameters['top_k'].default, 50)


if __name__ == "__main__":
    unittest.main()
