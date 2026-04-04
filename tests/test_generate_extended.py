"""
扩展生成模块测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestGenerateImports(unittest.TestCase):
    """测试生成模块导入"""

    def test_imports(self):
        """测试所有导入"""
        from hllm.generate import generate, stream_generate
        self.assertIsNotNone(generate)
        self.assertIsNotNone(stream_generate)


class TestGenerateSetup(unittest.TestCase):
    """测试生成设置"""

    def test_device_selection(self):
        """测试设备选择"""
        from hllm.generate import generate
        # 测试函数存在，参数正确
        import inspect
        sig = inspect.signature(generate)
        self.assertIn('device', sig.parameters)
        self.assertEqual(sig.parameters['device'].default, 'cpu')


class TestStreamGenerateSetup(unittest.TestCase):
    """测试流式生成设置"""

    def test_stream_params(self):
        """测试流式参数"""
        from hllm.generate import stream_generate
        import inspect
        sig = inspect.signature(stream_generate)
        self.assertIn('max_new_tokens', sig.parameters)
        self.assertIn('temperature', sig.parameters)


if __name__ == "__main__":
    unittest.main()
