"""
测试自动选择后端功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestAutoSelectBackend(unittest.TestCase):
    """测试自动选择后端"""

    def test_auto_select_returns_tuple(self):
        """测试 auto_select_backend 返回元组格式"""
        # 直接导入避免触发 hllm/__init__.py 的 transformers 导入
        import sys
        import importlib
        
        # 确保 backends 模块直接加载
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
        
        from hllm.backends import auto_select_backend
        
        result = auto_select_backend("test-model")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIn(result[0], ["pytorch", "mlx", "paged_pytorch"])
        self.assertIn("model_path", result[1])

    def test_auto_select_with_device_cpu(self):
        """测试指定 CPU 设备"""
        import sys
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
            
        from hllm.backends import auto_select_backend
        
        backend_name, kwargs = auto_select_backend("test-model", device="cpu")
        self.assertEqual(backend_name, "pytorch")

    def test_auto_select_without_model_path(self):
        """测试不指定模型路径"""
        import sys
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
            
        from hllm.backends import auto_select_backend
        
        backend_name, kwargs = auto_select_backend(device="cpu")
        self.assertEqual(backend_name, "pytorch")
        self.assertNotIn("model_path", kwargs)


if __name__ == "__main__":
    unittest.main()
