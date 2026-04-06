"""
扩展后端测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestBackendModule(unittest.TestCase):
    """测试后端模块功能"""

    def test_all_exports(self):
        """测试 __all__ 导出"""
        import sys
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
            
        from hllm.backends import __all__

        expected = ["BaseBackend", "create_backend", "list_backends", "get_backend_info"]
        for item in expected:
            self.assertIn(item, __all__)

    def test_backends_registered(self):
        """测试后端注册"""
        import sys
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
            
        from hllm.backends import list_backends

        backends = list_backends()
        self.assertIsInstance(backends, list)
        self.assertIn("pytorch", backends)

    def test_get_backend_info_returns_dict(self):
        """测试 get_backend_info 返回字典"""
        import sys
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
            
        from hllm.backends import get_backend_info

        info = get_backend_info()
        self.assertIsInstance(info, dict)
        self.assertIn("pytorch", info)


class TestAutoSelectExtended(unittest.TestCase):
    """扩展自动选择测试"""

    def test_auto_select_backend_on_darwin(self):
        """测试 Darwin 系统自动选择"""
        import sys
        if 'hllm.backends' in sys.modules:
            del sys.modules['hllm.backends']
        if 'hllm' in sys.modules:
            del sys.modules['hllm']
            
        from hllm.backends import auto_select_backend

        backend_name, kwargs = auto_select_backend("test-model")
        
        # Darwin 上可能是 mlx 或 pytorch
        self.assertIn(backend_name, ["mlx", "pytorch"])
        self.assertEqual(kwargs["model_path"], "test-model")


if __name__ == "__main__":
    unittest.main()
