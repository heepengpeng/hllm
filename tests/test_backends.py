"""
测试后端模块
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestBackendImports(unittest.TestCase):
    """测试后端导入"""

    def test_base_backend_import(self):
        """测试 BaseBackend 导入"""
        from hllm.backends.base import BaseBackend
        self.assertIsNotNone(BaseBackend)

    def test_pytorch_backend_import(self):
        """测试 PyTorchBackend 导入"""
        from hllm.backends.pytorch import PyTorchBackend
        self.assertIsNotNone(PyTorchBackend)

    def test_backend_functions_import(self):
        """测试后端函数导入"""
        from hllm.backends import create_backend, list_backends, get_backend_info
        self.assertIsNotNone(create_backend)
        self.assertIsNotNone(list_backends)
        self.assertIsNotNone(get_backend_info)


class TestBackendFunctions(unittest.TestCase):
    """测试后端功能"""

    def test_list_backends(self):
        """测试列出可用后端"""
        from hllm.backends import list_backends

        backends = list_backends()
        self.assertIsInstance(backends, list)
        # 至少应该包含 pytorch
        self.assertIn("pytorch", backends)

    def test_get_backend_info(self):
        """测试获取后端信息"""
        from hllm.backends import get_backend_info

        info = get_backend_info()
        self.assertIsInstance(info, dict)
        self.assertIn("pytorch", info)

        # 检查 pytorch 后端信息
        self.assertIn("available", info["pytorch"])
        self.assertIn("supports_quantization", info["pytorch"])
        self.assertIn("default_device", info["pytorch"])

    def test_create_backend_unknown(self):
        """测试创建未知后端"""
        from hllm.backends import create_backend

        with self.assertRaises(ValueError) as context:
            create_backend("unknown", model_path="test-model")

        self.assertIn("Unknown backend", str(context.exception))


if __name__ == "__main__":
    unittest.main()
