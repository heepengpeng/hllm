"""
测试 MLX 后端
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestMLXBackendImport(unittest.TestCase):
    """测试 MLX 后端导入"""

    def test_mlx_backend_import(self):
        """测试 MLXBackend 导入"""
        try:
            from hllm.backends.mlx import MLXBackend
            self.assertIsNotNone(MLXBackend)
        except ImportError:
            # MLX 可能未安装，跳过
            self.skipTest("MLX not installed")


class TestMLXBackend(unittest.TestCase):
    """测试 MLXBackend 功能"""

    def setUp(self):
        """设置测试环境"""
        try:
            from hllm.backends.mlx import MLXBackend
            self.MLXBackend = MLXBackend
            self.has_mlx = True
        except ImportError:
            self.has_mlx = False

    def test_backend_class_attributes(self):
        """测试后端类属性"""
        if not self.has_mlx:
            self.skipTest("MLX not installed")

        self.assertEqual(self.MLXBackend.NAME, "mlx")
        self.assertTrue(self.MLXBackend.SUPPORTS_QUANTIZATION)
        self.assertEqual(self.MLXBackend.DEFAULT_DEVICE, "mlx")


if __name__ == "__main__":
    unittest.main()
