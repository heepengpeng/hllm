"""
扩展后端测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestBackendModule(unittest.TestCase):
    """测试后端模块功能"""

    def test_all_exports(self):
        """测试 __all__ 导出"""
        from hllm.backends import __all__

        expected = ["BaseBackend", "create_backend", "list_backends", "get_backend_info"]
        for item in expected:
            self.assertIn(item, __all__)


class TestAutoSelectExtended(unittest.TestCase):
    """扩展自动选择测试"""

    @patch("platform.system")
    @patch("platform.machine")
    def test_apple_silicon_detection(self, mock_machine, mock_system):
        """测试 Apple Silicon 检测"""
        from hllm.backends import auto_select_backend

        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        backend_name, kwargs = auto_select_backend("test-model")

        # 在 Apple Silicon 上应该尝试 MLX
        self.assertIn(backend_name, ["mlx", "pytorch"])
        self.assertEqual(kwargs["model_path"], "test-model")

    @patch("platform.system")
    def test_linux_detection(self, mock_system):
        """测试 Linux 检测"""
        from hllm.backends import auto_select_backend

        mock_system.return_value = "Linux"

        backend_name, kwargs = auto_select_backend("test-model")

        self.assertEqual(backend_name, "pytorch")


if __name__ == "__main__":
    unittest.main()
