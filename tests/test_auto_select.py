"""
测试自动选择后端功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestAutoSelectBackend(unittest.TestCase):
    """测试自动选择后端"""

    @patch("platform.system")
    @patch("platform.machine")
    def test_auto_select_on_linux(self, mock_machine, mock_system):
        """测试在 Linux 上自动选择"""
        from hllm.backends import auto_select_backend

        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"

        backend_name, kwargs = auto_select_backend("test-model")

        self.assertEqual(backend_name, "pytorch")
        self.assertEqual(kwargs["model_path"], "test-model")

    @patch("platform.system")
    @patch("platform.machine")
    def test_auto_select_on_mac_intel(self, mock_machine, mock_system):
        """测试在 Intel Mac 上自动选择"""
        from hllm.backends import auto_select_backend

        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"

        backend_name, kwargs = auto_select_backend("test-model")

        self.assertEqual(backend_name, "pytorch")

    @patch("platform.system")
    @patch("platform.machine")
    def test_auto_select_on_windows(self, mock_machine, mock_system):
        """测试在 Windows 上自动选择"""
        from hllm.backends import auto_select_backend

        mock_system.return_value = "Windows"
        mock_machine.return_value = "AMD64"

        backend_name, kwargs = auto_select_backend("test-model")

        self.assertEqual(backend_name, "pytorch")


if __name__ == "__main__":
    unittest.main()
