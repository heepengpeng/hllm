"""
测试创建后端功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestBackendAvailability(unittest.TestCase):
    """测试后端可用性检查"""

    def test_pytorch_in_list(self):
        """测试 PyTorch 在可用后端列表中"""
        from hllm.backends import list_backends

        backends = list_backends()
        self.assertIn("pytorch", backends)

    def test_backend_info_structure(self):
        """测试后端信息结构"""
        from hllm.backends import get_backend_info

        info = get_backend_info()

        for backend_name in ["pytorch", "mlx"]:
            self.assertIn(backend_name, info)
            backend_info = info[backend_name]

            # 检查必需的字段
            self.assertIn("available", backend_info)
            self.assertIn("supports_quantization", backend_info)
            self.assertIn("default_device", backend_info)

    def test_create_backend_unknown_raises(self):
        """测试创建未知后端抛出异常"""
        from hllm.backends import create_backend

        with self.assertRaises(ValueError) as context:
            create_backend("unknown_backend", model_path="test")

        self.assertIn("Unknown backend", str(context.exception))


if __name__ == "__main__":
    unittest.main()
