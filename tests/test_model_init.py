"""
测试 HLLM 模型初始化
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestHLLMInit(unittest.TestCase):
    """测试 HLLM 初始化逻辑"""

    def test_model_path_storage(self):
        """测试模型路径存储"""
        from hllm.model import HLLM

        # 直接 patch backends.create_backend
        with patch('hllm.backends.create_backend') as mock_create:
            mock_backend = Mock()
            mock_backend.get_info.return_value = {}
            mock_create.return_value = mock_backend

            model = HLLM("test-model", backend="pytorch")
            self.assertEqual(model.model_path, "test-model")


class TestHLLMBackendSelection(unittest.TestCase):
    """测试后端选择逻辑"""

    def test_backend_name_storage(self):
        """测试后端名称存储"""
        from hllm.model import HLLM

        with patch('hllm.backends.create_backend') as mock_create:
            mock_backend = Mock()
            mock_backend.get_info.return_value = {}
            mock_create.return_value = mock_backend

            model = HLLM("test-model", backend="pytorch")
            self.assertEqual(model.backend_name, "pytorch")


if __name__ == "__main__":
    unittest.main()
