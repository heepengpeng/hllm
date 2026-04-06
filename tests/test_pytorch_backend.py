"""
测试 PyTorch 后端
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestPyTorchBackend(unittest.TestCase):
    """测试 PyTorchBackend"""

    def test_backend_class_exists(self):
        """测试后端类存在"""
        from hllm.backends.pytorch import PyTorchBackend
        self.assertTrue(hasattr(PyTorchBackend, 'NAME'))
        self.assertEqual(PyTorchBackend.NAME, "pytorch")

    def test_backend_class_attributes(self):
        """测试后端类属性"""
        from hllm.backends.pytorch import PyTorchBackend

        self.assertTrue(hasattr(PyTorchBackend, 'NAME'))
        self.assertTrue(hasattr(PyTorchBackend, 'SUPPORTS_QUANTIZATION'))
        self.assertTrue(hasattr(PyTorchBackend, 'SUPPORTS_GPU'))
        self.assertEqual(PyTorchBackend.SUPPORTS_GPU, True)

    def test_backend_module_structure(self):
        """测试后端模块结构"""
        from hllm.backends.pytorch import PyTorchBackend

        # 检查关键方法存在
        self.assertTrue(hasattr(PyTorchBackend, '_load_model'))
        self.assertTrue(hasattr(PyTorchBackend, '_generate_impl'))
        self.assertTrue(hasattr(PyTorchBackend, '_stream_generate_impl'))

    def test_get_backend_info(self):
        """测试获取后端信息"""
        from hllm.backends.pytorch import PyTorchBackend

        # 静态检查类属性
        self.assertEqual(PyTorchBackend.NAME, "pytorch")
        self.assertTrue(PyTorchBackend.SUPPORTS_GPU)
        self.assertTrue(hasattr(PyTorchBackend, 'DEFAULT_DEVICE'))


class TestPyTorchBackendIntegration(unittest.TestCase):
    """PyTorchBackend 集成测试（需要真实环境）"""

    @classmethod
    def setUpClass(cls):
        """跳过需要真实模型的测试"""
        cls.skip_tests = True
        try:
            import torch
            cls.skip_tests = False
        except ImportError:
            pass

    def test_pytorch_available(self):
        """测试 PyTorch 可用"""
        import torch
        self.assertIsNotNone(torch.__version__)

    def test_backend_creation_without_model(self):
        """测试不加载模型的后端创建"""
        from hllm.backends.pytorch import PyTorchBackend

        # 创建一个不加载真实模型的测试
        # 检查类的初始化参数
        import inspect
        sig = inspect.signature(PyTorchBackend.__init__)
        params = list(sig.parameters.keys())
        self.assertIn('model_path', params)


if __name__ == "__main__":
    unittest.main()
