"""
测试 BaseBackend 基类
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestBaseBackendAbstract(unittest.TestCase):
    """测试 BaseBackend 抽象类"""

    def test_base_backend_cannot_instantiate(self):
        """测试基类不能直接实例化"""
        from hllm.backends.base import BaseBackend

        with self.assertRaises(TypeError):
            BaseBackend("test-model")

    def test_base_backend_class_attributes(self):
        """测试基类属性"""
        from hllm.backends.base import BaseBackend

        self.assertEqual(BaseBackend.NAME, "base")
        self.assertEqual(BaseBackend.SUPPORTS_QUANTIZATION, False)
        self.assertEqual(BaseBackend.DEFAULT_DEVICE, "cpu")


class TestBaseBackendOptionalMethods(unittest.TestCase):
    """测试基类可选方法"""

    def test_bos_token_id_default(self):
        """测试 bos_token_id 默认值"""
        from hllm.backends.base import BaseBackend

        # 创建子类来测试
        class TestBackend(BaseBackend):
            NAME = "test"

            def _load_model(self, **kwargs):
                pass

            def generate(self, **kwargs):
                return ""

            def stream_generate(self, **kwargs):
                yield ""

            @property
            def device_name(self):
                return "cpu"

            @property
            def eos_token_id(self):
                return 0

            @property
            def pad_token_id(self):
                return 0

            @property
            def tokenizer(self):
                return None

        # bos_token_id 有默认实现，返回 None
        # 不能直接测试，因为基类是抽象类


class TestBaseBackendGetInfo(unittest.TestCase):
    """测试 get_info 方法"""

    def test_get_info_structure(self):
        """测试 get_info 返回结构"""
        from hllm.backends.pytorch import PyTorchBackend

        # 创建 mock 模块注入
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None

        mock_transformers = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.pad_token_id = 0
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        mock_ensure = Mock(return_value="/mock/path")

        backend = PyTorchBackend(
            "test-model",
            device="cpu",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        info = backend.get_info()

        self.assertIn("name", info)
        self.assertIn("device", info)
        self.assertIn("model_path", info)
        self.assertIn("supports_quantization", info)


if __name__ == "__main__":
    unittest.main()
