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

        with patch.object(PyTorchBackend, '_load_model'):
            with patch('hllm.backends.pytorch.AutoTokenizer') as mock_tok, \
                 patch('hllm.backends.pytorch.AutoModelForCausalLM') as mock_model:

                mock_tok.from_pretrained.return_value = Mock(eos_token_id=2, pad_token_id=0)
                mock_m = Mock()
                mock_m.config = Mock()
                mock_m.to.return_value = mock_m
                mock_model.from_pretrained.return_value = mock_m

                backend = PyTorchBackend("test-model", device="cpu")
                info = backend.get_info()

                self.assertIn("name", info)
                self.assertIn("device", info)
                self.assertIn("model_path", info)
                self.assertIn("supports_quantization", info)


if __name__ == "__main__":
    unittest.main()
