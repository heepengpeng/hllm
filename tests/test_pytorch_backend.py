"""
测试 PyTorch 后端
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestPyTorchBackend(unittest.TestCase):
    """测试 PyTorchBackend"""

    @patch("hllm.backends.pytorch.AutoTokenizer")
    @patch("hllm.backends.pytorch.AutoModelForCausalLM")
    def test_backend_init(self, mock_model_class, mock_tokenizer_class):
        """测试后端初始化"""
        from hllm.backends.pytorch import PyTorchBackend

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        backend = PyTorchBackend("test-model", device="cpu")

        self.assertEqual(backend.model_path, "test-model")
        self.assertEqual(backend.device_name, "cpu")
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_normalize_device(self, mock_cuda, mock_mps):
        """测试设备标准化"""
        from hllm.backends.pytorch import PyTorchBackend

        mock_cuda.return_value = False
        mock_mps.return_value = False

        with patch("hllm.backends.pytorch.AutoTokenizer") as mock_tok, \
             patch("hllm.backends.pytorch.AutoModelForCausalLM") as mock_model:

            mock_tok.from_pretrained.return_value = Mock(eos_token_id=2, pad_token_id=0)
            mock_m = Mock()
            mock_m.config = Mock()
            mock_m.to.return_value = mock_m
            mock_model.from_pretrained.return_value = mock_m

            backend = PyTorchBackend("test-model", device="mps")
            # 当 MPS 不可用时应该回退到 CPU
            self.assertEqual(backend.device_name, "cpu")

    def test_backend_properties(self):
        """测试后端属性"""
        from hllm.backends.pytorch import PyTorchBackend

        with patch("hllm.backends.pytorch.AutoTokenizer") as mock_tok, \
             patch("hllm.backends.pytorch.AutoModelForCausalLM") as mock_model:

            mock_tokenizer = Mock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token_id = 0
            mock_tok.from_pretrained.return_value = mock_tokenizer

            mock_m = Mock()
            mock_m.config = Mock()
            mock_m.to.return_value = mock_m
            mock_model.from_pretrained.return_value = mock_m

            backend = PyTorchBackend("test-model", device="cpu")

            self.assertEqual(backend.eos_token_id, 2)
            self.assertEqual(backend.pad_token_id, 0)
            self.assertEqual(backend.tokenizer, mock_tokenizer)
            self.assertEqual(backend.config, mock_m.config)

    def test_get_info(self):
        """测试获取后端信息"""
        from hllm.backends.pytorch import PyTorchBackend

        with patch("hllm.backends.pytorch.AutoTokenizer") as mock_tok, \
             patch("hllm.backends.pytorch.AutoModelForCausalLM") as mock_model:

            mock_tok.from_pretrained.return_value = Mock(eos_token_id=2, pad_token_id=0)
            mock_m = Mock()
            mock_m.config = Mock()
            mock_m.to.return_value = mock_m
            mock_model.from_pretrained.return_value = mock_m

            backend = PyTorchBackend("test-model", device="cpu")
            info = backend.get_info()

            self.assertEqual(info["name"], "pytorch")
            self.assertEqual(info["device"], "cpu")
            self.assertTrue(info["supports_quantization"])


if __name__ == "__main__":
    unittest.main()
