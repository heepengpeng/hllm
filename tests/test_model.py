"""
测试 HLLM 模型类
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestHLLM(unittest.TestCase):
    """测试 HLLM 类"""

    def test_import(self):
        """测试 HLLM 导入"""
        from hllm.model import HLLM
        self.assertIsNotNone(HLLM)


class TestHLLMProperties(unittest.TestCase):
    """测试 HLLM 属性"""

    def setUp(self):
        """设置测试环境"""
        # 创建一个模拟的后端
        self.mock_backend = Mock()
        self.mock_backend.eos_token_id = 2
        self.mock_backend.bos_token_id = 1
        self.mock_backend.pad_token_id = 0
        self.mock_backend.config = {"test": "config"}
        self.mock_backend.tokenizer = Mock()
        self.mock_backend.get_info.return_value = {"device": "cpu"}

    def test_backend_properties(self):
        """测试后端属性访问"""
        from hllm.model import HLLM

        # 直接创建 HLLM 实例并替换 _backend
        with patch.object(HLLM, '__init__', lambda s, *a, **k: None):
            model = HLLM.__new__(HLLM)
            model._backend = self.mock_backend
            model.backend_name = "pytorch"
            model.model_path = "test-model"

            self.assertEqual(model.eos_token_id, 2)
            self.assertEqual(model.bos_token_id, 1)
            self.assertEqual(model.pad_token_id, 0)
            self.assertEqual(model.config, {"test": "config"})
            self.assertEqual(model.tokenizer, self.mock_backend.tokenizer)

    def test_get_info(self):
        """测试获取信息"""
        from hllm.model import HLLM

        with patch.object(HLLM, '__init__', lambda s, *a, **k: None):
            model = HLLM.__new__(HLLM)
            model._backend = self.mock_backend
            model.backend_name = "pytorch"

            info = model.get_info()
            self.assertEqual(info["backend"], "pytorch")

    def test_generate(self):
        """测试生成方法"""
        from hllm.model import HLLM

        self.mock_backend.generate.return_value = "Generated text"

        with patch.object(HLLM, '__init__', lambda s, *a, **k: None):
            model = HLLM.__new__(HLLM)
            model._backend = self.mock_backend

            result = model.generate("Hello", max_new_tokens=50)
            self.assertEqual(result, "Generated text")
            self.mock_backend.generate.assert_called_once()

    def test_stream_generate(self):
        """测试流式生成"""
        from hllm.model import HLLM

        self.mock_backend.stream_generate.return_value = iter(["Hello", " world"])

        with patch.object(HLLM, '__init__', lambda s, *a, **k: None):
            model = HLLM.__new__(HLLM)
            model._backend = self.mock_backend

            tokens = list(model.stream_generate("Hello"))
            self.assertEqual(tokens, ["Hello", " world"])


if __name__ == "__main__":
    unittest.main()
