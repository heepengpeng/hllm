"""
扩展 HLLM 测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestHLLMExtended(unittest.TestCase):
    """扩展 HLLM 测试"""

    def test_bos_token_id_with_attribute(self):
        """测试 bos_token_id 有属性时"""
        from hllm.model import HLLM

        with patch('hllm.backends.create_backend') as mock_create:
            mock_backend = Mock()
            mock_backend.bos_token_id = 1
            mock_backend.get_info.return_value = {}
            mock_create.return_value = mock_backend

            model = HLLM("test-model", backend="pytorch")
            self.assertEqual(model.bos_token_id, 1)

    def test_bos_token_id_without_attribute(self):
        """测试 bos_token_id 无属性时"""
        from hllm.model import HLLM

        with patch('hllm.backends.create_backend') as mock_create:
            mock_backend = Mock()
            # 删除 bos_token_id 属性
            del mock_backend.bos_token_id
            mock_backend.get_info.return_value = {}
            mock_create.return_value = mock_backend

            model = HLLM("test-model", backend="pytorch")
            self.assertIsNone(model.bos_token_id)


class TestHLLMGenerateExtended(unittest.TestCase):
    """扩展生成测试"""

    def test_generate_passes_all_params(self):
        """测试 generate 传递所有参数"""
        from hllm.model import HLLM

        with patch('hllm.backends.create_backend') as mock_create:
            mock_backend = Mock()
            mock_backend.generate.return_value = "result"
            mock_backend.get_info.return_value = {}
            mock_create.return_value = mock_backend

            model = HLLM("test-model", backend="pytorch")
            result = model.generate(
                "prompt",
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                custom_param=True
            )

            mock_backend.generate.assert_called_once()
            call_kwargs = mock_backend.generate.call_args[1]
            self.assertEqual(call_kwargs['max_new_tokens'], 50)
            self.assertEqual(call_kwargs['temperature'], 0.7)
            self.assertEqual(call_kwargs['top_p'], 0.9)
            self.assertEqual(call_kwargs['top_k'], 40)
            self.assertEqual(call_kwargs['repetition_penalty'], 1.2)
            self.assertEqual(call_kwargs['custom_param'], True)

    def test_stream_generate_passes_params(self):
        """测试 stream_generate 传递参数"""
        from hllm.model import HLLM

        with patch('hllm.backends.create_backend') as mock_create:
            mock_backend = Mock()
            mock_backend.stream_generate.return_value = iter(["a", "b"])
            mock_backend.get_info.return_value = {}
            mock_create.return_value = mock_backend

            model = HLLM("test-model", backend="pytorch")
            list(model.stream_generate("prompt", max_new_tokens=30))

            mock_backend.stream_generate.assert_called_once()
            call_kwargs = mock_backend.stream_generate.call_args[1]
            self.assertEqual(call_kwargs['max_new_tokens'], 30)


if __name__ == "__main__":
    unittest.main()
