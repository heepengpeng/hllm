"""
测试 generate.py 核心逻辑
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestGenerateCore(unittest.TestCase):
    """测试生成核心功能"""

    def test_generate_function_signature(self):
        """测试 generate 函数签名"""
        import inspect
        from hllm.generate import generate

        sig = inspect.signature(generate)
        params = list(sig.parameters.keys())

        required_params = ['model', 'tokenizer', 'prompt', 'max_new_tokens',
                          'temperature', 'top_p', 'top_k', 'repetition_penalty', 'device']
        for param in required_params:
            self.assertIn(param, params)

    def test_stream_generate_function_signature(self):
        """测试 stream_generate 函数签名"""
        import inspect
        from hllm.generate import stream_generate

        sig = inspect.signature(stream_generate)
        params = list(sig.parameters.keys())

        required_params = ['model', 'tokenizer', 'prompt', 'max_new_tokens',
                          'temperature', 'top_p', 'top_k', 'repetition_penalty', 'device']
        for param in required_params:
            self.assertIn(param, params)


if __name__ == "__main__":
    unittest.main()
