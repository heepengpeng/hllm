import unittest
import torch


class TestHLLM(unittest.TestCase):
    """HLLM 测试"""

    def test_import(self):
        """测试导入"""
        from hllm import HLLM, Tokenizer, generate
        self.assertIsNotNone(HLLM)
        self.assertIsNotNone(Tokenizer)
        self.assertIsNotNone(generate)


if __name__ == "__main__":
    unittest.main()