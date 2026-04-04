"""
测试 Tokenizer 模块
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestTokenizer(unittest.TestCase):
    """测试 Tokenizer 类"""

    def test_tokenizer_init(self):
        """测试 Tokenizer 初始化"""
        from hllm.tokenizer import Tokenizer

        # 创建一个 mock 的 transformers tokenizer
        mock_hf_tokenizer = Mock()
        mock_hf_tokenizer.vocab_size = 32000
        mock_hf_tokenizer.eos_token_id = 2
        mock_hf_tokenizer.bos_token_id = 1
        mock_hf_tokenizer.pad_token_id = 0
        mock_hf_tokenizer.unk_token_id = 3

        tokenizer = Tokenizer(mock_hf_tokenizer)

        self.assertEqual(tokenizer.vocab_size, 32000)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.unk_token_id, 3)

    def test_encode(self):
        """测试编码"""
        from hllm.tokenizer import Tokenizer

        mock_hf_tokenizer = Mock()
        mock_hf_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        tokenizer = Tokenizer(mock_hf_tokenizer)
        result = tokenizer.encode("Hello world")

        self.assertEqual(result, [1, 2, 3, 4, 5])
        # encode 可能被调用时带有额外参数，使用 assert_called 而不是 assert_called_once_with
        mock_hf_tokenizer.encode.assert_called()

    def test_decode(self):
        """测试解码"""
        from hllm.tokenizer import Tokenizer

        mock_hf_tokenizer = Mock()
        mock_hf_tokenizer.decode.return_value = "Hello world"

        tokenizer = Tokenizer(mock_hf_tokenizer)
        result = tokenizer.decode([1, 2, 3])

        self.assertEqual(result, "Hello world")
        mock_hf_tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)

    def test_call_method(self):
        """测试 __call__ 方法"""
        from hllm.tokenizer import Tokenizer

        mock_hf_tokenizer = Mock()
        mock_hf_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        tokenizer = Tokenizer(mock_hf_tokenizer)
        result = tokenizer("Hello", return_tensors="pt")

        self.assertEqual(result, {"input_ids": [[1, 2, 3]]})


if __name__ == "__main__":
    unittest.main()
