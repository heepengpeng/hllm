"""
测试 generate 模块
"""

import pytest
import torch
from unittest.mock import Mock


class TestHelperFunctions:
    """测试辅助函数"""

    def test_apply_repetition_penalty(self):
        """测试重复惩罚函数"""
        from hllm.generate import _apply_repetition_penalty

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        generated_ids = torch.tensor([[1, 2]])
        penalty = 2.0

        _apply_repetition_penalty(logits, generated_ids, penalty)

        assert logits[0, 0].item() == 1.0
        assert logits[0, 1].item() == 1.0  # 2.0 / 2.0

    def test_top_k_filtering_basic(self):
        """测试 top-k 过滤"""
        from hllm.generate import _top_k_filtering

        # [1.0, 5.0, 3.0, 2.0, 4.0] -> top 2 是 5.0 和 4.0
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = _top_k_filtering(logits, top_k=2)

        # 位置 1 (5.0) 和 位置 4 (4.0) 应该保留
        assert not torch.isinf(filtered[0, 1]).item()  # 5.0
        assert not torch.isinf(filtered[0, 4]).item()  # 4.0
        # 其他应该被过滤
        assert torch.isinf(filtered[0, 0]).item()  # 1.0
        assert torch.isinf(filtered[0, 2]).item()  # 3.0
        assert torch.isinf(filtered[0, 3]).item()  # 2.0

    def test_top_k_filtering_large_k(self):
        """测试 k 大于词汇表大小"""
        from hllm.generate import _top_k_filtering

        logits = torch.tensor([[1.0, 5.0, 3.0]])
        filtered = _top_k_filtering(logits, top_k=100)
        assert not torch.isinf(filtered).any().item()

    def test_top_p_filtering(self):
        """测试 top-p 过滤"""
        from hllm.generate import _top_p_filtering

        # 第一个 token 有最高概率
        logits = torch.tensor([[10.0, 1.0, 1.0]])
        filtered = _top_p_filtering(logits, top_p=0.9)
        # 第一个 token 应该保留
        assert not torch.isinf(filtered[0, 0]).item()

    def test_top_p_one_keeps_all(self):
        """测试 top_p=1 不过滤"""
        from hllm.generate import _top_p_filtering

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_p_filtering(logits, top_p=1.0)
        assert not torch.isinf(filtered).any().item()


class TestRepetitionPenaltyEdge:
    """测试重复惩罚边界情况"""

    def test_penalty_one_no_change(self):
        """测试 penalty=1.0 不改变 logits"""
        from hllm.generate import _apply_repetition_penalty

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        generated_ids = torch.tensor([[1]])

        _apply_repetition_penalty(logits, generated_ids, 1.0)
        assert logits[0, 0].item() == 1.0
        assert logits[0, 1].item() == 2.0

    def test_penalty_greater_than_one(self):
        """测试 penalty > 1 降低重复 token"""
        from hllm.generate import _apply_repetition_penalty

        logits = torch.tensor([[1.0, 5.0, 3.0]])
        generated_ids = torch.tensor([[1]])

        _apply_repetition_penalty(logits, generated_ids, 2.0)
        assert logits[0, 1].item() < 5.0

    def test_multiple_repeated_tokens(self):
        """测试多个重复 token"""
        from hllm.generate import _apply_repetition_penalty

        logits = torch.tensor([[3.0, 3.0, 3.0]])
        generated_ids = torch.tensor([[0, 1]])

        _apply_repetition_penalty(logits, generated_ids, 2.0)
        assert logits[0, 0].item() < 3.0
        assert logits[0, 1].item() < 3.0
        assert logits[0, 2].item() == 3.0


class TestTopKEdgeCases:
    """测试 top-k 边界情况"""

    def test_top_k_equals_vocab(self):
        """测试 k 等于词汇表大小"""
        from hllm.generate import _top_k_filtering

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_k_filtering(logits, top_k=3)
        assert not torch.isinf(filtered).any().item()

    def test_top_k_equals_one(self):
        """测试 k=1 只保留最大值"""
        from hllm.generate import _top_k_filtering

        logits = torch.tensor([[1.0, 5.0, 3.0]])
        filtered = _top_k_filtering(logits, top_k=1)
        inf_count = sum(1 for i in range(3) if torch.isinf(filtered[0, i]).item())
        assert inf_count == 2


class TestTopPEdgeCases:
    """测试 top-p 边界情况"""

    def test_top_p_small_value(self):
        """测试小的 top_p 值"""
        from hllm.generate import _top_p_filtering

        logits = torch.tensor([[10.0, 1.0, 1.0]])
        filtered = _top_p_filtering(logits, top_p=0.1)
        inf_count = sum(1 for i in range(3) if torch.isinf(filtered[0, i]).item())
        assert inf_count > 0


class TestGenerateWithMocks:
    """使用 Mock 测试 generate 函数"""

    def test_generate_calls_tokenizer(self, mock_tokenizer, mock_model):
        """测试 generate 调用 tokenizer"""
        from hllm.generate import generate

        result = generate(mock_model, mock_tokenizer, "test", max_new_tokens=10, device="cpu")
        mock_tokenizer.encode.assert_called()
        assert result is not None

    def test_generate_calls_model(self, mock_tokenizer, mock_model):
        """测试 generate 调用 model"""
        from hllm.generate import generate

        result = generate(mock_model, mock_tokenizer, "test", max_new_tokens=5, device="cpu")
        mock_model.assert_called()
        assert result is not None

    def test_stream_generate_yields(self, mock_tokenizer, mock_model):
        """测试流式生成 yield tokens"""
        from hllm.generate import stream_generate

        tokens = list(stream_generate(mock_model, mock_tokenizer, "test", max_new_tokens=3, device="cpu"))
        assert isinstance(tokens, list)


class TestGenerateModuleExports:
    """测试模块导出"""

    def test_generate_exists(self):
        """测试 generate 函数存在"""
        from hllm.generate import generate
        assert callable(generate)

    def test_stream_generate_exists(self):
        """测试 stream_generate 函数存在"""
        from hllm.generate import stream_generate
        assert callable(stream_generate)
