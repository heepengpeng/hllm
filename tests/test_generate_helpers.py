"""
测试 generate.py 辅助函数
"""

import unittest
import torch


class TestRepetitionPenalty(unittest.TestCase):
    """测试重复惩罚函数"""

    def test_apply_repetition_penalty(self):
        """测试应用重复惩罚"""
        from hllm.generate import _apply_repetition_penalty

        # 创建测试数据
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        generated_ids = torch.tensor([[0, 1, 2]])
        penalty = 2.0

        original_logits = logits.clone()
        _apply_repetition_penalty(logits, generated_ids, penalty)

        # 验证惩罚已应用
        # token 0, 1, 2 应该被惩罚（除以 penalty）
        self.assertLess(logits[0, 0], original_logits[0, 0])
        self.assertLess(logits[0, 1], original_logits[0, 1])
        self.assertLess(logits[0, 2], original_logits[0, 2])


class TestTopKFiltering(unittest.TestCase):
    """测试 Top-k 过滤"""

    def test_top_k_filtering_basic(self):
        """测试基本 Top-k 过滤"""
        from hllm.generate import _top_k_filtering

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        top_k = 2

        result = _top_k_filtering(logits, top_k)

        # 检查只有 top-2 的值不是 -inf
        self.assertEqual(result[0, 4].item(), 5.0)  # 最大值
        self.assertEqual(result[0, 3].item(), 4.0)  # 第二大值
        self.assertEqual(result[0, 2].item(), float("-inf"))
        self.assertEqual(result[0, 1].item(), float("-inf"))
        self.assertEqual(result[0, 0].item(), float("-inf"))


class TestTopPFiltering(unittest.TestCase):
    """测试 Top-p (nucleus) 过滤"""

    def test_top_p_filtering_basic(self):
        """测试基本 Top-p 过滤"""
        from hllm.generate import _top_p_filtering

        # 创建概率分布明确的 logits
        # softmax([5.0, 4.0, 1.0, 0.0]) ≈ [0.64, 0.24, 0.08, 0.04]
        logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
        top_p = 0.8  # 应该保留前两个（累计约 0.88）

        result = _top_p_filtering(logits, top_p)

        # 验证过滤结果
        # 至少最大值应该保留
        self.assertNotEqual(result[0, 0].item(), float("-inf"))


if __name__ == "__main__":
    unittest.main()
