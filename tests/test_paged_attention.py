"""
测试 PagedAttention 模块
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestPagedAttention(unittest.TestCase):
    """测试 PagedAttention 核心"""

    def test_paged_attention_class_exists(self):
        """测试 PagedAttention 类存在"""
        from hllm.paged_attention.paged_attention import PagedAttention
        self.assertTrue(callable(PagedAttention))

    def test_paged_attention_init(self):
        """测试 PagedAttention 初始化"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(num_heads=32, head_dim=128)
        self.assertIsNotNone(pa)


class TestPagedAttentionAttributes(unittest.TestCase):
    """测试 PagedAttention 属性"""

    def test_paged_attention_attributes(self):
        """测试 PagedAttention 属性"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(num_heads=32, head_dim=128)
        self.assertTrue(hasattr(pa, 'num_heads'))
        self.assertTrue(hasattr(pa, 'head_dim'))


class TestBlockManager(unittest.TestCase):
    """测试 BlockManager"""

    def test_block_manager_class_exists(self):
        """测试 BlockManager 类存在"""
        from hllm.paged_attention.block_manager import BlockManager
        self.assertTrue(callable(BlockManager))


class TestScheduler(unittest.TestCase):
    """测试 Scheduler"""

    def test_scheduler_class_exists(self):
        """测试 Scheduler 类存在"""
        from hllm.paged_attention.scheduler import Scheduler
        self.assertTrue(callable(Scheduler))

    def test_scheduler_init(self):
        """测试 Scheduler 初始化"""
        from hllm.paged_attention.scheduler import Scheduler

        scheduler = Scheduler()
        self.assertIsNotNone(scheduler)


class TestModuleExports(unittest.TestCase):
    """测试模块导出"""

    def test_module_exports(self):
        """测试模块导出"""
        from hllm.paged_attention import PagedAttention, BlockManager, Scheduler
        self.assertTrue(callable(PagedAttention))
        self.assertTrue(callable(BlockManager))
        self.assertTrue(callable(Scheduler))


if __name__ == "__main__":
    unittest.main()
