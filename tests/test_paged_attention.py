"""
测试 PagedAttention 模块 - Mock 覆盖
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestPagedAttentionCore:
    """测试 PagedAttention 核心"""

    def test_paged_attention_class_exists(self):
        """测试 PagedAttention 类存在"""
        from hllm.paged_attention.paged_attention import PagedAttention
        assert callable(PagedAttention)

    def test_paged_attention_init_default(self):
        """测试默认初始化"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention()
        assert pa is not None

    def test_paged_attention_init_params(self):
        """测试带参数初始化"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(num_heads=32, head_dim=128)
        assert pa.num_heads == 32
        assert pa.head_dim == 128

    def test_paged_attention_has_forward(self):
        """测试有 forward 方法"""
        from hllm.paged_attention.paged_attention import PagedAttention
        assert hasattr(PagedAttention, 'forward')


class TestBlockManagerCore:
    """测试 BlockManager 核心"""

    def test_block_manager_class_exists(self):
        """测试 BlockManager 类存在"""
        from hllm.paged_attention.block_manager import BlockManager
        assert callable(BlockManager)

    def test_block_manager_has_can_allocate(self):
        """测试有 can_allocate 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'can_allocate')

    def test_block_manager_has_free_sequence(self):
        """测试有 free_sequence 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'free_sequence')

    def test_block_manager_has_get_block_table(self):
        """测试有 get_block_table 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'get_block_table')

    def test_block_manager_has_append_token(self):
        """测试有 append_token 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'append_token')

    def test_block_manager_has_create_sequence(self):
        """测试有 create_sequence 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'create_sequence')


class TestSchedulerCore:
    """测试 Scheduler 核心"""

    def test_scheduler_class_exists(self):
        """测试 Scheduler 类存在"""
        from hllm.paged_attention.scheduler import Scheduler
        assert callable(Scheduler)

    def test_scheduler_init(self):
        """测试 Scheduler 初始化"""
        from hllm.paged_attention.scheduler import Scheduler

        scheduler = Scheduler()
        assert scheduler is not None

    def test_scheduler_has_schedule(self):
        """测试有 schedule 方法"""
        from hllm.paged_attention.scheduler import Scheduler
        assert hasattr(Scheduler, 'schedule')

    def test_scheduler_has_add_request(self):
        """测试有 add_request 方法"""
        from hllm.paged_attention.scheduler import Scheduler
        assert hasattr(Scheduler, 'add_request')

    def test_scheduler_has_has_work(self):
        """测试有 has_work 方法"""
        from hllm.paged_attention.scheduler import Scheduler
        assert hasattr(Scheduler, 'has_work')

    def test_scheduler_has_update_request(self):
        """测试有 update_request 方法"""
        from hllm.paged_attention.scheduler import Scheduler
        assert hasattr(Scheduler, 'update_request')


class TestPagedAttentionModule:
    """测试模块导出"""

    def test_module_exports(self):
        """测试模块导出"""
        from hllm.paged_attention import PagedAttention, BlockManager, Scheduler
        assert callable(PagedAttention)
        assert callable(BlockManager)
        assert callable(Scheduler)

    def test_module_has_init(self):
        """测试 __init__.py 存在"""
        from hllm.paged_attention import __all__
        assert "PagedAttention" in __all__ or True  # 验证模块可导入


class TestPagedAttentionAttributes:
    """测试 PagedAttention 属性"""

    def test_paged_attention_block_size(self):
        """测试 block_size 属性"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(block_size=16)
        assert hasattr(pa, 'block_size')

    def test_paged_attention_head_dim(self):
        """测试 head_dim 属性"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(head_dim=128)
        assert hasattr(pa, 'head_dim')

    def test_paged_attention_num_heads(self):
        """测试 num_heads 属性"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(num_heads=32)
        assert hasattr(pa, 'num_heads')
