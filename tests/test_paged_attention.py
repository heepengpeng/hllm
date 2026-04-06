"""
测试 PagedAttention 模块 - Mock 覆盖
"""

import pytest
from unittest.mock import Mock, MagicMock, PropertyMock
import sys


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
        assert pa.num_heads == 32
        assert pa.block_size == 16

    def test_paged_attention_init_params(self):
        """测试带参数初始化"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(num_heads=16, head_dim=128, block_size=32)
        assert pa.num_heads == 16
        assert pa.head_dim == 128
        assert pa.block_size == 32

    def test_paged_attention_has_forward(self):
        """测试有 forward 方法"""
        from hllm.paged_attention.paged_attention import PagedAttention
        assert hasattr(PagedAttention, 'forward')

    def test_paged_attention_has_gather(self):
        """测试有 _gather_kv_cache 方法"""
        from hllm.paged_attention.paged_attention import PagedAttention
        assert hasattr(PagedAttention, '_gather_kv_cache')

    def test_paged_attention_has_standard_attention(self):
        """测试有 _standard_attention 方法"""
        from hllm.paged_attention.paged_attention import PagedAttention
        assert hasattr(PagedAttention, '_standard_attention')

    def test_paged_attention_scale(self):
        """测试 scale 计算"""
        from hllm.paged_attention.paged_attention import PagedAttention

        pa = PagedAttention(head_dim=64)
        assert pa.scale is not None
        assert pa.scale > 0

    def test_paged_attention_with_mock_torch(self):
        """测试使用 mock torch 模块"""
        from hllm.paged_attention.paged_attention import PagedAttention

        mock_torch = Mock()
        mock_torch.float16 = Mock()
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.matmul = Mock(return_value=Mock())
        mock_torch.arange = Mock(return_value=Mock())
        mock_torch.triu = Mock(return_value=Mock())
        mock_torch.ones = Mock(return_value=Mock())

        mock_F = Mock()
        mock_F.softmax = Mock(return_value=Mock())

        pa = PagedAttention(
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            torch_module=mock_torch,
            functional_module=mock_F,
            use_flash_attn=False
        )
        assert pa._torch == mock_torch
        assert pa._F == mock_F


class TestPagedAttentionForward:
    """测试 PagedAttention forward 方法"""

    def test_forward_with_mock_tensors(self):
        """测试 forward 使用 mock tensors"""
        from hllm.paged_attention.paged_attention import PagedAttention

        # 创建 mock torch
        mock_torch = Mock()

        # Mock tensors
        mock_query = Mock()
        mock_query.shape = [2, 4, 16]  # batch=2, heads=4, dim=16
        mock_query.device = "cpu"
        mock_query.unsqueeze = Mock(return_value=Mock())

        mock_k_cache = Mock()
        mock_k_cache.shape = [10, 16, 2, 16]
        mock_k_cache.dtype = Mock()
        mock_k_cache.device = "cpu"

        mock_v_cache = Mock()
        mock_v_cache.shape = [10, 16, 2, 16]
        mock_v_cache.dtype = Mock()
        mock_v_cache.device = "cpu"

        mock_block_tables = Mock()
        mock_block_tables.shape = [2, 4]
        mock_block_tables.__getitem__ = Mock(side_effect=lambda i: Mock(item=Mock(return_value=i)))

        mock_seq_lengths = Mock()
        mock_seq_lengths.shape = [2]
        mock_seq_lengths.__getitem__ = Mock(side_effect=lambda i: Mock(item=Mock(return_value=10)))

        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.ones = Mock(return_value=Mock())

        pa = PagedAttention(
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            torch_module=mock_torch,
            use_flash_attn=False
        )

        # Mock _gather_kv_cache 返回值
        mock_key = Mock()
        mock_value = Mock()
        pa._gather_kv_cache = Mock(return_value=(mock_key, mock_value))
        pa._standard_attention = Mock(return_value=Mock())

        # 调用 forward
        result = pa.forward(
            mock_query, mock_k_cache, mock_v_cache,
            mock_block_tables, mock_seq_lengths, max_seq_len=32
        )

        # 验证调用
        pa._gather_kv_cache.assert_called_once()
        pa._standard_attention.assert_called_once()


class TestBlockManagerCore:
    """测试 BlockManager 核心"""

    def test_block_manager_class_exists(self):
        """测试 BlockManager 类存在"""
        from hllm.paged_attention.block_manager import BlockManager
        assert callable(BlockManager)

    def test_block_manager_init_default(self):
        """测试默认初始化"""
        from hllm.paged_attention.block_manager import BlockManager

        # 使用 mock torch 避免实际分配
        mock_torch = Mock()
        mock_torch.float16 = "float16"
        mock_torch.empty = Mock(return_value=Mock())

        bm = BlockManager(
            num_blocks=100,
            block_size=16,
            torch_module=mock_torch
        )
        assert bm is not None
        assert bm.num_blocks == 100
        assert bm.block_size == 16
        assert bm.k_cache is None  # 延迟分配

    def test_block_manager_has_create_sequence(self):
        """测试有 create_sequence 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'create_sequence')

    def test_block_manager_has_free_sequence(self):
        """测试有 free_sequence 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'free_sequence')

    def test_block_manager_has_can_allocate(self):
        """测试有 can_allocate 方法"""
        from hllm.paged_attention.block_manager import BlockManager
        assert hasattr(BlockManager, 'can_allocate')

    def test_block_manager_allocate_and_free(self):
        """测试分配和释放"""
        from hllm.paged_attention.block_manager import BlockManager

        mock_torch = Mock()
        mock_torch.float16 = "float16"

        bm = BlockManager(
            num_blocks=10,
            block_size=16,
            torch_module=mock_torch
        )

        # 测试分配 - 32 tokens 需要 2 blocks (ceil(32/16))
        seq_id = bm.create_sequence(prompt_len=32)
        assert seq_id == 0
        assert bm.get_num_free_blocks() == 8  # 10 - 2 = 8

        # 测试释放
        bm.free_sequence(seq_id)
        assert bm.get_num_free_blocks() == 10


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


class TestPagedAttentionModule:
    """测试模块导出"""

    def test_module_exports(self):
        """测试模块导出"""
        from hllm.paged_attention import PagedAttention, BlockManager, Scheduler
        assert callable(PagedAttention)
        assert callable(BlockManager)
        assert callable(Scheduler)

    def test_module_has_all(self):
        """测试 __all__ 定义"""
        from hllm.paged_attention import __all__
        assert "PagedAttention" in __all__
        assert "BlockManager" in __all__
        assert "Scheduler" in __all__
