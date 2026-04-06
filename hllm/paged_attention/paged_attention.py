"""PagedAttention: 基于 Block 的 Attention 计算"""

import logging
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def _get_torch():
    """获取 torch 模块，支持 mock"""
    import torch
    import torch.nn.functional as F
    return torch, F


def _check_flash_attn_available() -> bool:
    """检查是否可用 Flash Attention"""
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False


class PagedAttention:
    """
    Paged Attention 实现

    核心思想：
    1. 根据 block table 从物理块中 gather KV cache
    2. 计算 attention (使用 flash attention 或标准实现)
    3. 支持 continuous batching

    Args:
        num_heads: Query heads 数量
        num_kv_heads: Key/Value heads 数量 (GQA)
        head_dim: Head 维度
        scale: Attention scale (默认 1/sqrt(head_dim))
        block_size: 每个 block 的 token 数量
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        scale: Optional[float] = None,
        block_size: int = 16,
        use_flash_attn: bool = True,
        torch_module: Any = None,
        functional_module: Any = None,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)
        self.block_size = block_size
        self._torch = torch_module
        self._F = functional_module

        if self._torch is None:
            self._torch, self._F = _get_torch()

        self.use_flash_attn = use_flash_attn and _check_flash_attn_available()

        if self.use_flash_attn:
            logger.info("Using Flash Attention for PagedAttention")
        else:
            logger.info("Using standard attention for PagedAttention")

    def forward(
        self,
        query: Any,
        k_cache: Any,
        v_cache: Any,
        block_tables: Any,
        seq_lengths: Any,
        max_seq_len: int
    ) -> Any:
        """
        前向传播

        Args:
            query: [batch_size, num_heads, head_dim]
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_tables: [batch_size, max_num_blocks] 块表
            seq_lengths: [batch_size] 序列长度
            max_seq_len: 最大序列长度

        Returns:
            output: [batch_size, num_heads, head_dim]
        """
        batch_size = query.shape[0]

        # Gather KV cache from blocks
        key, value = self._gather_kv_cache(
            k_cache, v_cache, block_tables, seq_lengths, max_seq_len
        )

        # 计算 attention
        if self.use_flash_attn:
            output = self._flash_attention(query, key, value)
        else:
            output = self._standard_attention(query, key, value, seq_lengths)

        return output

    def _gather_kv_cache(
        self,
        k_cache: Any,
        v_cache: Any,
        block_tables: Any,
        seq_lengths: Any,
        max_seq_len: int
    ) -> Tuple[Any, Any]:
        """
        根据 block table gather KV cache

        Args:
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_tables: [batch_size, max_num_blocks]
            seq_lengths: [batch_size]
            max_seq_len: 最大序列长度

        Returns:
            key: [batch_size, max_seq_len, num_kv_heads, head_dim]
            value: [batch_size, max_seq_len, num_kv_heads, head_dim]
        """
        torch = self._torch
        batch_size = block_tables.shape[0]
        num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
        max_num_blocks = block_tables.shape[1]

        # 准备输出
        key = torch.zeros(
            (batch_size, max_seq_len, num_kv_heads, head_dim),
            dtype=k_cache.dtype, device=k_cache.device
        )
        value = torch.zeros(
            (batch_size, max_seq_len, num_kv_heads, head_dim),
            dtype=v_cache.dtype, device=v_cache.device
        )

        # 逐个序列 gather
        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            blocks = block_tables[i]

            pos = 0
            for block_id in blocks:
                if block_id < 0 or pos >= seq_len:
                    break

                block_id = block_id.item()
                if block_id >= num_blocks:
                    break

                # 计算这个块需要复制多少 token
                remaining = seq_len - pos
                tokens_to_copy = min(block_size, remaining)

                # 从物理块复制到序列位置
                key[i, pos:pos+tokens_to_copy] = k_cache[block_id, :tokens_to_copy]
                value[i, pos:pos+tokens_to_copy] = v_cache[block_id, :tokens_to_copy]

                pos += tokens_to_copy

        return key, value

    def _flash_attention(
        self,
        query: Any,
        key: Any,
        value: Any
    ) -> Any:
        """使用 Flash Attention"""
        try:
            from flash_attn import flash_attn_func
            # query: [batch, heads, dim] -> [batch, 1, heads, dim]
            q = query.unsqueeze(1)
            k = key.unsqueeze(1)
            v = value.unsqueeze(1)

            # Flash attention 需要 [batch, seqlen, heads, dim]
            out = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=True)
            return out.squeeze(1)  # [batch, heads, dim]
        except Exception as e:
            logger.warning(f"Flash attention failed: {e}, falling back to standard")
            return self._standard_attention(query, key, value, None)

    def _standard_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        seq_lengths: Optional[Any]
    ) -> Any:
        """标准 attention 实现"""
        torch = self._torch
        F = self._F

        batch_size, num_heads, head_dim = query.shape
        _, seq_len, num_kv_heads, _ = key.shape

        # 扩展 query heads (GQA)
        if num_heads != num_kv_heads:
            # GQA: 重复 kv heads
            num_repeat = num_heads // num_kv_heads
            key = key.unsqueeze(2).repeat(1, 1, num_repeat, 1, 1)
            key = key.view(batch_size, seq_len, num_heads, head_dim)
            value = value.unsqueeze(2).repeat(1, 1, num_repeat, 1, 1)
            value = value.view(batch_size, seq_len, num_heads, head_dim)

        # 转置用于矩阵乘法
        q = query.unsqueeze(2)  # [batch, num_heads, 1, head_dim]
        k = key.transpose(1, 2).transpose(2, 3)  # [batch, num_heads, head_dim, seq_len]
        v = value.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        # 计算 attention scores
        scores = torch.matmul(q, k) * self.scale  # [batch, num_heads, 1, seq_len]

        # 应用 causal mask
        if seq_lengths is not None:
            # 创建 mask
            mask = torch.arange(seq_len, device=query.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask, float('-inf'))
        else:
            # 简单的 causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 计算输出
        output = torch.matmul(attn_weights, v)  # [batch, num_heads, 1, head_dim]
        output = output.squeeze(2)  # [batch, num_heads, head_dim]

        return output

    def update_cache(
        self,
        k_cache: Any,
        v_cache: Any,
        new_key: Any,
        new_value: Any,
        block_tables: List[List[int]],
        seq_positions: List[int]
    ):
        """
        更新 KV cache（存储新生成的 token）

        Args:
            k_cache: 物理 K cache
            v_cache: 物理 V cache
            new_key: [batch, num_kv_heads, head_dim] 新的 key
            new_value: [batch, num_kv_heads, head_dim] 新的 value
            block_tables: 每个序列的块表
            seq_positions: 每个序列的当前位置
        """
        batch_size = new_key.shape[0]

        for i in range(batch_size):
            pos = seq_positions[i]
            blocks = block_tables[i]

            # 计算应该存储到哪个 block
            block_idx = pos // self.block_size
            offset = pos % self.block_size

            if block_idx >= len(blocks):
                logger.warning(f"Block index {block_idx} out of range for seq {i}")
                continue

            block_id = blocks[block_idx]

            # 写入 cache
            k_cache[block_id, offset] = new_key[i]
            v_cache[block_id, offset] = new_value[i]
