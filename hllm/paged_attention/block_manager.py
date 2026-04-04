"""BlockManager: 管理 KV Cache 的物理块分配"""

import torch
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """物理块，存储实际的 KV Cache"""
    block_id: int
    num_tokens: int = 0  # 当前块中存储的 token 数量
    ref_count: int = 0   # 引用计数，用于 Copy-on-Write
    
    def is_empty(self) -> bool:
        return self.num_tokens == 0
    
    def is_full(self, block_size: int) -> bool:
        return self.num_tokens >= block_size


@dataclass  
class Sequence:
    """序列状态，记录每个请求的状态"""
    seq_id: int
    prompt_len: int
    generated_len: int = 0
    block_table: List[int] = field(default_factory=list)  # 逻辑块到物理块的映射
    status: str = "waiting"  # waiting/running/swapped
    
    def get_len(self) -> int:
        return self.prompt_len + self.generated_len
    
    def get_num_blocks(self, block_size: int) -> int:
        """计算需要的块数量"""
        return (self.get_len() + block_size - 1) // block_size


class BlockManager:
    """
    管理物理块的分配和释放
    
    核心功能：
    1. 维护空闲块池
    2. 为序列分配/释放块
    3. 支持块共享 (Copy-on-Write)
    4. 动态扩展序列的块
    
    Args:
        num_blocks: 物理块总数
        block_size: 每个块能存储的 token 数量 (默认 16)
        num_kv_heads: KV heads 数量
        head_dim: 每个 head 的维度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # 初始化物理块池
        self.blocks: Dict[int, Block] = {
            i: Block(block_id=i) for i in range(num_blocks)
        }
        
        # 空闲块列表
        self.free_blocks: Set[int] = set(range(num_blocks))
        
        # 序列状态管理
        self.sequences: Dict[int, Sequence] = {}
        self.next_seq_id = 0
        
        # 预分配 KV Cache 物理存储
        # shape: [num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache = torch.empty(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.empty(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype, device=device
        )
        
        logger.info(
            f"BlockManager initialized: "
            f"{num_blocks} blocks × {block_size} tokens = "
            f"{num_blocks * block_size} max tokens, "
            f"{(num_blocks * block_size * num_kv_heads * head_dim * 2 * 2) / 1024**3:.2f} GB"
        )
    
    def create_sequence(self, prompt_len: int) -> int:
        """
        创建新序列，分配初始块
        
        Args:
            prompt_len: prompt 长度
            
        Returns:
            seq_id: 序列 ID
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        seq = Sequence(
            seq_id=seq_id,
            prompt_len=prompt_len,
            status="waiting"
        )
        
        # 为 prompt 分配块
        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        block_ids = self._allocate_blocks(num_blocks_needed)
        
        if block_ids is None:
            raise RuntimeError(
                f"Out of memory: need {num_blocks_needed} blocks, "
                f"but only {len(self.free_blocks)} available"
            )
        
        seq.block_table = block_ids
        self.sequences[seq_id] = seq
        
        logger.debug(f"Created sequence {seq_id}, allocated {len(block_ids)} blocks")
        return seq_id
    
    def free_sequence(self, seq_id: int):
        """释放序列及其占用的块"""
        if seq_id not in self.sequences:
            return
        
        seq = self.sequences[seq_id]
        
        # 释放所有块
        for block_id in seq.block_table:
            self._free_block(block_id)
        
        del self.sequences[seq_id]
        logger.debug(f"Freed sequence {seq_id}")
    
    def append_token(self, seq_id: int) -> bool:
        """
        为序列追加一个 token，必要时分配新块
        
        Args:
            seq_id: 序列 ID
            
        Returns:
            是否成功
        """
        if seq_id not in self.sequences:
            return False
        
        seq = self.sequences[seq_id]
        seq.generated_len += 1
        
        total_len = seq.get_len()
        num_blocks = (total_len + self.block_size - 1) // self.block_size
        
        # 需要分配新块
        if num_blocks > len(seq.block_table):
            new_block_id = self._allocate_block()
            if new_block_id is None:
                logger.warning(f"Cannot allocate new block for seq {seq_id}")
                seq.generated_len -= 1  # 回滚
                return False
            seq.block_table.append(new_block_id)
            logger.debug(f"Allocated new block {new_block_id} for seq {seq_id}")
        
        # 更新最后一块的 token 计数
        last_block_id = seq.block_table[-1]
        block = self.blocks[last_block_id]
        block.num_tokens = min(total_len % self.block_size, self.block_size)
        if block.num_tokens == 0:
            block.num_tokens = self.block_size
        
        return True
    
    def get_block_table(self, seq_id: int) -> List[int]:
        """获取序列的块表"""
        if seq_id not in self.sequences:
            return []
        return self.sequences[seq_id].block_table
    
    def get_seq_position(self, seq_id: int) -> int:
        """获取序列当前长度（用于计算 attention）"""
        if seq_id not in self.sequences:
            return 0
        return self.sequences[seq_id].get_len()
    
    def can_allocate(self, prompt_len: int) -> bool:
        """检查是否有足够内存分配给新序列"""
        num_blocks_needed = (prompt_len + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= num_blocks_needed
    
    def get_num_free_blocks(self) -> int:
        """获取空闲块数量"""
        return len(self.free_blocks)
    
    def get_num_running_seqs(self) -> int:
        """获取运行中的序列数量"""
        return sum(1 for seq in self.sequences.values() if seq.status == "running")
    
    def _allocate_blocks(self, num_blocks: int) -> Optional[List[int]]:
        """分配多个块"""
        if len(self.free_blocks) < num_blocks:
            return None
        
        block_ids = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            self.blocks[block_id].ref_count = 1
            self.blocks[block_id].num_tokens = 0
            block_ids.append(block_id)
        
        return block_ids
    
    def _allocate_block(self) -> Optional[int]:
        """分配单个块"""
        if not self.free_blocks:
            return None
        
        block_id = self.free_blocks.pop()
        self.blocks[block_id].ref_count = 1
        self.blocks[block_id].num_tokens = 0
        return block_id
    
    def _free_block(self, block_id: int):
        """释放块"""
        if block_id in self.blocks:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count <= 0:
                self.blocks[block_id].ref_count = 0
                self.blocks[block_id].num_tokens = 0
                self.free_blocks.add(block_id)
    
    def get_cache_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 KV Cache 张量"""
        return self.k_cache, self.v_cache
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free_blocks": len(self.free_blocks),
            "used_blocks": self.num_blocks - len(self.free_blocks),
            "num_sequences": len(self.sequences),
            "running_sequences": self.get_num_running_seqs(),
            "memory_gb": (self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim * 2 * 2) / 1024**3
        }
