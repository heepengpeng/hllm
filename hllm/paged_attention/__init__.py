"""PagedAttention 优化模块

实现类似 vLLM 的 PagedAttention 机制，提升 GPU 推理吞吐量和显存效率。
"""

from .block_manager import BlockManager
from .paged_attention import PagedAttention, _check_flash_attn_available
from .scheduler import Scheduler

__all__ = ["BlockManager", "PagedAttention", "Scheduler", "_check_flash_attn_available"]
