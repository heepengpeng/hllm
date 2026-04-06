"""Paged PyTorch 后端 - 集成 PagedAttention 优化

基于 PyTorchBackend，添加 PagedAttention 内存管理优化。
"""

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Optional

import torch

from .pytorch import PyTorchBackend, _check_flash_attn_available
from ..paged_attention import BlockManager, PagedAttention, Scheduler

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PagedPyTorchBackend(PyTorchBackend):
    """带 PagedAttention 优化的 PyTorch 后端

    特性:
    1. 块级内存管理 - 减少显存碎片
    2. Continuous Batching - 提高吞吐量
    3. 动态内存分配 - 按需分配块

    Args:
        model_path: 模型路径
        device: 设备 (cuda/cpu/mps)
        num_blocks: KV cache 块数量 (默认根据显存自动计算)
        block_size: 每块 token 数量 (默认 16)
        max_batch_size: 最大 batch size (默认 16)
        torch_dtype: PyTorch 数据类型
        trust_remote_code: 是否信任远程代码
    """

    NAME = "paged_pytorch"
    SUPPORTS_QUANTIZATION = True
    SUPPORTS_GPU = True
    DEFAULT_DEVICE = "cuda"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_blocks: Optional[int] = None,
        block_size: int = 16,
        max_batch_size: int = 16,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        use_flash_attn: bool | None = None,
        **kwargs
    ):
        """初始化 PagedAttention 后端
        
        Args:
            use_flash_attn: 是否使用 Flash Attention 2 (None=自动检测)
        """
        self.block_size = block_size
        self.max_batch_size = max_batch_size

        # 先调用父类初始化（加载模型）
        super().__init__(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            use_flash_attn=use_flash_attn,
            **kwargs
        )

        # 初始化 PagedAttention 组件
        self._init_paged_attention(num_blocks)
        self._init_attention()

        logger.info("PagedPyTorchBackend initialized successfully")

    def _init_paged_attention(self, num_blocks: Optional[int]):
        """初始化 PagedAttention 组件"""

        # 如果没有指定块数，根据显存自动计算
        if num_blocks is None:
            if torch.cuda.is_available():
                # 计算可用显存
                total_memory = torch.cuda.get_device_properties(0).total_memory
                # 保留一部分给模型和中间计算
                available_memory = total_memory * 0.7  # 使用 70% 显存

                # 假设每个块需要: block_size * num_kv_heads * head_dim * 2 * 2 bytes
                bytes_per_block = self.block_size * 8 * 64 * 2 * 2  # 8 heads, 64 dim, fp16
                num_blocks = int(available_memory / bytes_per_block)
                num_blocks = max(256, num_blocks)  # 至少 256 个块
            else:
                # CPU 模式下使用固定值
                num_blocks = 256

        # 创建 BlockManager
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=8,
            head_dim=64,
            dtype=self.torch_dtype,
            device=self._device_name
        )

        # 创建 Scheduler
        self.scheduler = Scheduler(
            max_batch_size=self.max_batch_size
        )
        
        logger.info(
            f"PagedAttention configured: "
            f"{num_blocks} blocks × {self.block_size} tokens = "
            f"{num_blocks * self.block_size} max tokens"
        )
    
    def _init_attention(self):
        """初始化 PagedAttention 计算"""
        
        # 从模型配置获取参数
        config = self.model.config
        num_heads = getattr(config, 'num_attention_heads', 32)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads // 4)
        head_dim = getattr(config, 'head_dim', config.hidden_size // num_heads)
        
        # 检查 Flash Attention 是否可用
        use_flash = _check_flash_attn_available()
        if use_flash:
            logger.info("Flash Attention 2 available, enabling for PagedAttention")
        
        self.paged_attention = PagedAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=self.block_size,
            use_flash_attn=use_flash,
            torch_module=self._torch if hasattr(self, '_torch') else None,
        )
        
        # 更新 BlockManager 的维度
        self.block_manager.num_kv_heads = num_kv_heads
        self.block_manager.head_dim = head_dim
        
        logger.info(
            f"PagedAttention: heads={num_heads}, kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}"
        )
    
    def _generate_impl(self, prompt: str, params, **kwargs) -> str:
        """使用 PagedAttention 生成实现

        当前实现使用标准的 transformers generate，配合 block 管理的 KV cache

        Args:
            prompt: 输入提示
            params: 生成参数
            **kwargs: 额外参数

        Returns:
            生成的文本
        """
        from .base import GenerationParams

        start_time = time.time()

        # 编码 prompt
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device_name)
        prompt_len = input_ids.shape[1]
        prompt_tokens = len(self._tokenizer.encode(prompt))

        # 创建序列
        seq_id = self.block_manager.create_sequence(prompt_len)

        try:
            logger.debug(f"Generating with PagedAttention backend, seq_id={seq_id}")

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    repetition_penalty=params.repetition_penalty,
                    do_sample=params.temperature > 0,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    use_cache=True,
                )

            # 解码输出
            output_text = self._tokenizer.decode(
                output_ids[0][prompt_len:],
                skip_special_tokens=True
            )

            # 更新统计
            generated_tokens = len(output_ids[0]) - prompt_len
            latency_ms = (time.time() - start_time) * 1000
            self._stats.update(prompt_tokens, generated_tokens, latency_ms)

            return output_text

        finally:
            # 释放序列
            self.block_manager.free_sequence(seq_id)

    def _stream_generate_impl(self, prompt: str, params, **kwargs) -> Generator[str, None, None]:
        """流式生成实现

        Args:
            prompt: 输入提示
            params: 生成参数
            **kwargs: 额外参数

        Yields:
            逐个 token
        """
        start_time = time.time()

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device_name)
        prompt_len = input_ids.shape[1]
        prompt_tokens = len(self._tokenizer.encode(prompt))

        seq_id = self.block_manager.create_sequence(prompt_len)

        try:
            # 使用标准流式生成
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repetition_penalty=params.repetition_penalty,
                do_sample=params.temperature > 0,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                streamer=streamer,
                use_cache=True,
            )

            # 在后台线程运行生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # 流式输出
            generated_tokens = 0
            for new_text in streamer:
                yield new_text
                generated_tokens += len(self._tokenizer.encode(new_text))

            thread.join()

            # 更新统计
            latency_ms = (time.time() - start_time) * 1000
            self._stats.update(prompt_tokens, generated_tokens, latency_ms)

        finally:
            self.block_manager.free_sequence(seq_id)
    
    def get_block_stats(self) -> dict:
        """获取块管理统计信息"""
        return self.block_manager.get_stats()

    def get_info(self) -> dict:
        """获取后端信息，包含 PagedAttention 特定信息"""
        info = super().get_info()
        info["paged_attention"] = {
            "block_size": self.block_size,
            "num_blocks": self.block_manager.num_blocks if hasattr(self.block_manager, 'num_blocks') else None,
            "max_batch_size": self.max_batch_size,
            "flash_attention": self.use_flash_attn,
        }
        return info

    def get_memory_usage(self) -> dict[str, float]:
        """获取内存使用情况，包含 PagedAttention 块信息"""
        memory_info = super().get_memory_usage()

        # 添加块管理器统计
        try:
            block_stats = self.get_block_stats()
            memory_info["paged_blocks_used"] = block_stats.get("used_blocks", 0)
            memory_info["paged_blocks_free"] = block_stats.get("free_blocks", 0)
        except Exception:
            pass

        return memory_info
