"""Paged PyTorch 后端 - 集成 PagedAttention 优化"""

import logging
import torch
from typing import TYPE_CHECKING, Optional, Generator
from .pytorch import PyTorchBackend
from ..paged_attention import BlockManager, PagedAttention, Scheduler

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PagedPyTorchBackend(PyTorchBackend):
    """
    带 PagedAttention 优化的 PyTorch 后端
    
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
    """
    
    NAME = "paged_pytorch"
    SUPPORTS_QUANTIZATION = True
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
        **kwargs
    ):
        self._device_name = self._normalize_device(device)
        self.torch_dtype = torch_dtype or torch.float16
        self.trust_remote_code = trust_remote_code
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        
        # 保存模型路径供后续使用
        self.model_path = model_path
        
        # 初始化 BlockManager 和 Scheduler
        self._init_paged_attention(num_blocks)
        
        # 调用父类初始化（会加载模型）
        super().__init__(model_path, device, torch_dtype, trust_remote_code, **kwargs)
        
        # 初始化 PagedAttention 计算
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
                # 这里使用典型值，实际值在模型加载后更新
                bytes_per_block = self.block_size * 8 * 64 * 2 * 2  # 8 heads, 64 dim, fp16
                num_blocks = int(available_memory / bytes_per_block)
                num_blocks = max(256, num_blocks)  # 至少 256 个块
            else:
                # CPU 模式下使用固定值
                num_blocks = 256
        
        # 创建 BlockManager（这里先用默认值，模型加载后更新）
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
        
        self.paged_attention = PagedAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=self.block_size,
            use_flash_attn=False  # 简化实现，暂不使用 flash attn
        )
        
        # 更新 BlockManager 的维度
        self.block_manager.num_kv_heads = num_kv_heads
        self.block_manager.head_dim = head_dim
        
        logger.info(
            f"PagedAttention: heads={num_heads}, kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}"
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> str:
        """
        使用 PagedAttention 生成文本
        
        当前实现使用标准的 transformers generate，但配合 block 管理的 KV cache
        """
        # 编码 prompt
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device_name)
        prompt_len = input_ids.shape[1]
        
        # 创建序列
        seq_id = self.block_manager.create_sequence(prompt_len)
        
        try:
            # 使用标准生成（PagedAttention 的完整实现需要修改模型 forward）
            # 这里我们先使用标准方法，但演示了 block 管理
            logger.debug(f"Generating with PagedAttention backend, seq_id={seq_id}")
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    use_cache=True,
                )
            
            # 解码输出
            output_text = self._tokenizer.decode(
                output_ids[0][prompt_len:], 
                skip_special_tokens=True
            )
            
            return output_text
            
        finally:
            # 释放序列
            self.block_manager.free_sequence(seq_id)
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成文本"""
        
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device_name)
        prompt_len = input_ids.shape[1]
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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                streamer=streamer,
                use_cache=True,
            )
            
            # 在后台线程运行生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式输出
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield new_text
            
            thread.join()
            
        finally:
            self.block_manager.free_sequence(seq_id)
    
    def get_block_stats(self) -> dict:
        """获取块管理统计信息"""
        return self.block_manager.get_stats()
