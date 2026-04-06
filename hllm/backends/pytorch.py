"""PyTorch 后端实现"""

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseBackend, GenerationParams
from ..utils.model_downloader import ensure_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PyTorchBackend(BaseBackend):
    """PyTorch 推理后端

    支持 CPU、CUDA、MPS (Apple Silicon)。提供统一的 Transformer 模型推理。
    
    Example:
        >>> backend = PyTorchBackend("microsoft/Phi-3-mini-4k-instruct", device="cuda")
        >>> result = backend.generate("Hello, how are you?", max_new_tokens=50)
    """

    NAME = "pytorch"
    SUPPORTS_QUANTIZATION = True
    SUPPORTS_GPU = True
    DEFAULT_DEVICE = "cpu"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        """初始化 PyTorch 后端
        
        Args:
            model_path: 模型路径或 HuggingFace model ID
            device: 计算设备 (cpu/cuda/mps)
            torch_dtype: PyTorch 数据类型
            trust_remote_code: 是否信任远程代码
            **kwargs: 额外参数
        """
        self._device_name = self._normalize_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.trust_remote_code = trust_remote_code
        super().__init__(model_path, **kwargs)

    def _normalize_device(self, device: str) -> str:
        """标准化设备名称并验证可用性
        
        Args:
            device: 设备名称
            
        Returns:
            标准化后的设备名称
        """
        device = device.lower()
        
        if device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS not available, falling back to CPU")
                return "cpu"
            return device
            
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
            return device
            
        if device not in ("cpu", "auto"):
            logger.warning(f"Unknown device: {device}, using CPU")
            
        return "cpu"

    @property
    def device_name(self) -> str:
        """返回设备名称"""
        return self._device_name

    def _load_model(self, **kwargs) -> None:
        """加载 PyTorch 模型
        
        Args:
            **kwargs: 加载参数
                - quantization_config: 量化配置
                - attn_implementation: 注意力实现方式
                - compile: 是否使用 torch.compile
        """
        # 自动下载模型（如果不是本地路径）
        model_path = ensure_model(
            self.model_path,
            use_modelscope=kwargs.get("use_modelscope", True),
            use_hf_mirror=kwargs.get("use_hf_mirror", True),
            prefer_modelscope=kwargs.get("prefer_modelscope", True),
        )
        
        logger.info(f"Loading PyTorch model from {model_path} on {self.device_name}")

        # 加载分词器
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
        )
        
        # 确保有 pad_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 加载模型
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        # 支持量化配置
        if "quantization_config" in kwargs:
            load_kwargs["quantization_config"] = kwargs["quantization_config"]
        if "attn_implementation" in kwargs:
            load_kwargs["attn_implementation"] = kwargs["attn_implementation"]

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        self.model.to(self.device_name)
        self.model.eval()

        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, "compile") and kwargs.get("compile", False):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        logger.info(f"PyTorch model loaded successfully on {self.device_name}")

    def _generate_impl(self, prompt: str, params: GenerationParams, **kwargs) -> str:
        """生成实现
        
        Args:
            prompt: 输入提示
            params: 生成参数
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        import time
        
        start_time = time.time()
        prompt_tokens = len(self._tokenizer.encode(prompt))
        
        from ..generate import generate as _generate

        result = _generate(
            model=self.model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            stop_sequences=params.stop_sequences,
            device=self.device_name,
            **kwargs
        )
        
        # 更新统计
        generated_tokens = len(self._tokenizer.encode(result))
        latency_ms = (time.time() - start_time) * 1000
        self._stats.update(prompt_tokens, generated_tokens, latency_ms)
        
        return result

    def _stream_generate_impl(
        self, 
        prompt: str, 
        params: GenerationParams, 
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成实现
        
        Args:
            prompt: 输入提示
            params: 生成参数
            **kwargs: 额外参数
            
        Yields:
            逐个 token
        """
        start_time = time.time()
        prompt_tokens = len(self._tokenizer.encode(prompt))
        
        from ..generate import stream_generate as _stream_generate

        generated_tokens = 0
        for token in _stream_generate(
            model=self.model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            stop_sequences=params.stop_sequences,
            device=self.device_name,
            **kwargs
        ):
            yield token
            generated_tokens += 1
        
        # 更新统计
        latency_ms = (time.time() - start_time) * 1000
        self._stats.update(prompt_tokens, generated_tokens, latency_ms)

    @property
    def eos_token_id(self) -> int | None:
        """结束 token ID"""
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """填充 token ID"""
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        """词汇表大小"""
        return len(self._tokenizer)

    @property
    def config(self):
        """模型配置"""
        return self.model.config

    @property
    def tokenizer(self):
        """分词器"""
        return self._tokenizer

    def get_memory_usage(self) -> dict[str, float]:
        """获取内存使用情况
        
        Returns:
            内存使用信息
        """
        memory_info = {
            "device": self.device_name,
        }
        
        if self.device_name == "cuda" and torch.cuda.is_available():
            memory_info["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            # CPU 内存估算
            import psutil
            process = psutil.Process()
            memory_info["rss_mb"] = process.memory_info().rss / 1024 / 1024
            
        return memory_info

    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> None:
        """预热模型
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
        """
        logger.info(f"Warming up model with batch_size={batch_size}, seq_len={seq_len}")
        
        dummy_input = torch.zeros(
            batch_size, 
            seq_len, 
            dtype=torch.long, 
            device=self.device_name
        )
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        if self.device_name == "cuda":
            torch.cuda.synchronize()
            
        logger.info("Warmup complete")

    def cleanup(self) -> None:
        """清理资源"""
        super().cleanup()
        
        del self.model
        del self._tokenizer
        
        if self.device_name == "cuda":
            torch.cuda.empty_cache()
            
        logger.info("PyTorch backend cleaned up")
