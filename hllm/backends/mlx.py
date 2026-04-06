"""MLX 后端实现 (Apple Silicon)

专为 Apple Silicon (M1/M2/M3) 优化的推理后端，使用 MLX 框架。
"""

import logging
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

from .base import BaseBackend, GenerationParams

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class MLXBackend(BaseBackend):
    """MLX 推理后端 (专为 Apple Silicon 优化)

    使用 mlx-lm 库进行高效推理，支持 Apple Silicon 的统一内存架构。
    
    Example:
        >>> backend = MLXBackend("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>> result = backend.generate("Hello, how are you?", max_new_tokens=50)
    """

    NAME = "mlx"
    SUPPORTS_QUANTIZATION = True
    SUPPORTS_GPU = True  # Apple Silicon GPU
    DEFAULT_DEVICE = "mlx"

    def __init__(self, model_path: str, **kwargs):
        """初始化 MLX 后端
        
        Args:
            model_path: 模型路径或 HuggingFace model ID
            **kwargs: 额外参数
        """
        self._model = None
        self._tokenizer = None
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs) -> None:
        """加载 MLX 模型
        
        Args:
            **kwargs: 加载参数
            
        Raises:
            ImportError: 缺少 mlx 或 mlx-lm
            RuntimeError: 模型加载失败
        """
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "MLX backend requires 'mlx' and 'mlx-lm' packages. "
                "Install with: pip install mlx mlx-lm"
            ) from e

        logger.info(f"Loading MLX model from {self.model_path}")

        # mlx_lm.load 返回 (model, tokenizer) 或 (model, tokenizer, config) 元组
        result = load(self.model_path)
        self._model = result[0]
        self._tokenizer = result[1]

        logger.info("MLX model loaded successfully")

    @property
    def device_name(self) -> str:
        """设备名称"""
        return "mlx"

    def _generate_impl(self, prompt: str, params: GenerationParams, **kwargs) -> str:
        """生成实现
        
        Args:
            prompt: 输入提示
            params: 生成参数
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        # 计算统计
        start_time = time.time()
        prompt_tokens = len(self._tokenizer.encode(prompt))

        # 构建采样器
        # mlx_lm 使用 temp=0 表示 greedy，而不是 temperature=1.0
        temp = 0.0 if params.temperature <= 0.1 else params.temperature
        sampler = make_sampler(
            temp=temp, 
            top_p=params.top_p if params.top_p < 1.0 else 0.0
        )

        response = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        
        # 更新统计
        generated_tokens = len(self._tokenizer.encode(response))
        latency_ms = (time.time() - start_time) * 1000
        self._stats.update(prompt_tokens, generated_tokens, latency_ms)

        return response

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
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        prompt_tokens = len(self._tokenizer.encode(prompt))

        # 构建采样器
        temp = 0.0 if params.temperature <= 0.1 else params.temperature
        sampler = make_sampler(
            temp=temp, 
            top_p=params.top_p if params.top_p < 1.0 else 0.0
        )

        # 流式生成
        generated_tokens = 0
        for response in mlx_stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            sampler=sampler,
        ):
            # GenerationResponse 有 text 属性
            if hasattr(response, 'text'):
                yield response.text
            else:
                yield str(response)
            generated_tokens += 1
        
        # 更新统计
        latency_ms = (time.time() - start_time) * 1000
        self._stats.update(prompt_tokens, generated_tokens, latency_ms)

    @property
    def eos_token_id(self) -> int | None:
        """结束 token ID"""
        if self._tokenizer:
            eos_id = getattr(self._tokenizer, 'eos_token_id', None)
            if eos_id is not None and isinstance(eos_id, int):
                return eos_id
        return None

    @property
    def pad_token_id(self) -> int | None:
        """填充 token ID"""
        if self._tokenizer:
            pad_id = getattr(self._tokenizer, 'pad_token_id', None)
            if pad_id is not None and isinstance(pad_id, int):
                return pad_id
            # 如果没有 pad_token，使用 eos_token
            return self.eos_token_id
        return None

    @property
    def vocab_size(self) -> int | None:
        """词汇表大小"""
        if self._tokenizer:
            return len(self._tokenizer)
        return None

    @property
    def tokenizer(self):
        """分词器"""
        return self._tokenizer

    @property
    def config(self):
        """模型配置"""
        if self._model and hasattr(self._model, 'config'):
            return self._model.config
        return None

    def get_memory_usage(self) -> dict[str, float]:
        """获取内存使用情况
        
        Returns:
            内存使用信息
        """
        # MLX 使用统一内存，无法精确区分 CPU/GPU
        import psutil
        process = psutil.Process()
        
        return {
            "device": "mlx",
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "note": "MLX uses unified memory architecture"
        }

    def cleanup(self) -> None:
        """清理资源"""
        super().cleanup()
        
        del self._model
        del self._tokenizer
        
        logger.info("MLX backend cleaned up")
