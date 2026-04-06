"""后端抽象基类

提供统一的 LLM 推理后端接口，所有后端实现必须继承此类。
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class GenerationParams:
    """标准化生成参数
    
    所有后端使用统一的生成参数结构，便于切换后端时保持一致性。
    """
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """验证参数有效性"""
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


@dataclass  
class BackendStats:
    """后端性能统计
    
    记录推理性能指标，用于监控和优化。
    """
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    avg_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    
    def update(self, prompt_tokens: int, generated_tokens: int, latency_ms: float) -> None:
        """更新统计信息"""
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_tokens_generated += generated_tokens
        
        # 使用滑动平均更新
        alpha = 0.1  # 平滑因子
        self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
        
        if latency_ms > 0:
            tps = generated_tokens / (latency_ms / 1000)
            self.avg_tokens_per_second = (1 - alpha) * self.avg_tokens_per_second + alpha * tps


@runtime_checkable
class TokenizerProtocol(Protocol):
    """分词器协议
    
    定义分词器必须实现的最小接口。
    """
    
    def encode(self, text: str, **kwargs) -> list[int]:
        """编码文本为 token IDs"""
        ...
    
    def decode(self, token_ids: list[int], **kwargs) -> str:
        """解码 token IDs 为文本"""
        ...
    
    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        """应用对话模板"""
        ...
    
    @property
    def eos_token_id(self) -> int | None:
        """结束 token ID"""
        ...
    
    @property
    def pad_token_id(self) -> int | None:
        """填充 token ID"""
        ...
    
    @property
    def vocab_size(self) -> int:
        """词汇表大小"""
        ...


class BaseBackend(ABC):
    """HLLM 后端抽象基类
    
    所有推理后端必须实现此接口。提供统一的模型加载、文本生成、
    流式生成和性能监控能力。
    
    Example:
        >>> class MyBackend(BaseBackend):
        ...     NAME = "my_backend"
        ...     
        ...     def _load_model(self, **kwargs) -> None:
        ...         # 加载模型逻辑
        ...         pass
        ...     
        ...     def generate(self, prompt: str, params: GenerationParams) -> str:
        ...         # 生成逻辑
        ...         pass
    """
    
    # 类属性：后端标识和能力
    NAME: str = "base"
    """后端名称标识"""
    
    SUPPORTS_QUANTIZATION: bool = False
    """是否支持量化"""
    
    SUPPORTS_GPU: bool = False
    """是否支持 GPU 加速"""
    
    DEFAULT_DEVICE: str = "cpu"
    """默认设备类型"""
    
    def __init__(self, model_path: str, **kwargs):
        """初始化后端
        
        Args:
            model_path: 模型路径或 HuggingFace model ID
            **kwargs: 后端特定参数
            
        Raises:
            RuntimeError: 模型加载失败
            ImportError: 缺少必要依赖
        """
        self.model_path = model_path
        self._stats = BackendStats()
        self._is_loaded = False
        
        try:
            self._load_model(**kwargs)
            self._is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e
    
    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._is_loaded
    
    @property
    def stats(self) -> BackendStats:
        """获取性能统计"""
        return self._stats
    
    @abstractmethod
    def _load_model(self, **kwargs) -> None:
        """加载模型，子类必须实现
        
        此方法在 __init__ 中被调用，子类应在此完成：
        - 模型权重加载
        - 分词器初始化
        - 设备配置
        
        Args:
            **kwargs: 后端特定参数
        """
        pass
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs
    ) -> str:
        """生成文本
        
        统一的生成接口，所有后端应遵循此签名。
        子类可以重写此方法来提供优化实现。
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数 (0.0-2.0)
            top_p: nucleus sampling 概率 (0.0-1.0)
            top_k: top-k 采样 (0 表示禁用)
            repetition_penalty: 重复惩罚 (>= 1.0)
            stop_sequences: 停止序列列表
            **kwargs: 后端特定参数
            
        Returns:
            生成的文本
            
        Raises:
            RuntimeError: 模型未加载
            ValueError: 参数无效
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # 标准化参数
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences or []
        )
        params.validate()
        
        return self._generate_impl(prompt, params, **kwargs)
    
    @abstractmethod
    def _generate_impl(self, prompt: str, params: GenerationParams, **kwargs) -> str:
        """生成实现，子类必须重写
        
        Args:
            prompt: 输入提示
            params: 标准化的生成参数
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        pass
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成文本
        
        逐个生成 token 并 yield，用于实时显示。
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 概率
            top_k: top-k 采样
            repetition_penalty: 重复惩罚
            stop_sequences: 停止序列
            **kwargs: 后端特定参数
            
        Yields:
            逐个生成的 token 字符串
            
        Raises:
            RuntimeError: 模型未加载
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences or []
        )
        params.validate()
        
        yield from self._stream_generate_impl(prompt, params, **kwargs)
    
    @abstractmethod
    def _stream_generate_impl(
        self, 
        prompt: str, 
        params: GenerationParams, 
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成实现，子类必须重写
        
        Args:
            prompt: 输入提示
            params: 标准化的生成参数
            **kwargs: 额外参数
            
        Yields:
            逐个生成的 token
        """
        pass
    
    @property
    @abstractmethod
    def device_name(self) -> str:
        """返回设备名称
        
        Returns:
            设备标识，如 "cpu", "cuda", "mps", "mlx"
        """
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """返回结束 token ID"""
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """返回填充 token ID"""
        pass
    
    @property
    @abstractmethod
    def tokenizer(self) -> TokenizerProtocol:
        """返回分词器
        
        Returns:
            符合 TokenizerProtocol 的分词器实例
        """
        pass
    
    @property
    def bos_token_id(self) -> int | None:
        """返回开始 token ID (可选)"""
        return None
    
    @property
    def vocab_size(self) -> int | None:
        """返回词汇表大小 (可选)"""
        return None
    
    @property
    def config(self) -> Any:
        """返回模型配置 (可选)"""
        return None
    
    def get_memory_usage(self) -> dict[str, float]:
        """获取内存使用情况 (可选)
        
        Returns:
            内存使用信息，包含 used_mb, peak_mb 等
        """
        return {}
    
    def warmup(self, batch_size: int = 1, seq_len: int = 128) -> None:
        """预热模型 (可选)
        
        通过运行一次前向传播来初始化 CUDA 缓存等。
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
        """
        pass
    
    def get_info(self) -> dict[str, Any]:
        """获取后端信息
        
        Returns:
            包含后端元信息的字典
        """
        return {
            "name": self.NAME,
            "device": self.device_name,
            "model_path": self.model_path,
            "supports_quantization": self.SUPPORTS_QUANTIZATION,
            "supports_gpu": self.SUPPORTS_GPU,
            "is_loaded": self._is_loaded,
            "stats": {
                "total_requests": self._stats.total_requests,
                "avg_tokens_per_second": round(self._stats.avg_tokens_per_second, 2),
            }
        }
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self._stats = BackendStats()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动清理资源"""
        self.cleanup()
    
    def cleanup(self) -> None:
        """清理资源 (可选)
        
        释放模型占用的内存和显存。
        """
        self._is_loaded = False
