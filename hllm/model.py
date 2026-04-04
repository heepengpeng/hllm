import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HLLM:
    """HLLM: 轻量级 LLM 推理框架
    
    支持多种后端：
    - pytorch: PyTorch 后端 (CPU/CUDA/MPS)
    - mlx: MLX 后端 (Apple Silicon, 推荐)
    - auto: 自动选择最佳后端
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        device: str | None = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        初始化 HLLM 模型

        Args:
            model_path: HuggingFace 模型路径或本地路径
            backend: 推理后端 ("auto", "pytorch", "mlx")
            device: 运行设备 (仅 PyTorch 后端, "cpu", "cuda", "mps")
            trust_remote_code: 是否信任远程代码
            torch_dtype: 模型数据类型 (仅 PyTorch 后端)
            **kwargs: 额外参数
        """
        self.model_path = model_path
        self.backend_name = backend

        # 自动选择后端
        if backend == "auto":
            from .backends import auto_select_backend
            backend_name, backend_kwargs = auto_select_backend(model_path)
            # 合并用户参数
            if device:
                backend_kwargs["device"] = device
            if torch_dtype:
                backend_kwargs["torch_dtype"] = torch_dtype
            backend_kwargs["trust_remote_code"] = trust_remote_code
            backend_kwargs.update(kwargs)
        else:
            backend_name = backend
            backend_kwargs = {
                "model_path": model_path,
                "trust_remote_code": trust_remote_code,
            }
            if device:
                backend_kwargs["device"] = device
            if torch_dtype:
                backend_kwargs["torch_dtype"] = torch_dtype
            backend_kwargs.update(kwargs)

        # 创建后端
        from .backends import create_backend
        self._backend = create_backend(backend_name, **backend_kwargs)
        self.backend_name = backend_name

        logger.info(f"HLLM initialized with {backend_name} backend")

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
        生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 概率
            top_k: top-k 采样
            repetition_penalty: 重复惩罚

        Returns:
            生成的文本
        """
        return self._backend.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs
        )

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ):
        """
        流式生成文本 (yield token)

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 概率
            top_k: top-k 采样
            repetition_penalty: 重复惩罚

        Yields:
            生成的 token
        """
        yield from self._backend.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs
        )

    @property
    def config(self):
        """返回模型配置"""
        return self._backend.config

    @property
    def eos_token_id(self) -> int | None:
        """返回结束 token ID"""
        return self._backend.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """返回开始 token ID"""
        if hasattr(self._backend, 'bos_token_id'):
            return self._backend.bos_token_id
        return None

    @property
    def pad_token_id(self) -> int | None:
        """返回填充 token ID"""
        return self._backend.pad_token_id

    def get_info(self) -> dict:
        """获取模型信息"""
        info = self._backend.get_info()
        info["backend"] = self.backend_name
        return info
