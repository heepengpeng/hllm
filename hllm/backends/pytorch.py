"""PyTorch 后端实现"""

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseBackend

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PyTorchBackend(BaseBackend):
    """PyTorch 推理后端

    支持 CPU、CUDA、MPS (Apple Silicon)
    """

    NAME = "pytorch"
    SUPPORTS_QUANTIZATION = True
    DEFAULT_DEVICE = "cpu"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        self._device_name = self._normalize_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.trust_remote_code = trust_remote_code
        super().__init__(model_path, **kwargs)

    def _normalize_device(self, device: str) -> str:
        """标准化设备名称"""
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            return "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return device

    @property
    def device_name(self) -> str:
        """返回设备名称"""
        return self._device_name

    def _load_model(self, **kwargs) -> None:
        """加载模型"""
        logger.info(f"Loading PyTorch model from {self.model_path} on {self.device_name}")

        # 加载分词器
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

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
            self.model_path,
            **load_kwargs
        )
        self.model.to(self.device_name)
        self.model.eval()

        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, "compile") and kwargs.get("compile", False):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        logger.info("PyTorch model loaded successfully")

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
        """生成文本"""
        from ..generate import generate as _generate

        return _generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=self.device_name,
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
    ) -> Generator[str, None, None]:
        """流式生成文本"""
        from ..generate import stream_generate as _stream_generate

        yield from _stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=self.device_name,
            **kwargs
        )

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    @property
    def config(self):
        return self.model.config
