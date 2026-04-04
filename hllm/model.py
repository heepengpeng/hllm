import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class HLLM:
    """HLLM: 轻量级 LLM 推理框架"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        初始化 HLLM 模型

        Args:
            model_path: HuggingFace 模型路径或本地路径
            device: 运行设备 ("cpu", "cuda", "mps")
            trust_remote_code: 是否信任远程代码
            torch_dtype: 模型数据类型
        """
        self.model_path = model_path
        self.device = self._normalize_device(device)

        # 默认使用 float32 在 CPU 上
        if torch_dtype is None:
            torch_dtype = torch.float32

        logger.info(f"Loading model from {model_path} on {self.device}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")

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
    def config(self) -> AutoConfig:
        """返回模型配置"""
        return self.model.config

    @property
    def eos_token_id(self) -> int:
        """返回结束 token ID"""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """返回开始 token ID"""
        return self.tokenizer.bos_token_id

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
        from .generate import generate

        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=self.device,
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
        from .generate import stream_generate

        yield from stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=self.device,
            **kwargs
        )