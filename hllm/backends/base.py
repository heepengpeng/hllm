"""后端抽象基类"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


class BaseBackend(ABC):
    """HLLM 后端抽象基类

    所有推理后端必须实现此接口
    """

    # 后端标识
    NAME: str = "base"
    SUPPORTS_QUANTIZATION: bool = False
    DEFAULT_DEVICE: str = "cpu"

    def __init__(self, model_path: str, **kwargs):
        """初始化后端

        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
            **kwargs: 后端特定参数
        """
        self.model_path = model_path
        self._load_model(**kwargs)

    @abstractmethod
    def _load_model(self, **kwargs) -> None:
        """加载模型，子类必须实现"""
        pass

    @abstractmethod
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
        """生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 概率
            top_k: top-k 采样
            repetition_penalty: 重复惩罚
            **kwargs: 额外参数

        Returns:
            生成的文本
        """
        pass

    @abstractmethod
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
        """流式生成文本

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 概率
            top_k: top-k 采样
            repetition_penalty: 重复惩罚
            **kwargs: 额外参数

        Yields:
            逐个生成的 token
        """
        pass

    @property
    @abstractmethod
    def device_name(self) -> str:
        """返回设备名称"""
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
    def tokenizer(self) -> Any:
        """返回分词器"""
        pass

    @property
    def bos_token_id(self) -> int | None:
        """返回开始 token ID (可选)"""
        return None

    @property
    def config(self) -> Any:
        """返回模型配置 (可选)"""
        return None

    def get_info(self) -> dict[str, Any]:
        """获取后端信息"""
        return {
            "name": self.NAME,
            "device": self.device_name,
            "model_path": self.model_path,
            "supports_quantization": self.SUPPORTS_QUANTIZATION,
        }
