"""MLX 后端实现 (Apple Silicon)"""

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

from .base import BaseBackend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class MLXBackend(BaseBackend):
    """MLX 推理后端 (专为 Apple Silicon 优化)

    使用 mlx-lm 库进行高效推理
    """

    NAME = "mlx"
    SUPPORTS_QUANTIZATION = True
    DEFAULT_DEVICE = "mlx"

    def __init__(
        self,
        model_path: str,
        **kwargs
    ):
        self._model = None
        self._tokenizer = None
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs) -> None:
        """加载 MLX 模型"""
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "MLX backend requires 'mlx' and 'mlx-lm' packages. "
                "Install with: pip install mlx mlx-lm"
            ) from e

        logger.info(f"Loading MLX model from {self.model_path}")

        # mlx_lm.load 返回 (model, tokenizer) 元组
        self._model, self._tokenizer = load(self.model_path)

        logger.info("MLX model loaded successfully")

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
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        logger.info(f"MLX generating with max_tokens={max_new_tokens}")

        # 构建采样器
        # mlx_lm 使用 temp=0 表示 greedy，而不是 temperature=1.0
        temp = 0.0 if temperature <= 0.1 else temperature
        sampler = make_sampler(temp=temp, top_p=top_p if top_p < 1.0 else 0.0)

        response = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

        return response

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
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        logger.info(f"MLX streaming generation with max_tokens={max_new_tokens}")

        # 构建采样器
        temp = 0.0 if temperature <= 0.1 else temperature
        sampler = make_sampler(temp=temp, top_p=top_p if top_p < 1.0 else 0.0)

        # mlx_lm.stream_generate 返回 GenerationResponse 对象
        for response in mlx_stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        ):
            # GenerationResponse 有 text 属性
            if hasattr(response, 'text'):
                yield response.text
            else:
                yield str(response)

    @property
    def device_name(self) -> str:
        return "mlx"

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id if self._tokenizer else None

    @property
    def pad_token_id(self) -> int | None:
        # MLX tokenizer 可能没有 pad_token_id
        if self._tokenizer and hasattr(self._tokenizer, 'pad_token_id'):
            return self._tokenizer.pad_token_id
        return None

    @property
    def tokenizer(self):
        """返回分词器"""
        return self._tokenizer

    @property
    def config(self):
        """获取模型配置"""
        if self._model and hasattr(self._model, 'config'):
            return self._model.config
        return None
