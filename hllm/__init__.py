"""
HLLM - 轻量级 LLM 推理框架

提供统一的 API，支持多种后端：
- PyTorch (CUDA/MPS/CPU)
- MLX (Apple Silicon)
- PagedAttention (CUDA 优化)

Example:
    >>> from hllm import HLLM
    >>> model = HLLM("microsoft/Phi-3-mini-4k-instruct")
    >>> result = model.generate("Hello, how are you?")
"""

__version__ = "0.2.0"

from .model import HLLM
from .config import (
    HLLMConfig,
    ModelConfig,
    ServerConfig,
    GenerationConfig,
    get_config,
    reload_config,
)
from .tokenizer import Tokenizer
from .generate import generate

__all__ = [
    "HLLM",
    "HLLMConfig",
    "ModelConfig",
    "ServerConfig",
    "GenerationConfig",
    "get_config",
    "reload_config",
    "Tokenizer",
    "generate",
    "__version__",
]

# 条件导出 Server (需要 fastapi)
try:
    from .server import Server
    __all__.append("Server")
except ImportError:
    pass

# 条件导出 HLLMClient (需要 requests)
try:
    from .client import HLLMClient
    __all__.append("HLLMClient")
except ImportError:
    pass