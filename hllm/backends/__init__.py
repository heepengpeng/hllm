"""HLLM 后端模块

支持多种推理后端：
- pytorch: PyTorch 后端 (CPU/CUDA/MPS)
- mlx: MLX 后端 (Apple Silicon)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseBackend

__all__ = ["BaseBackend", "create_backend", "list_backends", "get_backend_info"]


# 延迟导入，避免强制依赖
def _import_pytorch_backend():
    from .pytorch import PyTorchBackend
    return PyTorchBackend


def _import_mlx_backend():
    from .mlx import MLXBackend
    return MLXBackend


# 后端注册表
_BACKENDS = {
    "pytorch": _import_pytorch_backend,
    "mlx": _import_mlx_backend,
}


def list_backends() -> list[str]:
    """列出所有可用的后端"""
    available = []
    for name, importer in _BACKENDS.items():
        try:
            importer()
            available.append(name)
        except ImportError:
            pass
    return available


def get_backend_info() -> dict:
    """获取后端详细信息"""
    info = {}
    for name in _BACKENDS.keys():
        try:
            backend_class = _BACKENDS[name]()
            info[name] = {
                "available": True,
                "supports_quantization": getattr(backend_class, 'SUPPORTS_QUANTIZATION', False),
                "default_device": getattr(backend_class, 'DEFAULT_DEVICE', 'cpu'),
            }
        except ImportError as e:
            info[name] = {
                "available": False,
                "error": str(e),
            }
    return info


def create_backend(
    backend_name: str,
    model_path: str,
    **kwargs
) -> "BaseBackend":
    """创建后端实例

    Args:
        backend_name: 后端名称 (pytorch, mlx)
        model_path: 模型路径
        **kwargs: 额外参数

    Returns:
        BaseBackend 实例
    """
    if backend_name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available: {list_backends()}"
        )

    backend_class = _BACKENDS[backend_name]()
    return backend_class(model_path, **kwargs)


def auto_select_backend(model_path: str) -> tuple[str, dict]:
    """自动选择最佳后端

    优先级:
    1. MLX (如果是 Apple Silicon)
    2. PyTorch MPS (如果是 Apple Silicon)
    3. PyTorch CUDA
    4. PyTorch CPU
    """
    import platform

    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    # 检查 MLX 可用性
    if is_apple_silicon:
        try:
            import mlx
            import mlx_lm
            return "mlx", {"model_path": model_path}
        except ImportError:
            pass

    # 回退到 PyTorch
    import torch
    if torch.backends.mps.is_available():
        return "pytorch", {"model_path": model_path, "device": "mps"}
    elif torch.cuda.is_available():
        return "pytorch", {"model_path": model_path, "device": "cuda"}
    else:
        return "pytorch", {"model_path": model_path, "device": "cpu"}
