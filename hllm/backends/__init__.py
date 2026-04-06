"""HLLM 后端模块

提供统一的 LLM 推理后端接口，支持多种推理框架：
- PyTorch: CPU/CUDA/MPS 通用推理
- MLX: Apple Silicon 专用优化
- PagedAttention: CUDA 高性能推理

Example:
    >>> from hllm.backends import create_backend, list_backends
    >>> 
    >>> # 列出可用后端
    >>> print(list_backends())  # ['pytorch', 'mlx', 'paged_pytorch']
    >>> 
    >>> # 创建后端
    >>> backend = create_backend("pytorch", model_path="model", device="cuda")
    >>> 
    >>> # 生成文本
    >>> result = backend.generate("Hello", max_new_tokens=50)
"""

from typing import Type, Any

from .base import (
    BaseBackend,
    GenerationParams,
    BackendStats,
    TokenizerProtocol,
)

# 后端注册表
_BACKENDS: dict[str, Type[BaseBackend]] = {}


def register_backend(name: str, backend_class: Type[BaseBackend]) -> None:
    """注册后端类
    
    Args:
        name: 后端名称
        backend_class: 后端类，必须继承 BaseBackend
        
    Raises:
        ValueError: 后端名称已存在
        TypeError: 后端类不合法
        
    Example:
        >>> class MyBackend(BaseBackend):
        ...     NAME = "my_backend"
        ...     # ... 实现抽象方法
        >>> register_backend("my_backend", MyBackend)
    """
    if name in _BACKENDS:
        raise ValueError(f"Backend '{name}' already registered")
    
    if not issubclass(backend_class, BaseBackend):
        raise TypeError(f"Backend class must inherit from BaseBackend")
    
    _BACKENDS[name] = backend_class
    

def list_backends() -> list[str]:
    """列出所有可用的后端名称
    
    Returns:
        后端名称列表
        
    Example:
        >>> print(list_backends())
        ['pytorch', 'mlx', 'paged_pytorch']
    """
    return list(_BACKENDS.keys())


def get_backend_class(name: str) -> Type[BaseBackend]:
    """获取后端类
    
    Args:
        name: 后端名称
        
    Returns:
        后端类
        
    Raises:
        ValueError: 后端不存在
    """
    if name not in _BACKENDS:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")
    
    return _BACKENDS[name]


def create_backend(
    backend_name: str,
    model_path: str,
    **kwargs
) -> BaseBackend:
    """创建后端实例
    
    工厂函数，根据名称创建对应的后端实例。
    
    Args:
        backend_name: 后端名称 (pytorch/mlx/paged_pytorch)
        model_path: 模型路径或 HuggingFace model ID
        **kwargs: 后端特定参数
            - pytorch: device, torch_dtype, trust_remote_code
            - mlx: (无特殊参数)
            - paged_pytorch: device, block_size, num_blocks
            
    Returns:
        后端实例
        
    Raises:
        ValueError: 后端名称无效
        RuntimeError: 模型加载失败
        
    Example:
        >>> # PyTorch CUDA
        >>> backend = create_backend("pytorch", "model", device="cuda")
        >>> 
        >>> # MLX (Apple Silicon)
        >>> backend = create_backend("mlx", "model")
        >>> 
        >>> # PagedAttention
        >>> backend = create_backend("paged_pytorch", "model", device="cuda")
    """
    backend_class = get_backend_class(backend_name)
    return backend_class(model_path=model_path, **kwargs)


def auto_select_backend(device: str | None = None) -> str:
    """自动选择最佳后端
    
    根据当前硬件环境自动选择最优后端。
    
    Args:
        device: 指定设备 (可选)，如 cuda/mps/cpu
        
    Returns:
        推荐的后端名称
        
    Example:
        >>> backend_name = auto_select_backend()
        >>> print(backend_name)  # 'mlx' 或 'pytorch'
    """
    # 优先检查 MLX (Apple Silicon)
    try:
        import mlx.core as mx
        if device is None or device == "mlx":
            return "mlx"
    except ImportError:
        pass
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            # 检查是否有 PagedAttention
            if "paged_pytorch" in list_backends():
                if device is None or device == "cuda":
                    return "paged_pytorch"
            if device is None or device in ("cuda", "auto"):
                return "pytorch"
    except ImportError:
        pass
    
    # 检查 MPS (Apple Silicon fallback)
    try:
        import torch
        if torch.backends.mps.is_available():
            if device is None or device in ("mps", "auto"):
                return "pytorch"
    except ImportError:
        pass
    
    # 默认使用 PyTorch CPU
    return "pytorch"


# 延迟导入和注册后端
def _register_builtin_backends():
    """注册内置后端"""
    
    # PyTorch
    try:
        from .pytorch import PyTorchBackend
        register_backend("pytorch", PyTorchBackend)
    except ImportError as e:
        pass  # PyTorch 未安装
    
    # MLX (Apple Silicon)
    try:
        from .mlx import MLXBackend
        register_backend("mlx", MLXBackend)
    except ImportError as e:
        pass  # MLX 未安装
    
    # PagedAttention
    try:
        from .paged_pytorch import PagedPyTorchBackend
        register_backend("paged_pytorch", PagedPyTorchBackend)
    except ImportError as e:
        pass  # 依赖未安装


# 初始化注册
_register_builtin_backends()


__all__ = [
    # 基础类
    "BaseBackend",
    "GenerationParams",
    "BackendStats",
    "TokenizerProtocol",
    
    # 函数
    "create_backend",
    "list_backends",
    "get_backend_class",
    "auto_select_backend",
    "register_backend",
    
    # 后端类 (如果已安装)
    "PyTorchBackend",
    "MLXBackend",
    "PagedPyTorchBackend",
]

# 条件导出
try:
    from .pytorch import PyTorchBackend
except ImportError:
    pass

try:
    from .mlx import MLXBackend
except ImportError:
    pass

try:
    from .paged_pytorch import PagedPyTorchBackend
except ImportError:
    pass
