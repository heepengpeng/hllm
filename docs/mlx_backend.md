# MLX 后端支持设计文档

## 概述

为 HLLM 添加 Apple Silicon (M1/M2/M3) 原生 MLX 后端支持，提供比 PyTorch MPS 更快的推理性能。

## MLX 优势

- **专为 Apple Silicon 设计**：充分利用 Unified Memory 架构
- **内存效率**：比 PyTorch 节省 30-50% 内存
- **性能提升**：通常比 MPS 快 2-5 倍
- **量化支持**：原生支持 4-bit/8-bit 量化

## 架构设计

```
hllm/
├── backends/
│   ├── __init__.py       # 后端注册和工厂
│   ├── base.py           # 抽象基类
│   ├── pytorch.py        # PyTorch 后端 (现有)
│   └── mlx.py            # MLX 后端 (新增)
```

## 抽象基类接口

```python
class BaseBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...
    
    @abstractmethod  
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]: ...
    
    @property
    @abstractmethod
    def device_name(self) -> str: ...
```

## 使用方式

```python
from hllm import HLLM

# 自动选择最佳后端
model = HLLM("microsoft/Phi-3-mini-4k-instruct")

# 显式指定后端
model = HLLM("microsoft/Phi-3-mini-4k-instruct", backend="mlx")
model = HLLM("microsoft/Phi-3-mini-4k-instruct", backend="pytorch", device="mps")
```

## 依赖

```toml
[project.optional-dependencies]
mlx = ["mlx>=0.15.0", "mlx-lm>=0.15.0"]
```

## 性能对比指标

1. **首 token 延迟** (time to first token)
2. **吞吐量** (tokens/second)
3. **内存占用** (峰值内存)
4. **模型加载时间**

## 支持模型

MLX 支持的模型列表：https://github.com/ml-explore/mlx-examples/tree/main/llms

常见支持模型：
- Llama 2/3
- Mistral
- Phi-3
- Qwen2
- Gemma
