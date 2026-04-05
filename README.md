# light-llm-hp - 轻量级 LLM 推理框架

[![PyPI version](https://badge.fury.io/py/light-llm-hp.svg)](https://badge.fury.io/py/light-llm-hp)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://pypi.org/project/light-llm-hp/)
[![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](docs/coverage_report.md)
[![CI](https://github.com/heepengpeng/hllm/actions/workflows/ci.yml/badge.svg)](https://github.com/heepengpeng/hllm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

在 CPU/GPU 上运行的简化推理框架，支持 REST API 服务。

**🚀 多平台优化**:
- **NVIDIA GPU**: PyTorch CUDA 后端，RTX 3080 Ti 达 55+ tok/s
- **Apple Silicon**: MLX 后端，比 PyTorch MPS 快 8+ 倍
- **统一 API**: 同一套代码，自动选择最佳后端

## 快速开始

```python
from hllm import HLLM

# 自动选择最佳后端 (Apple Silicon 上自动使用 MLX)
model = HLLM(model_path="microsoft/Phi-3-mini-4k-instruct")

# 生成文本
result = model.generate("Write a short story about a robot.")
print(result)
```

### NVIDIA GPU 优化 (CUDA)

在 NVIDIA GPU 上自动使用 CUDA 加速，支持模型自动下载：

```bash
# 安装 PyTorch CUDA 版本
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install light-llm-hp
```

```python
from hllm import HLLM

# 自动从 ModelScope/HuggingFace 下载模型
model = HLLM(model_path="Llama-3.2-1B-Instruct", backend="pytorch", device="cuda")

# 或使用完整模型 ID
model = HLLM(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda")

# 查看后端信息
print(model.get_info())
# {'name': 'pytorch', 'device': 'cuda', ...}
```

### Apple Silicon 优化 (MLX)

在 M1/M2/M3 Mac 上，使用 MLX 后端可获得最佳性能：

```bash
# 安装 MLX 支持
pip install light-llm-hp[mlx]
```

```python
from hllm import HLLM

# 显式使用 MLX 后端 (推荐)
model = HLLM(model_path="mlx-community/Llama-3.2-1B-Instruct-4bit", backend="mlx")

# 或使用 PyTorch MPS
model = HLLM(model_path="microsoft/Phi-3-mini-4k-instruct", backend="pytorch", device="mps")

# 查看后端信息
print(model.get_info())
# {'name': 'mlx', 'device': 'mlx', ...}
```

### 性能对比

**GPU 服务器** (RTX 3080 Ti, Llama-3.2-1B, 100 tokens):

| 后端 | 首 token 延迟 | 吞吐量 | 显存占用 |
|------|--------------|--------|----------|
| PyTorch CUDA | **19ms** 🏆 | **55.1 tok/s** 🏆 | ~2.3GB |

**Apple Silicon** (M1/M2/M3, Llama-3.2-1B, 100 tokens):

| 后端 | 首 token 延迟 | 吞吐量 | 内存占用 |
|------|--------------|--------|----------|
| MLX | ~50ms | 32.5 tok/s | ~780MB |
| PyTorch MPS | ~150ms | 3.8 tok/s | ~1GB |
| PyTorch CPU | ~500ms | 1.6 tok/s | ~464MB |

> 📊 详细报告: [docs/benchmark_report.md](docs/benchmark_report.md)

运行基准测试：
```bash
# Apple Silicon
python examples/benchmark.py

# GPU 服务器
python benchmark_remote.py
```

## REST API 服务 (OpenAI 兼容)

### 安装 API 依赖

```bash
pip install light-llm-hp[api]
```

### 启动服务

```bash
# 使用本地模型
python -m hllm.server --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000

# 自动下载模型 (从 ModelScope/HuggingFace)
python -m hllm.server --model Llama-3.2-1B-Instruct --port 8000

# GPU 服务器 (CUDA)
python -m hllm.server --model Llama-3.2-1B-Instruct --device cuda

# Apple Silicon (MLX)
python -m hllm.server --model mlx-community/Llama-3.2-1B-Instruct-4bit --backend mlx
```

### 使用 OpenAI 官方客户端

```python
import httpx
from openai import OpenAI

# 禁用代理避免 502 错误
http_client = httpx.Client(trust_env=False)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    http_client=http_client
)

# 对话
response = client.chat.completions.create(
    model="hllm-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

完整示例：[examples/test_openai_client.py](examples/test_openai_client.py)

### OpenAI 兼容端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | 对话补全 (支持流式) |
| `/v1/completions` | POST | 文本补全 (支持流式) |

详细 API 文档见 [docs/api.md](docs/api.md)。

## 目录结构

```
hllm/
├── hllm/                   # 核心模块
│   ├── __init__.py
│   ├── model.py            # 模型加载与推理
│   ├── tokenizer.py        # 分词器封装
│   ├── generate.py         # 生成逻辑
│   ├── server.py           # REST API 服务端
│   ├── client.py           # REST API 客户端
│   ├── backends/           # 后端实现
│   │   ├── base.py         # 后端基类
│   │   ├── pytorch.py      # PyTorch CUDA/MPS/CPU
│   │   ├── mlx.py          # Apple Silicon MLX
│   │   └── paged_pytorch.py # PagedAttention 优化
│   ├── paged_attention/    # PagedAttention 模块
│   │   ├── block_manager.py   # 内存块管理
│   │   ├── paged_attention.py # 注意力计算
│   │   └── scheduler.py       # 请求调度
│   └── utils/
│       └── model_downloader.py # 模型下载工具
├── tests/                  # 测试
├── examples/               # 示例
└── docs/                   # 文档
```

---

## ✅ 当前支持的特性

### 多后端支持

| 后端 | 设备 | 状态 | 性能 |
|------|------|------|------|
| **MLX** | Apple Silicon | ✅ 稳定 | 32.5 tok/s |
| **PyTorch** | CUDA (NVIDIA) | ✅ 稳定 | 55.1 tok/s |
| **PyTorch** | MPS (Apple) | ✅ 稳定 | 3.8 tok/s |
| **PyTorch** | CPU | ✅ 稳定 | 1.6 tok/s |
| **PagedAttention** | CUDA | ✅ 已发布 | **64.6 tok/s** 🏆 |

### 核心功能

- ✅ **自动后端选择**: 根据硬件自动选择最优后端
- ✅ **模型自动下载**: 支持 ModelScope 和 HuggingFace 镜像
- ✅ **REST API 服务**: OpenAI 兼容的 API 接口
- ✅ **流式生成**: 支持 SSE 流式输出
- ✅ **量化支持**: MLX 4-bit 量化
- ✅ **PagedAttention**: vLLM 风格的内存优化
- ✅ **连续批处理**: 动态批处理提升吞吐量
- ✅ **Copy-on-Write**: 内存共享机制

### 模型支持

- ✅ Llama 系列 (1B, 3B, 7B+)
- ✅ Phi-3 系列
- ✅ TinyLlama
- ✅ 其他 HuggingFace Transformers 模型

### 开发工具

- ✅ **Benchmark 工具**: 多平台性能测试
- ✅ **覆盖率测试**: 72%+ 代码覆盖率
- ✅ **CI/CD**: GitHub Actions 自动化
- ✅ **类型检查**: Pyright 静态分析

---

## 🚀 Roadmap (后续规划)

### Phase 1: 性能优化 (近期)

- 🔄 **Prefix Caching**: 共享前缀的 KV cache 复用
- 🔄 **Speculative Decoding**: 投机解码加速
- 🔄 **Flash Attention 2**: 集成 FA2 进一步加速
- 🔄 **CUDA Graph**: 捕获计算图减少 CPU 开销
- 🔄 **INT8/FP8 量化**: 更低显存占用

### Phase 2: 功能扩展 (中期)

- 📋 **Function Calling**: OpenAI 风格的函数调用
- 📋 **Vision Models**: 多模态模型支持 (Llava 等)
- 📋 **Embedding API**: 文本嵌入接口
- 📋 **LoRA Adapter**: 动态加载 LoRA 适配器
- 📋 **Tensor Parallelism**: 多 GPU 张量并行

### Phase 3: 生产就绪 (远期)

- 📋 **Pipeline Parallelism**: 流水线并行
- 📋 **Dynamic Batching**: 更智能的动态批处理
- 📋 **Request Scheduling**: 优先级调度、QoS 保证
- 📋 **Metrics & Monitoring**: Prometheus 指标导出
- 📋 **Model Quantization**: GPTQ/AWQ 支持
- 📋 **vLLM 兼容**: 与 vLLM API 完全兼容

### Phase 4: 生态系统

- 📋 **Docker 镜像**: 官方 Docker 部署方案
- 📋 **K8s Operator**: Kubernetes 部署支持
- 📋 **Model Hub**: 预优化模型仓库
- 📋 **Chat UI**: 内置 Web 聊天界面
- 📋 **Mobile SDK**: iOS/Android 推理 SDK

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

```bash
# 克隆仓库
git clone https://github.com/heepengpeng/hllm.git
cd hllm

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 运行基准测试
python examples/benchmark.py
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

*用 ❤️ 和 🤖 构建 - Vibe Coding 驱动开发*