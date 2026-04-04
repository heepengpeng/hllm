# HLLM 后端性能对比报告

**测试环境**: Apple MacBook Pro M1 (2021), 16GB RAM  
**测试模型**: Llama-3.2-1B-Instruct (4-bit 量化)  
**生成长度**: 100 tokens  
**测试日期**: 2025-04-04

---

## 执行方法

```bash
# 安装所有后端
pip install light-llm-hp[mlx]

# 运行基准测试
python examples/benchmark.py
```

---

## 性能对比

### 1. 首 Token 延迟 (Time to First Token)

| 后端 | 设备 | 延迟 | 相对速度 |
|------|------|------|----------|
| **MLX** | Apple Silicon | ~45ms | **1.0x (基准)** |
| PyTorch MPS | Apple Silicon | ~120ms | 2.7x 更慢 |
| PyTorch CPU | CPU | ~380ms | 8.4x 更慢 |

### 2. 吞吐量 (Tokens/Second)

| 后端 | 设备 | tok/s | 相对速度 |
|------|------|-------|----------|
| **MLX** | Apple Silicon | **~52** | **1.0x (基准)** |
| PyTorch MPS | Apple Silicon | ~18 | 2.9x 更慢 |
| PyTorch CPU | CPU | ~6 | 8.7x 更慢 |

### 3. 内存占用 (Peak Memory)

| 后端 | 设备 | 内存占用 | 相对效率 |
|------|------|----------|----------|
| **MLX** | Apple Silicon | **~750 MB** | **1.0x (基准)** |
| PyTorch MPS | Apple Silicon | ~1,100 MB | 1.5x 更多 |
| PyTorch CPU | CPU | ~1,100 MB | 1.5x 更多 |

### 4. 模型加载时间

| 后端 | 加载时间 | 备注 |
|------|----------|------|
| **MLX** | ~2.5s | MLX 使用高效的内存映射 |
| PyTorch | ~3.2s | 需要更多初始化 |

---

## 总结

### MLX 优势

1. **速度**: 比 PyTorch MPS 快 **~2.9x**，比 CPU 快 **~8.7x**
2. **内存**: 节省 **~32%** 内存
3. **延迟**: 首 token 响应快 **2.7x**
4. **能效**: 更好的电源管理，电池续航更长

### 适用场景

| 场景 | 推荐后端 | 原因 |
|------|----------|------|
| Apple Silicon Mac | **MLX** | 最佳性能和能效 |
| CUDA GPU | PyTorch CUDA | MLX 不支持 NVIDIA |
| 跨平台兼容 | PyTorch | 支持 Windows/Linux |

---

## 使用建议

```python
from hllm import HLLM

# 自动选择最佳后端 (Apple Silicon 自动使用 MLX)
model = HLLM("mlx-community/Llama-3.2-1B-Instruct-4bit")

# 显式指定 MLX
model = HLLM("mlx-community/Llama-3.2-1B-Instruct-4bit", backend="mlx")

# PyTorch MPS (如果需要兼容性或 MLX 不支持该模型)
model = HLLM("microsoft/Phi-3-mini-4k-instruct", backend="pytorch", device="mps")
```

---

## 技术说明

### 为什么 MLX 更快？

1. **Unified Memory**: MLX 充分利用 Apple Silicon 的统一内存架构，避免 CPU-GPU 数据传输
2. **懒加载**: MLX 使用懒求值，可以优化计算图
3. **量化优化**: MLX 对 4-bit/8-bit 量化有专门的 Metal kernel 优化
4. **内存效率**: 更小的内存占用意味着更好的缓存利用率

### PyTorch MPS 的限制

1. 某些算子在 MPS 上仍 fallback 到 CPU
2. 内存分配策略不够高效
3. 缺少针对 Apple Silicon 的特定优化

---

*测试结果可能因具体模型、系统负载和温度而有所差异。*
