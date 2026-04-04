"""HLLM 后端性能对比基准测试

测试 PyTorch CPU/MPS vs MLX 在 Apple Silicon 上的性能
"""

import time
import psutil
import os
from typing import Callable
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    backend: str
    device: str
    load_time: float  # 模型加载时间 (秒)
    first_token_time: float  # 首 token 延迟 (秒)
    tokens_per_sec: float  # 吞吐量 (tokens/second)
    peak_memory_mb: float  # 峰值内存 (MB)
    total_tokens: int  # 生成的总 token 数


def measure_memory() -> float:
    """测量当前内存使用 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_backend(
    backend_name: str,
    model_path: str,
    prompt: str,
    max_new_tokens: int = 100,
    warmup: int = 1,
    runs: int = 3,
    **kwargs
) -> BenchmarkResult:
    """测试单个后端性能"""
    from hllm import HLLM

    print(f"\n{'='*60}")
    print(f"测试后端: {backend_name}")
    print(f"{'='*60}")

    # 测量加载时间
    print("加载模型...")
    mem_before = measure_memory()
    load_start = time.time()

    try:
        if backend_name == "mlx":
            model = HLLM(model_path, backend="mlx")
        else:
            device = kwargs.get("device", "cpu")
            dtype = kwargs.get("dtype")
            model = HLLM(model_path, backend="pytorch", device=device, torch_dtype=dtype)
    except Exception as e:
        print(f"加载失败: {e}")
        raise

    load_time = time.time() - load_start
    mem_after = measure_memory()
    peak_memory = mem_after - mem_before

    print(f"✓ 加载时间: {load_time:.2f}s")
    print(f"✓ 内存占用: {peak_memory:.1f} MB")

    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for _ in range(warmup):
        _ = model.generate(prompt, max_new_tokens=10)

    # 正式测试
    print(f"正式测试 ({runs} runs)...")
    first_token_times = []
    total_times = []

    for i in range(runs):
        # 测量首 token 时间
        start = time.time()
        # 流式生成来测量首 token
        tokens = []
        for j, token in enumerate(model.stream_generate(prompt, max_new_tokens=max_new_tokens)):
            if j == 0:
                first_token_time = time.time() - start
                first_token_times.append(first_token_time)
            tokens.append(token)

        total_time = time.time() - start
        total_times.append(total_time)

        print(f"  Run {i+1}: {len(tokens)} tokens in {total_time:.2f}s "
              f"({len(tokens)/total_time:.1f} tok/s), "
              f"first token: {first_token_time*1000:.1f}ms")

    avg_first_token = mean(first_token_times)
    avg_total_time = mean(total_times)
    tokens_per_sec = max_new_tokens / avg_total_time

    return BenchmarkResult(
        backend=backend_name,
        device=kwargs.get("device", "mlx"),
        load_time=load_time,
        first_token_time=avg_first_token,
        tokens_per_sec=tokens_per_sec,
        peak_memory_mb=peak_memory,
        total_tokens=max_new_tokens,
    )


def print_comparison(results: list[BenchmarkResult]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("性能对比报告")
    print("="*80)

    # 表头
    print(f"{'后端':<15} {'设备':<10} {'加载(s)':<10} {'首token(ms)':<12} {'速度(tok/s)':<12} {'内存(MB)':<10}")
    print("-"*80)

    # 数据
    for r in results:
        print(f"{r.backend:<15} {r.device:<10} {r.load_time:<10.2f} "
              f"{r.first_token_time*1000:<12.1f} {r.tokens_per_sec:<12.1f} {r.peak_memory_mb:<10.1f}")

    print("="*80)

    # 如果有 MLX 结果，计算加速比
    mlx_result = next((r for r in results if r.backend == "mlx"), None)
    if mlx_result:
        print("\n📊 MLX 相对于其他后端的加速比:")
        for r in results:
            if r.backend != "mlx":
                speedup = r.tokens_per_sec / mlx_result.tokens_per_sec
                print(f"  vs {r.backend} ({r.device}): {1/speedup:.1f}x {'更快' if speedup < 1 else '更慢'}")


def main():
    """主函数"""
    import platform

    # 检查是否在 Apple Silicon 上
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    if not is_apple_silicon:
        print("⚠️  本测试专为 Apple Silicon (M1/M2/M3) 设计")
        print("   当前平台可能无法运行 MLX 后端")

    # 测试配置
    MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    PROMPT = "Explain quantum computing in simple terms. Start with:"
    MAX_TOKENS = 50

    print(f"\n模型: {MODEL}")
    print(f"提示: {PROMPT[:50]}...")
    print(f"生成长度: {MAX_TOKENS} tokens")

    results = []

    # 1. 测试 MLX (如果可用)
    try:
        import mlx
        result = benchmark_backend(
            "mlx", MODEL, PROMPT,
            max_new_tokens=MAX_TOKENS,
            warmup=1, runs=3
        )
        results.append(result)
    except ImportError:
        print("\n⚠️  MLX 未安装，跳过")
        print("   安装: pip install mlx mlx-lm")
    except Exception as e:
        print(f"\n❌ MLX 测试失败: {e}")

    # 2. 测试 PyTorch MPS (如果可用)
    import torch
    if torch.backends.mps.is_available():
        try:
            result = benchmark_backend(
                "pytorch", MODEL, PROMPT,
                max_new_tokens=MAX_TOKENS,
                device="mps",
                dtype=torch.float16,
                warmup=1, runs=3
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ PyTorch MPS 测试失败: {e}")
    else:
        print("\n⚠️  MPS 不可用，跳过 PyTorch MPS 测试")

    # 3. 测试 PyTorch CPU
    try:
        result = benchmark_backend(
            "pytorch", MODEL, PROMPT,
            max_new_tokens=MAX_TOKENS,
            device="cpu",
            warmup=1, runs=2  # CPU 较慢，减少 runs
        )
        results.append(result)
    except Exception as e:
        print(f"\n❌ PyTorch CPU 测试失败: {e}")

    # 打印对比结果
    if results:
        print_comparison(results)
    else:
        print("\n❌ 没有成功的测试结果")


if __name__ == "__main__":
    main()
