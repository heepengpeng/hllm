"""MLX 后端使用示例

在 Apple Silicon Mac 上获得最佳性能
"""

from hllm import HLLM

# 使用 MLX 推荐的 4-bit 量化模型
MODEL_PATH = "mlx-community/Llama-3.2-1B-Instruct-4bit"

def main():
    print("=" * 60)
    print("HLLM MLX 后端示例")
    print("=" * 60)

    # 方法 1: 显式指定 MLX 后端
    print("\n1. 加载 MLX 模型...")
    model = HLLM(MODEL_PATH, backend="mlx")
    print(f"   后端信息: {model.get_info()}")

    # 生成文本
    prompt = "Explain machine learning in one sentence:"
    print(f"\n2. 生成文本")
    print(f"   提示: {prompt}")

    print("\n   生成结果:")
    result = model.generate(prompt, max_new_tokens=50, temperature=0.7)
    print(f"   {result}")

    # 流式生成
    print("\n3. 流式生成示例")
    prompt2 = "List 3 benefits of exercise:"
    print(f"   提示: {prompt2}")
    print("   输出: ", end="", flush=True)

    for token in model.stream_generate(prompt2, max_new_tokens=30):
        print(token, end="", flush=True)
    print()

    # 方法 2: 自动选择后端
    print("\n4. 自动选择后端")
    print("   使用 backend='auto' 自动选择最佳后端")
    model_auto = HLLM(MODEL_PATH, backend="auto")
    print(f"   自动选择的后端: {model_auto.get_info()['backend']}")

    print("\n" + "=" * 60)
    print("MLX 后端优势:")
    print("  - 专为 Apple Silicon 优化")
    print("  - 比 PyTorch MPS 快 2-5 倍")
    print("  - 内存占用减少 30-50%")
    print("  - 支持 4-bit/8-bit 量化")
    print("=" * 60)


if __name__ == "__main__":
    main()
