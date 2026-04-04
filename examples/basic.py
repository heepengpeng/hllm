#!/usr/bin/env python3
"""HLLM 使用示例"""

from hllm import HLLM


def main():
    # 使用本地模型路径
    model_path = "./TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading model: {model_path}")
    print("This may take a few minutes on CPU...\n")

    # 初始化模型 (CPU)
    model = HLLM(model_path=model_path, device="cpu")

    # 测试生成
    prompt = "Write a short story about a robot."

    print(f"Prompt: {prompt}\n")
    print("Generating...\n")

    # 非流式生成
    result = model.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    print(f"Result:\n{result}\n")

    # 流式生成
    print("Streaming generate:")
    for token in model.stream_generate(
        prompt,
        max_new_tokens=50,
        temperature=0.7,
    ):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()