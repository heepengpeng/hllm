#!/usr/bin/env python3
"""
使用 OpenAI 官方客户端测试 HLLM API

使用方法:
1. 先启动服务:
   python examples/api_server.py --model ./TinyLlama-1.1B-Chat-v1.0

2. 再运行此脚本:
   python test_openai_client.py
"""

import httpx
from openai import OpenAI


def main():
    # 创建自定义 httpx 客户端（禁用代理，避免 502 错误）
    http_client = httpx.Client(trust_env=False)
    
    # 创建客户端（与 OpenAI 兼容）
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="not-needed",  # HLLM 不验证 API key
        http_client=http_client  # 使用自定义客户端
    )

    print("=" * 60)
    print("OpenAI 客户端测试 HLLM API")
    print("=" * 60)

    # 1. 获取模型列表
    print("\n1. 获取模型列表")
    print("-" * 40)
    try:
        models = client.models.list()
        for model in models.data:
            print(f"  模型: {model.id}")
    except Exception as e:
        print(f"  错误: {e}")

    # 2. 对话补全（非流式）
    print("\n2. 对话补全（非流式）")
    print("-" * 40)
    try:
        response = client.chat.completions.create(
            model="hllm-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in 3 words"}
            ],
            max_tokens=20,
            temperature=0.7
        )
        print(f"  回复: {response.choices[0].message.content}")
        print(f"  使用: {response.usage}")
    except Exception as e:
        print(f"  错误: {e}")

    # 3. 对话补全（流式）
    print("\n3. 对话补全（流式）")
    print("-" * 40)
    try:
        print("  回复: ", end="", flush=True)
        for chunk in client.chat.completions.create(
            model="hllm-model",
            messages=[{"role": "user", "content": "Count: 1, 2"}],
            max_tokens=15,
            stream=True
        ):
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"  错误: {e}")

    # 4. 文本补全
    print("\n4. 文本补全")
    print("-" * 40)
    try:
        response = client.completions.create(
            model="hllm-model",
            prompt="Once upon a time",
            max_tokens=20
        )
        print(f"  补全: {response.choices[0].text}")
    except Exception as e:
        print(f"  错误: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
