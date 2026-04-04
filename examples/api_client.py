#!/usr/bin/env python3
"""
HLLM OpenAI Compatible API Client Example

演示如何使用 HLLMClient 或 OpenAI 官方客户端访问 HLLM API 服务。

首先启动服务端:
    python api_server.py --model ./TinyLlama-1.1B-Chat-v1.0

然后运行客户端:
    python api_client.py
"""

import sys

from hllm.client import HLLMClient


def demo_with_hllm_client():
    """使用 HLLMClient（推荐）"""
    print("\n" + "=" * 60)
    print("使用 HLLMClient（OpenAI 兼容格式）")
    print("=" * 60)

    base_url = "http://localhost:8000"
    print(f"连接至: {base_url}")

    try:
        with HLLMClient(base_url) as client:
            # 1. 获取模型列表
            print("\n1. 模型列表")
            print("-" * 40)
            models = client.models.list()
            for model in models["data"]:
                print(f"  - {model['id']}")

            # 2. 对话（非流式）
            print("\n2. 对话 - 非流式")
            print("-" * 40)
            response = client.chat.completions.create(
                model="hllm-model",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is Python?"}
                ],
                max_tokens=100,
                temperature=0.7
            )
            print(f"Response: {response.choices[0].message.content}")
            print(f"Tokens: {response.usage}")

            # 3. 对话（流式）
            print("\n3. 对话 - 流式")
            print("-" * 40)
            print("Response: ", end="", flush=True)
            for chunk in client.chat.completions.create(
                model="hllm-model",
                messages=[{"role": "user", "content": "Say hi!"}],
                max_tokens=50,
                stream=True
            ):
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()

            # 4. 文本补全
            print("\n4. 文本补全")
            print("-" * 40)
            response = client.completions.create(
                model="hllm-model",
                prompt="Once upon a time",
                max_tokens=50
            )
            print(f"Completion: {response.choices[0].text}")

    except Exception as e:
        print(f"错误: {e}")
        print("请确保服务端已启动: python api_server.py --model <model_path>")
        sys.exit(1)


def demo_with_openai_client():
    """使用 OpenAI 官方客户端"""
    print("\n" + "=" * 60)
    print("使用 OpenAI 官方客户端")
    print("=" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("请先安装 OpenAI 客户端: pip install openai")
        return

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"  # HLLM 不验证 API key
    )

    try:
        # 对话
        print("\n对话示例")
        print("-" * 40)
        response = client.chat.completions.create(
            model="hllm-model",
            messages=[
                {"role": "user", "content": "Hello! How are you?"}
            ],
            max_tokens=50
        )
        print(f"Response: {response.choices[0].message.content}")

        # 流式对话
        print("\n流式对话示例")
        print("-" * 40)
        print("Response: ", end="", flush=True)
        for chunk in client.chat.completions.create(
            model="hllm-model",
            messages=[{"role": "user", "content": "Count to 5"}],
            max_tokens=30,
            stream=True
        ):
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()

    except Exception as e:
        print(f"错误: {e}")


def demo_with_requests():
    """使用原始 HTTP 请求"""
    print("\n" + "=" * 60)
    print("使用原始 HTTP 请求 (requests)")
    print("=" * 60)

    import requests

    base_url = "http://localhost:8000"

    try:
        # 健康检查
        print("\n1. 健康检查")
        print("-" * 40)
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.json()}")

        # 模型列表
        print("\n2. 模型列表")
        print("-" * 40)
        response = requests.get(f"{base_url}/v1/models")
        print(f"Models: {response.json()}")

        # 对话
        print("\n3. 对话")
        print("-" * 40)
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "hllm-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 30
            }
        )
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Usage: {result['usage']}")

    except Exception as e:
        print(f"错误: {e}")


def main():
    print("HLLM OpenAI Compatible API Client Demo")
    print("=" * 60)

    # 运行各种客户端示例
    demo_with_hllm_client()
    demo_with_openai_client()
    demo_with_requests()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
