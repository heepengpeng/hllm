#!/usr/bin/env python3
"""
HLLM REST API 客户端示例

演示如何使用 Python 客户端访问 HLLM API 服务。

首先启动服务端:
    python api_server.py --model ./TinyLlama-1.1B-Chat-v1.0

然后运行客户端:
    python api_client.py
"""

import sys

from hllm.client import HLLMClient


def demo_health_check(client: HLLMClient):
    """演示健康检查"""
    print("\n" + "=" * 50)
    print("1. 健康检查")
    print("=" * 50)
    
    try:
        health = client.health()
        print(f"状态: {health['status']}")
        print(f"模型: {health['model']}")
        print(f"设备: {health['device']}")
    except Exception as e:
        print(f"错误: {e}")
        print("请确保服务端已启动: python api_server.py --model <model_path>")
        sys.exit(1)


def demo_model_info(client: HLLMClient):
    """演示获取模型信息"""
    print("\n" + "=" * 50)
    print("2. 模型信息")
    print("=" * 50)
    
    info = client.get_models()
    print(f"模型名称: {info['model']}")
    print(f"设备: {info['device']}")
    print(f"最大长度: {info.get('max_length', 'N/A')}")
    print(f"词表大小: {info.get('vocab_size', 'N/A')}")


def demo_generate(client: HLLMClient):
    """演示文本生成"""
    print("\n" + "=" * 50)
    print("3. 文本生成")
    print("=" * 50)
    
    prompt = "Write a short greeting:"
    print(f"提示: {prompt}")
    print("-" * 50)
    
    response = client.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7
    )
    
    print(f"生成结果:\n{response.text}")
    print("-" * 50)
    print(f"Token 使用: {response.usage}")


def demo_chat(client: HLLMClient):
    """演示对话"""
    print("\n" + "=" * 50)
    print("4. 对话生成")
    print("=" * 50)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    print("对话历史:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print("-" * 50)
    
    response = client.chat(
        messages=messages,
        max_new_tokens=100,
        temperature=0.7
    )
    
    print(f"助手回复:\n{response.message.content}")
    print("-" * 50)
    print(f"Token 使用: {response.usage}")


def demo_stream(client: HLLMClient):
    """演示流式生成"""
    print("\n" + "=" * 50)
    print("5. 流式生成")
    print("=" * 50)
    
    prompt = "Count from 1 to 5:"
    print(f"提示: {prompt}")
    print("流式输出: ", end="", flush=True)
    
    for chunk in client.generate_stream(
        prompt=prompt,
        max_new_tokens=20
    ):
        print(chunk.token, end="", flush=True)
    
    print()  # 换行


def demo_simple_chat(client: HLLMClient):
    """演示简化对话"""
    print("\n" + "=" * 50)
    print("6. 简化对话")
    print("=" * 50)
    
    response = client.chat_simple(
        "What is Python?",
        system="You are a programming expert.",
        max_new_tokens=80
    )
    
    print(f"回复: {response}")


def main():
    # 创建客户端
    base_url = "http://localhost:8000"
    print(f"连接至: {base_url}")
    
    with HLLMClient(base_url) as client:
        # 运行演示
        demo_health_check(client)
        demo_model_info(client)
        demo_generate(client)
        demo_chat(client)
        demo_stream(client)
        demo_simple_chat(client)
        
        print("\n" + "=" * 50)
        print("所有演示完成!")
        print("=" * 50)


if __name__ == "__main__":
    main()
