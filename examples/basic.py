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

    # TinyLlama 使用 chat template，需要按照指定格式
    # 格式: <|user|>\n{prompt}<|eot_id|><|assistant|>\n
    prompt = "Write a short story about a robot."

    # 使用 tokenizer 的 apply_chat_template 构建正确的输入
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"Prompt: {prompt}\n")
    print(f"Formatted: {prompt_formatted[:100]}...\n")
    print("Generating...\n")

    # 非流式生成
    result = model.generate(
        prompt_formatted,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    print(f"Result:\n{result}\n")


if __name__ == "__main__":
    main()