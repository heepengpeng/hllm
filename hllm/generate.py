import torch
import torch.nn.functional as F
import logging
from typing import List, Generator
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    device: str = "cpu",
    **kwargs
) -> str:
    """
    生成文本（非流式）

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: nucleus sampling 概率
        top_k: top-k 采样
        repetition_penalty: 重复惩罚
        device: 运行设备

    Returns:
        生成的文本
    """
    logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")

    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    logger.info(f"Input encoded: {input_length} tokens")

    # 记录生成的 token IDs 用于惩罚
    generated_ids = input_ids.clone()

    # 生成循环
    for i in range(max_new_tokens):
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # 只取最后一个 token 的 logits

        # 应用重复惩罚
        if repetition_penalty != 1.0:
            _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k 采样
        if top_k > 0:
            logits = _top_k_filtering(logits, top_k)

        # Top-p (nucleus) 采样
        if top_p < 1.0:
            logits = _top_p_filtering(logits, top_p)

        # 采样下一个 token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 检查是否生成结束
        if next_token.item() == tokenizer.eos_token_id:
            logger.info(f"EOS token generated at step {i}")
            break

        # 追加到序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # 每10个token输出一次进度
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1} tokens")

    logger.info(f"Generation completed: {input_length - input_length} new tokens")

    # 解码输出（排除输入 prompt）
    output_text = tokenizer.decode(input_ids[0][input_length:], skip_special_tokens=True)
    return output_text


def stream_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    device: str = "cpu",
    **kwargs
) -> Generator[str, None, None]:
    """
    流式生成文本（yield token）

    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: nucleus sampling 概率
        top_k: top-k 采样
        repetition_penalty: 重复惩罚
        device: 运行设备

    Yields:
        逐个生成的 token
    """
    logger.info(f"Starting streaming generation with max_new_tokens={max_new_tokens}")

    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    logger.info(f"Input encoded: {input_length} tokens")

    # 记录生成的 token IDs 用于惩罚
    generated_ids = input_ids.clone()

    # 生成循环
    for i in range(max_new_tokens):
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

        # 应用重复惩罚
        if repetition_penalty != 1.0:
            _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k 采样
        if top_k > 0:
            logits = _top_k_filtering(logits, top_k)

        # Top-p (nucleus) 采样
        if top_p < 1.0:
            logits = _top_p_filtering(logits, top_p)

        # 采样下一个 token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 检查是否生成结束
        if next_token.item() == tokenizer.eos_token_id:
            logger.info(f"EOS token generated at step {i}")
            break

        # 追加到序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # yield 生成的 token
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        if token_text:  # 跳过空 token
            # 每10个token输出一次进度
            if (i + 1) % 10 == 0:
                logger.info(f"Streamed {i + 1} tokens")
            yield token_text

    logger.info("Streaming generation completed")


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float):
    """应用重复惩罚"""
    for token_id in generated_ids.unique():
        logits[0, token_id] = logits[0, token_id] / penalty


def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k 过滤"""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float("-inf")
    return logits


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p (nucleus) 过滤"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float("-inf")
    return logits