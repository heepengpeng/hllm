"""模型下载工具 - 支持 ModelScope 国内镜像"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 模型映射：HuggingFace -> ModelScope
MODELSCOPE_MAPPINGS = {
    # TinyLlama 系列
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "wangyueqian004/tinyllama-1.1b-chat-v1.0",
    "tinyllama-1.1b": "wangyueqian004/tinyllama-1.1b-chat-v1.0",
    
    # Llama 系列
    "meta-llama/Llama-2-7b-chat-hf": "modelscope/Llama-2-7b-chat-ms",
    "meta-llama/Llama-2-13b-chat-hf": "modelscope/Llama-2-13b-chat-ms",
    "meta-llama/Meta-Llama-3-8B-Instruct": "modelscope/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    "Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    
    # Qwen 系列
    "Qwen/Qwen2-7B-Instruct": "qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen1.5-7B-Chat": "qwen/Qwen1.5-7B-Chat",
    
    # ChatGLM 系列
    "THUDM/chatglm3-6b": "ZhipuAI/chatglm3-6b",
    "THUDM/chatglm2-6b": "ZhipuAI/chatglm2-6b",
    
    # Baichuan 系列
    "baichuan-inc/Baichuan2-7B-Chat": "baichuan-inc/Baichuan2-7B-Chat",
    
    # Yi 系列
    "01-ai/Yi-6B-Chat": "01ai/Yi-6B-Chat",
    "01-ai/Yi-34B-Chat": "01ai/Yi-34B-Chat",
    
    # InternLM 系列
    "internlm/internlm2-chat-7b": "Shanghai_AI_Laboratory/internlm2-chat-7b",
    
    # Mistral 系列
    "mistralai/Mistral-7B-Instruct-v0.2": "modelscope/Mistral-7B-Instruct-v0.2",
}


def get_modelscope_id(hf_model_id: str) -> Optional[str]:
    """获取 ModelScope 模型 ID
    
    Args:
        hf_model_id: HuggingFace 模型 ID
        
    Returns:
        ModelScope 模型 ID，如果没有映射则返回 None
    """
    # 直接匹配
    if hf_model_id in MODELSCOPE_MAPPINGS:
        return MODELSCOPE_MAPPINGS[hf_model_id]
    
    # 尝试部分匹配（简化名称）
    for key, value in MODELSCOPE_MAPPINGS.items():
        if key.lower() in hf_model_id.lower() or hf_model_id.lower() in key.lower():
            return value
    
    return None


def download_from_modelscope(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """从 ModelScope 下载模型
    
    Args:
        model_id: ModelScope 模型 ID
        cache_dir: 缓存目录
        local_dir: 本地保存目录（优先使用）
        revision: 模型版本
        
    Returns:
        模型本地路径
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        logger.info("ModelScope 未安装，正在安装...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
        from modelscope import snapshot_download
    
    logger.info(f"从 ModelScope 下载模型: {model_id}")
    
    kwargs = {"model_id": model_id}
    if local_dir:
        kwargs["local_dir"] = local_dir
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision
    
    path = snapshot_download(**kwargs)
    logger.info(f"模型下载完成: {path}")
    return path


def download_from_hf(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    use_mirror: bool = True,
) -> str:
    """从 HuggingFace 下载模型
    
    Args:
        model_id: HuggingFace 模型 ID
        cache_dir: 缓存目录
        local_dir: 本地保存目录
        use_mirror: 是否使用 hf-mirror.com 镜像
        
    Returns:
        模型本地路径
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("请安装 huggingface-hub: pip install huggingface-hub")
    
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        logger.info(f"使用 HF-Mirror 下载: {model_id}")
    else:
        logger.info(f"从 HuggingFace 下载: {model_id}")
    
    kwargs = {"repo_id": model_id}
    if local_dir:
        kwargs["local_dir"] = local_dir
        kwargs["local_dir_use_symlinks"] = False
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    
    path = snapshot_download(**kwargs)
    logger.info(f"模型下载完成: {path}")
    return path


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    use_modelscope: bool = True,
    use_hf_mirror: bool = True,
    prefer_modelscope: bool = True,
) -> str:
    """下载模型（自动选择最佳来源）
    
    优先级:
    1. ModelScope 国内镜像（最快）
    2. HF-Mirror 镜像
    3. 官方 HuggingFace
    
    Args:
        model_id: 模型 ID（支持 HuggingFace 格式）
        cache_dir: 缓存目录，默认为 ~/.cache/modelscope 或 ~/.cache/huggingface
        local_dir: 本地保存目录（优先于 cache_dir）
        use_modelscope: 是否尝试使用 ModelScope
        use_hf_mirror: 是否尝试使用 HF-Mirror
        prefer_modelscope: 是否优先使用 ModelScope
        
    Returns:
        模型本地路径
        
    Raises:
        RuntimeError: 所有下载方式都失败
    """
    # 检查是否是本地路径
    if os.path.exists(model_id):
        logger.info(f"使用本地模型: {model_id}")
        return model_id
    
    errors = []
    
    # 尝试 1: ModelScope
    if use_modelscope and prefer_modelscope:
        ms_id = get_modelscope_id(model_id)
        if ms_id:
            try:
                return download_from_modelscope(ms_id, cache_dir, local_dir)
            except Exception as e:
                errors.append(f"ModelScope: {e}")
                logger.warning(f"ModelScope 下载失败，尝试其他方式...")
    
    # 尝试 2: HF-Mirror
    if use_hf_mirror:
        try:
            return download_from_hf(model_id, cache_dir, local_dir, use_mirror=True)
        except Exception as e:
            errors.append(f"HF-Mirror: {e}")
            logger.warning(f"HF-Mirror 下载失败，尝试官方源...")
    
    # 尝试 3: 官方 HuggingFace
    try:
        return download_from_hf(model_id, cache_dir, local_dir, use_mirror=False)
    except Exception as e:
        errors.append(f"HuggingFace: {e}")
    
    # 尝试 4: ModelScope（非优先模式，尝试更多映射）
    if use_modelscope and not prefer_modelscope:
        ms_id = get_modelscope_id(model_id)
        if ms_id:
            try:
                return download_from_modelscope(ms_id, cache_dir, local_dir)
            except Exception as e:
                errors.append(f"ModelScope (fallback): {e}")
    
    # 所有方式都失败
    error_msg = "\n".join([f"  - {e}" for e in errors])
    raise RuntimeError(f"无法下载模型 '{model_id}'，所有方式都失败:\n{error_msg}")


def ensure_model(
    model_path: str,
    cache_dir: Optional[str] = None,
    **kwargs
) -> str:
    """确保模型可用，自动下载如果需要
    
    Args:
        model_path: 模型路径或 HuggingFace/ModelScope 模型 ID
        cache_dir: 缓存目录
        **kwargs: 传递给 download_model 的其他参数
        
    Returns:
        本地模型路径
    """
    # 如果是本地路径，直接返回
    if os.path.isdir(model_path):
        logger.info(f"使用本地模型: {model_path}")
        return model_path
    
    # 尝试下载
    return download_model(model_path, cache_dir=cache_dir, **kwargs)


# 便捷函数
def register_model_mapping(hf_id: str, ms_id: str):
    """注册新的模型映射
    
    Args:
        hf_id: HuggingFace 模型 ID
        ms_id: ModelScope 模型 ID
    """
    MODELSCOPE_MAPPINGS[hf_id] = ms_id
    logger.info(f"注册模型映射: {hf_id} -> {ms_id}")
