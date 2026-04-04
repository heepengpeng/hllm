#!/usr/bin/env python3
"""使用 ModelScope 国内镜像下载模型（比 HF-Mirror 更快）"""

import os
import sys

# 设置 ModelScope 镜像
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')

def download_with_modelscope():
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("正在安装 modelscope...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope import snapshot_download
    
    model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    local_dir = "/Users/hp/CodeBuddy/hllm/models/llama-3.2-1b"
    
    print(f"正在从 ModelScope 下载: {model_id}")
    print(f"下载目录: {local_dir}")
    print("-" * 60)
    
    try:
        path = snapshot_download(
            model_id,
            cache_dir=local_dir,
            local_files_only=False
        )
        print(f"\n✅ 下载完成: {path}")
        return path
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return None

def download_with_hf_mirror():
    """备用：使用 HF-Mirror"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("正在安装 huggingface-hub...")
        os.system(f"{sys.executable} -m pip install huggingface-hub -q")
        from huggingface_hub import snapshot_download
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("使用 HF-Mirror 下载...")
    path = snapshot_download(
        repo_id='mlx-community/Llama-3.2-1B-Instruct-4bit',
        local_dir='/Users/hp/CodeBuddy/hllm/models/llama-3.2-1b',
        local_dir_use_symlinks=False,
        resume_download=True
    )
    return path

if __name__ == "__main__":
    # 先尝试 ModelScope
    result = download_with_modelscope()
    
    if not result:
        print("\n尝试备用方案...")
        result = download_with_hf_mirror()
    
    if result:
        print(f"\n模型已保存到: {result}")
    else:
        print("\n所有下载方式都失败了")
