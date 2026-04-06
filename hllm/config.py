"""
HLLM 统一配置管理

使用 Pydantic Settings 管理所有配置，支持：
- 环境变量
- 配置文件
- 命令行参数
- 默认值
"""

import os
from typing import Optional, Literal, List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """模型配置"""
    model_config = SettingsConfigDict(
        env_prefix="HLLM_MODEL_",
        extra="ignore"
    )
    
    path: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="模型路径或 HuggingFace model ID"
    )
    backend: Literal["auto", "pytorch", "mlx", "paged_pytorch"] = Field(
        default="auto",
        description="推理后端"
    )
    device: Optional[Literal["cpu", "cuda", "mps"]] = Field(
        default=None,
        description="设备 (PyTorch backend)"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="是否信任远程代码"
    )


class ServerConfig(BaseSettings):
    """服务器配置"""
    model_config = SettingsConfigDict(
        env_prefix="HLLM_SERVER_",
        extra="ignore"
    )
    
    host: str = Field(
        default="0.0.0.0",
        description="监听地址"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="监听端口"
    )
    reload: bool = Field(
        default=False,
        description="是否启用热重载"
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="工作进程数"
    )
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1024 and os.getuid() != 0:
            raise ValueError("Ports < 1024 require root privileges")
        return v


class GenerationConfig(BaseSettings):
    """生成配置"""
    model_config = SettingsConfigDict(
        env_prefix="HLLM_GEN_",
        extra="ignore"
    )
    
    max_new_tokens: int = Field(
        default=100,
        ge=1,
        le=4096,
        description="最大生成 token 数"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="采样温度"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="核采样概率阈值"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k 采样"
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="重复惩罚"
    )
    stop_sequences: List[str] = Field(
        default_factory=list,
        description="停止序列"
    )


class PagedAttentionConfig(BaseSettings):
    """PagedAttention 配置"""
    model_config = SettingsConfigDict(
        env_prefix="HLLM_PAGED_",
        extra="ignore"
    )
    
    enabled: bool = Field(
        default=True,
        description="是否启用 PagedAttention"
    )
    block_size: int = Field(
        default=16,
        ge=8,
        le=256,
        description="KV cache 块大小"
    )
    num_blocks: int = Field(
        default=1024,
        ge=64,
        description="总块数"
    )
    max_num_seqs: int = Field(
        default=256,
        ge=1,
        description="最大并发序列数"
    )


class LoggingConfig(BaseSettings):
    """日志配置"""
    model_config = SettingsConfigDict(
        env_prefix="HLLM_LOG_",
        extra="ignore"
    )
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="日志级别"
    )
    format: str = Field(
        default="[%(asctime)s] %(levelname)s - %(message)s",
        description="日志格式"
    )
    file: Optional[str] = Field(
        default=None,
        description="日志文件路径"
    )


class HLLMConfig(BaseSettings):
    """
    HLLM 统一配置
    
    支持从以下位置加载配置（优先级从高到低）：
    1. 命令行参数
    2. 环境变量
    3. 配置文件 (.env, .yaml)
    4. 默认值
    
    环境变量格式:
    - HLLM_MODEL_PATH=microsoft/Phi-3-mini-4k-instruct
    - HLLM_SERVER_PORT=8080
    - HLLM_GEN_TEMPERATURE=0.8
    """
    model_config = SettingsConfigDict(
        env_prefix="HLLM_",
        env_nested_delimiter="__",
        yaml_file="hllm.yaml",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    paged_attention: PagedAttentionConfig = Field(default_factory=PagedAttentionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "HLLMConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
    
    def setup_logging(self) -> None:
        """配置日志系统"""
        import logging
        
        handlers = [logging.StreamHandler()]
        if self.logging.file:
            handlers.append(logging.FileHandler(self.logging.file))
        
        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
            handlers=handlers
        )


@lru_cache()
def get_config() -> HLLMConfig:
    """
    获取全局配置（单例）
    
    Returns:
        HLLMConfig 实例
        
    Example:
        >>> from hllm.config import get_config
        >>> config = get_config()
        >>> print(config.model.path)
        >>> print(config.server.port)
    """
    return HLLMConfig()


def reload_config() -> HLLMConfig:
    """重新加载配置"""
    get_config.cache_clear()
    return get_config()


# 兼容性：保留旧的导入方式
__all__ = [
    "HLLMConfig",
    "ModelConfig",
    "ServerConfig",
    "GenerationConfig",
    "PagedAttentionConfig",
    "LoggingConfig",
    "get_config",
    "reload_config",
]