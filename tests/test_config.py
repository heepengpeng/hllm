"""
HLLM 配置系统单元测试
"""

import os
import pytest
from pydantic import ValidationError

from hllm.config import (
    HLLMConfig,
    ModelConfig,
    ServerConfig,
    GenerationConfig,
    PagedAttentionConfig,
    LoggingConfig,
    get_config,
    reload_config,
)


class TestModelConfig:
    """ModelConfig 测试"""

    def test_default_values(self):
        """测试默认值"""
        config = ModelConfig()
        assert config.path == "microsoft/Phi-3-mini-4k-instruct"
        assert config.backend == "auto"
        assert config.device is None
        assert config.trust_remote_code is False

    def test_custom_values(self):
        """测试自定义值"""
        config = ModelConfig(
            path="meta-llama/Llama-3.2-1B",
            backend="pytorch",
            device="cuda",
            trust_remote_code=True
        )
        assert config.path == "meta-llama/Llama-3.2-1B"
        assert config.backend == "pytorch"
        assert config.device == "cuda"
        assert config.trust_remote_code is True

    def test_invalid_backend(self):
        """测试无效后端"""
        with pytest.raises(ValidationError):
            ModelConfig(backend="invalid")


class TestServerConfig:
    """ServerConfig 测试"""

    def test_default_values(self):
        """测试默认值"""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.reload is False
        assert config.workers == 1

    def test_port_range(self):
        """测试端口范围"""
        # 有效端口
        config = ServerConfig(port=8080)
        assert config.port == 8080

        # 无效端口 (小于 1)
        with pytest.raises(ValidationError):
            ServerConfig(port=0)

        # 无效端口 (大于 65535)
        with pytest.raises(ValidationError):
            ServerConfig(port=70000)


class TestGenerationConfig:
    """GenerationConfig 测试"""

    def test_default_values(self):
        """测试默认值"""
        config = GenerationConfig()
        assert config.max_new_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0

    def test_temperature_range(self):
        """测试温度范围"""
        # 有效值
        config = GenerationConfig(temperature=1.5)
        assert config.temperature == 1.5

        # 无效值 (小于 0)
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.1)

        # 无效值 (大于 2)
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=2.1)

    def test_top_p_range(self):
        """测试 top_p 范围"""
        # 边界值
        config = GenerationConfig(top_p=0.0)
        assert config.top_p == 0.0

        config = GenerationConfig(top_p=1.0)
        assert config.top_p == 1.0

        # 无效值 (大于 1)
        with pytest.raises(ValidationError):
            GenerationConfig(top_p=1.1)


class TestPagedAttentionConfig:
    """PagedAttentionConfig 测试"""

    def test_default_values(self):
        """测试默认值"""
        config = PagedAttentionConfig()
        assert config.enabled is True
        assert config.block_size == 16
        assert config.num_blocks == 1024
        assert config.max_num_seqs == 256

    def test_block_size_range(self):
        """测试 block_size 范围"""
        # 有效值
        config = PagedAttentionConfig(block_size=32)
        assert config.block_size == 32

        # 无效值 (小于 8)
        with pytest.raises(ValidationError):
            PagedAttentionConfig(block_size=4)


class TestHLLMConfig:
    """HLLMConfig 测试"""

    def test_default_values(self):
        """测试默认值"""
        config = HLLMConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.paged_attention, PagedAttentionConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_nested_config(self):
        """测试嵌套配置"""
        config = HLLMConfig(
            model={"path": "test-model", "backend": "pytorch"},
            server={"host": "127.0.0.1", "port": 9000}
        )
        assert config.model.path == "test-model"
        assert config.model.backend == "pytorch"
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000

    def test_to_dict(self):
        """测试转换为字典"""
        config = HLLMConfig()
        data = config.model_dump()
        assert "model" in data
        assert "server" in data
        assert data["model"]["path"] == "microsoft/Phi-3-mini-4k-instruct"


class TestEnvironmentVariables:
    """环境变量测试"""

    def test_env_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("HLLM_MODEL_PATH", "env-model")
        monkeypatch.setenv("HLLM_SERVER_PORT", "9999")
        
        config = reload_config()
        assert config.model.path == "env-model"
        assert config.server.port == 9999
        
        # 清理缓存
        reload_config()

    def test_nested_env(self, monkeypatch):
        """测试嵌套环境变量 (使用下划线格式)"""
        monkeypatch.setenv("HLLM_MODEL_BACKEND", "mlx")
        monkeypatch.setenv("HLLM_GEN_TEMPERATURE", "0.5")
        
        config = reload_config()
        assert config.model.backend == "mlx"
        assert config.generation.temperature == 0.5
        
        # 清理
        del os.environ["HLLM_MODEL_BACKEND"]
        del os.environ["HLLM_GEN_TEMPERATURE"]
        reload_config()


class TestSingleton:
    """单例模式测试"""

    def test_singleton(self):
        """测试配置单例"""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reload(self):
        """测试重新加载"""
        config1 = get_config()
        config2 = reload_config()
        # 重新加载后应该是不同的对象
        assert config1 is not config2


class TestYAMLConfig:
    """YAML 配置测试"""

    def test_to_yaml(self, tmp_path):
        """测试保存到 YAML"""
        config = HLLMConfig(
            model={"path": "yaml-model"},
            server={"port": 8888}
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(str(yaml_path))
        
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "yaml-model" in content
        assert "8888" in content

    def test_from_yaml(self, tmp_path):
        """测试从 YAML 加载"""
        yaml_content = """
model:
  path: loaded-model
  backend: mlx
server:
  port: 7777
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)
        
        config = HLLMConfig.from_yaml(str(yaml_path))
        assert config.model.path == "loaded-model"
        assert config.model.backend == "mlx"
        assert config.server.port == 7777


if __name__ == "__main__":
    pytest.main([__file__, "-v"])