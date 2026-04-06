"""
测试 config 模块
"""

import unittest
from unittest.mock import patch, mock_open
import os


class TestConfigClasses(unittest.TestCase):
    """测试配置类"""

    def test_model_config_class_exists(self):
        """测试 ModelConfig 类存在"""
        from hllm.config import ModelConfig
        self.assertTrue(callable(ModelConfig))

    def test_model_config_defaults(self):
        """测试 ModelConfig 默认值"""
        from hllm.config import ModelConfig

        config = ModelConfig()
        self.assertEqual(config.path, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(config.backend, "auto")

    def test_model_config_custom(self):
        """测试 ModelConfig 自定义值"""
        from hllm.config import ModelConfig

        config = ModelConfig(path="custom/model", backend="pytorch")
        self.assertEqual(config.path, "custom/model")
        self.assertEqual(config.backend, "pytorch")

    def test_model_config_device(self):
        """测试 ModelConfig 设备选项"""
        from hllm.config import ModelConfig

        config = ModelConfig(device="cuda")
        self.assertEqual(config.device, "cuda")

    def test_model_config_trust_remote_code(self):
        """测试 ModelConfig trust_remote_code"""
        from hllm.config import ModelConfig

        config = ModelConfig(trust_remote_code=True)
        self.assertTrue(config.trust_remote_code)

    def test_server_config_class_exists(self):
        """测试 ServerConfig 类存在"""
        from hllm.config import ServerConfig
        self.assertTrue(callable(ServerConfig))

    def test_server_config_defaults(self):
        """测试 ServerConfig 默认值"""
        from hllm.config import ServerConfig

        config = ServerConfig()
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.reload, False)
        self.assertEqual(config.workers, 1)

    def test_server_config_custom(self):
        """测试 ServerConfig 自定义值"""
        from hllm.config import ServerConfig

        config = ServerConfig(host="localhost", port=9000)
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 9000)

    def test_server_config_reload(self):
        """测试 ServerConfig reload 选项"""
        from hllm.config import ServerConfig

        config = ServerConfig(reload=True)
        self.assertTrue(config.reload)

    def test_server_config_workers(self):
        """测试 ServerConfig workers 选项"""
        from hllm.config import ServerConfig

        config = ServerConfig(workers=4)
        self.assertEqual(config.workers, 4)

    def test_generation_config_class_exists(self):
        """测试 GenerationConfig 类存在"""
        from hllm.config import GenerationConfig
        self.assertTrue(callable(GenerationConfig))

    def test_generation_config_defaults(self):
        """测试 GenerationConfig 默认值"""
        from hllm.config import GenerationConfig

        config = GenerationConfig()
        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.repetition_penalty, 1.0)

    def test_generation_config_custom(self):
        """测试 GenerationConfig 自定义值"""
        from hllm.config import GenerationConfig

        config = GenerationConfig(
            max_new_tokens=200,
            temperature=1.0,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.2
        )
        self.assertEqual(config.max_new_tokens, 200)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.top_k, 100)
        self.assertEqual(config.repetition_penalty, 1.2)

    def test_generation_config_stop_sequences(self):
        """测试 GenerationConfig 停止序列"""
        from hllm.config import GenerationConfig

        config = GenerationConfig(stop_sequences=["###", "END"])
        self.assertEqual(config.stop_sequences, ["###", "END"])


class TestPagedAttentionConfig(unittest.TestCase):
    """测试 PagedAttentionConfig"""

    def test_paged_attention_config_exists(self):
        """测试 PagedAttentionConfig 类存在"""
        from hllm.config import PagedAttentionConfig
        self.assertTrue(callable(PagedAttentionConfig))

    def test_paged_attention_config_defaults(self):
        """测试 PagedAttentionConfig 默认值"""
        from hllm.config import PagedAttentionConfig

        config = PagedAttentionConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.num_blocks, 1024)
        self.assertEqual(config.max_num_seqs, 256)

    def test_paged_attention_config_custom(self):
        """测试 PagedAttentionConfig 自定义值"""
        from hllm.config import PagedAttentionConfig

        config = PagedAttentionConfig(
            enabled=False,
            block_size=32,
            num_blocks=2048,
            max_num_seqs=512
        )
        self.assertFalse(config.enabled)
        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.num_blocks, 2048)
        self.assertEqual(config.max_num_seqs, 512)


class TestLoggingConfig(unittest.TestCase):
    """测试 LoggingConfig"""

    def test_logging_config_exists(self):
        """测试 LoggingConfig 类存在"""
        from hllm.config import LoggingConfig
        self.assertTrue(callable(LoggingConfig))

    def test_logging_config_defaults(self):
        """测试 LoggingConfig 默认值"""
        from hllm.config import LoggingConfig

        config = LoggingConfig()
        self.assertEqual(config.level, "INFO")
        self.assertEqual(config.format, "[%(asctime)s] %(levelname)s - %(message)s")
        self.assertIsNone(config.file)

    def test_logging_config_custom(self):
        """测试 LoggingConfig 自定义值"""
        from hllm.config import LoggingConfig

        config = LoggingConfig(
            level="DEBUG",
            format="custom-format",
            file="/var/log/hllm.log"
        )
        self.assertEqual(config.level, "DEBUG")
        self.assertEqual(config.format, "custom-format")
        self.assertEqual(config.file, "/var/log/hllm.log")


class TestHLLMConfig(unittest.TestCase):
    """测试 HLLMConfig"""

    def test_hllm_config_class_exists(self):
        """测试 HLLMConfig 类存在"""
        from hllm.config import HLLMConfig
        self.assertTrue(callable(HLLMConfig))

    def test_hllm_config_nested(self):
        """测试 HLLMConfig 嵌套配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig()
        self.assertTrue(hasattr(config, 'model'))
        self.assertTrue(hasattr(config, 'server'))
        self.assertTrue(hasattr(config, 'generation'))
        self.assertTrue(hasattr(config, 'paged_attention'))
        self.assertTrue(hasattr(config, 'logging'))

    def test_hllm_config_model_override(self):
        """测试覆盖模型配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig(model={"path": "custom/model"})
        self.assertEqual(config.model.path, "custom/model")

    def test_hllm_config_server_override(self):
        """测试覆盖服务器配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig(server={"port": 9000})
        self.assertEqual(config.server.port, 9000)

    def test_hllm_config_generation_override(self):
        """测试覆盖生成配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig(generation={"temperature": 1.5})
        self.assertEqual(config.generation.temperature, 1.5)

    def test_hllm_config_paged_attention_override(self):
        """测试覆盖 PagedAttention 配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig(paged_attention={"enabled": False})
        self.assertFalse(config.paged_attention.enabled)

    def test_hllm_config_logging_override(self):
        """测试覆盖日志配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig(logging={"level": "DEBUG"})
        self.assertEqual(config.logging.level, "DEBUG")

    def test_hllm_config_to_yaml(self):
        """测试 HLLMConfig 导出 YAML"""
        from hllm.config import HLLMConfig
        import tempfile
        import yaml

        config = HLLMConfig(model={"path": "test/model"})
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            with open(f.name) as fp:
                data = yaml.safe_load(fp)
        self.assertIn('model', data)

    def test_hllm_config_setup_logging(self):
        """测试 HLLMConfig 日志设置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig()
        # 不应抛出异常
        config.setup_logging()

    def test_hllm_config_setup_logging_with_file(self):
        """测试 HLLMConfig 带文件的日志设置"""
        from hllm.config import HLLMConfig
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            config = HLLMConfig(logging={"file": f.name})
            # 不应抛出异常
            config.setup_logging()

    def test_hllm_config_from_yaml(self):
        """测试从 YAML 加载配置"""
        from hllm.config import HLLMConfig
        import tempfile
        import os

        yaml_content = """model:
  path: test/model
server:
  port: 9000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = HLLMConfig.from_yaml(f.name)
            os.unlink(f.name)

        self.assertEqual(config.model.path, "test/model")
        self.assertEqual(config.server.port, 9000)


class TestConfigFunctions(unittest.TestCase):
    """测试配置函数"""

    def test_get_config_exists(self):
        """测试 get_config 函数存在"""
        from hllm.config import get_config
        self.assertTrue(callable(get_config))

    def test_reload_config_exists(self):
        """测试 reload_config 函数存在"""
        from hllm.config import reload_config
        self.assertTrue(callable(reload_config))

    def test_get_config_returns_instance(self):
        """测试 get_config 返回配置实例"""
        from hllm.config import get_config

        config = get_config()
        self.assertIsNotNone(config)

    def test_reload_config(self):
        """测试重新加载配置"""
        from hllm.config import reload_config

        # 不应抛出异常
        config = reload_config()
        self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main()
