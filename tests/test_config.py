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

    def test_server_config_custom(self):
        """测试 ServerConfig 自定义值"""
        from hllm.config import ServerConfig

        config = ServerConfig(host="localhost", port=9000)
        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 9000)

    def test_generation_config_class_exists(self):
        """测试 GenerationConfig 类存在"""
        from hllm.config import GenerationConfig
        self.assertTrue(callable(GenerationConfig))


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

    def test_hllm_config_model_override(self):
        """测试覆盖模型配置"""
        from hllm.config import HLLMConfig

        config = HLLMConfig(model={"path": "custom/model"})
        self.assertEqual(config.model.path, "custom/model")


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


if __name__ == "__main__":
    unittest.main()
