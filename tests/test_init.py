"""
测试 hllm 包初始化
"""

import unittest


class TestHLLMInit(unittest.TestCase):
    """测试 HLLM 包初始化"""

    def test_version(self):
        """测试版本号"""
        import hllm
        self.assertIsNotNone(hllm.__version__)

    def test_version_format(self):
        """测试版本号格式"""
        import hllm
        version = hllm.__version__
        # 版本号应该是 X.Y.Z 格式
        parts = version.split(".")
        self.assertGreaterEqual(len(parts), 2)

    def test_exports(self):
        """测试导出的符号"""
        import hllm

        expected_exports = ["HLLM", "Tokenizer", "generate"]
        for export in expected_exports:
            self.assertIn(export, hllm.__all__)
            self.assertTrue(hasattr(hllm, export))

    def test_hllm_class(self):
        """测试 HLLM 类"""
        from hllm import HLLM
        self.assertIsNotNone(HLLM)

    def test_tokenizer_class(self):
        """测试 Tokenizer 类"""
        from hllm import Tokenizer
        self.assertIsNotNone(Tokenizer)

    def test_generate_function(self):
        """测试 generate 函数"""
        from hllm import generate
        self.assertTrue(callable(generate))

    def test_config_exports(self):
        """测试配置类导出"""
        from hllm import HLLMConfig, ModelConfig, ServerConfig, GenerationConfig
        self.assertTrue(callable(HLLMConfig))
        self.assertTrue(callable(ModelConfig))
        self.assertTrue(callable(ServerConfig))
        self.assertTrue(callable(GenerationConfig))

    def test_get_config_function(self):
        """测试 get_config 函数"""
        from hllm import get_config
        self.assertTrue(callable(get_config))

    def test_reload_config_function(self):
        """测试 reload_config 函数"""
        from hllm import reload_config
        self.assertTrue(callable(reload_config))


class TestOptionalImports(unittest.TestCase):
    """测试可选导入"""

    def test_server_import(self):
        """测试 Server 可选导入"""
        try:
            from hllm import Server
            self.assertIsNotNone(Server)
        except ImportError:
            # FastAPI 可能未安装
            pass

    def test_client_import(self):
        """测试 HLLMClient 可选导入"""
        try:
            from hllm import HLLMClient
            self.assertIsNotNone(HLLMClient)
        except ImportError:
            # requests 可能未安装
            pass

    def test_server_in_all_if_available(self):
        """测试 Server 在 __all__ 中（如果可用）"""
        import hllm
        try:
            from hllm import Server
            self.assertIn("Server", hllm.__all__)
        except ImportError:
            # FastAPI 未安装，Server 不在 __all__ 中
            pass

    def test_client_in_all_if_available(self):
        """测试 HLLMClient 在 __all__ 中（如果可用）"""
        import hllm
        try:
            from hllm import HLLMClient
            self.assertIn("HLLMClient", hllm.__all__)
        except ImportError:
            # requests 未安装，HLLMClient 不在 __all__ 中
            pass


if __name__ == "__main__":
    unittest.main()
