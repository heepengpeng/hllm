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


if __name__ == "__main__":
    unittest.main()
