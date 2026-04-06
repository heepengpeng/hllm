"""
测试 model_downloader 模块
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os


class TestModelDownloadFunctions(unittest.TestCase):
    """测试模型下载函数"""

    def test_download_model_exists(self):
        """测试 download_model 函数存在"""
        from hllm.utils.model_downloader import download_model
        self.assertTrue(callable(download_model))

    def test_download_from_hf_exists(self):
        """测试 download_from_hf 函数存在"""
        from hllm.utils.model_downloader import download_from_hf
        self.assertTrue(callable(download_from_hf))

    def test_download_from_modelscope_exists(self):
        """测试 download_from_modelscope 函数存在"""
        from hllm.utils.model_downloader import download_from_modelscope
        self.assertTrue(callable(download_from_modelscope))

    def test_ensure_model_exists(self):
        """测试 ensure_model 函数存在"""
        from hllm.utils.model_downloader import ensure_model
        self.assertTrue(callable(ensure_model))

    def test_register_model_mapping_exists(self):
        """测试 register_model_mapping 函数存在"""
        from hllm.utils.model_downloader import register_model_mapping
        self.assertTrue(callable(register_model_mapping))


class TestModelMappings(unittest.TestCase):
    """测试模型映射"""

    def test_modelscope_mappings_exist(self):
        """测试 ModelScope 映射存在"""
        from hllm.utils.model_downloader import MODELSCOPE_MAPPINGS
        self.assertIsInstance(MODELSCOPE_MAPPINGS, dict)

    def test_modelscope_mappings_can_register(self):
        """测试可以注册新的映射"""
        from hllm.utils.model_downloader import register_model_mapping

        # 测试注册函数可以调用
        register_model_mapping("test/model", "test/model/id")


if __name__ == "__main__":
    unittest.main()
