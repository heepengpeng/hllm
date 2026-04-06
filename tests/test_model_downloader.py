"""
测试 model_downloader 模块 - Mock 覆盖
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestModelDownloadFunctions:
    """测试模型下载函数"""

    def test_download_model_exists(self):
        """测试 download_model 函数存在"""
        from hllm.utils.model_downloader import download_model
        assert callable(download_model)

    def test_download_from_hf_exists(self):
        """测试 download_from_hf 函数存在"""
        from hllm.utils.model_downloader import download_from_hf
        assert callable(download_from_hf)

    def test_download_from_modelscope_exists(self):
        """测试 download_from_modelscope 函数存在"""
        from hllm.utils.model_downloader import download_from_modelscope
        assert callable(download_from_modelscope)

    def test_ensure_model_exists(self):
        """测试 ensure_model 函数存在"""
        from hllm.utils.model_downloader import ensure_model
        assert callable(ensure_model)

    def test_register_model_mapping_exists(self):
        """测试 register_model_mapping 函数存在"""
        from hllm.utils.model_downloader import register_model_mapping
        assert callable(register_model_mapping)


class TestModelMappings:
    """测试模型映射"""

    def test_modelscope_mappings_exist(self):
        """测试 ModelScope 映射存在"""
        from hllm.utils.model_downloader import MODELSCOPE_MAPPINGS
        assert isinstance(MODELSCOPE_MAPPINGS, dict)

    def test_modelscope_mappings_type(self):
        """测试映射是字典"""
        from hllm.utils.model_downloader import MODELSCOPE_MAPPINGS
        # 验证是可迭代的字典
        for key, value in MODELSCOPE_MAPPINGS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            break  # 只检查第一个

    def test_register_mapping(self):
        """测试注册映射"""
        from hllm.utils.model_downloader import register_model_mapping

        # 测试函数可调用
        register_model_mapping("test/model", "test/model/scope")


class TestDownloaderHelpers:
    """测试下载器辅助函数"""

    def test_get_modelscope_id_function(self):
        """测试 get_modelscope_id 函数存在"""
        from hllm.utils.model_downloader import get_modelscope_id
        assert callable(get_modelscope_id)

    def test_download_model_function(self):
        """测试 download_model 可以被调用"""
        from hllm.utils.model_downloader import download_model
        # 验证函数签名
        import inspect
        sig = inspect.signature(download_model)
        params = list(sig.parameters.keys())
        assert "model_id" in params


class TestDownloaderLogging:
    """测试下载器日志"""

    def test_logger_exists(self):
        """测试 logger 存在"""
        from hllm.utils.model_downloader import logger
        assert logger is not None

    def test_logger_is_logging_logger(self):
        """测试是 logging.Logger"""
        import logging
        from hllm.utils.model_downloader import logger
        assert isinstance(logger, logging.Logger)
