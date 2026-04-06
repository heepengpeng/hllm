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


class TestGetModelscopeId:
    """测试 get_modelscope_id 函数"""

    def test_direct_mapping(self):
        """测试直接映射"""
        from hllm.utils.model_downloader import get_modelscope_id

        result = get_modelscope_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert result == "wangyueqian004/tinyllama-1.1b-chat-v1.0"

    def test_short_name_mapping(self):
        """测试短名称映射"""
        from hllm.utils.model_downloader import get_modelscope_id

        result = get_modelscope_id("tinyllama-1.1b")
        assert result == "wangyueqian004/tinyllama-1.1b-chat-v1.0"

    def test_no_mapping(self):
        """测试无映射返回 None"""
        from hllm.utils.model_downloader import get_modelscope_id

        result = get_modelscope_id("nonexistent/model-xyz")
        assert result is None

    def test_case_insensitive_partial_match(self):
        """测试大小写不敏感的部分匹配"""
        from hllm.utils.model_downloader import get_modelscope_id

        result = get_modelscope_id("Qwen/Qwen2-7B-Instruct")
        assert result == "qwen/Qwen2-7B-Instruct"

    def test_partial_match_reverse(self):
        """测试反向部分匹配"""
        from hllm.utils.model_downloader import get_modelscope_id

        result = get_modelscope_id("qwen")
        assert result is not None


class TestEnsureModel:
    """测试 ensure_model 函数"""

    def test_local_path_returns_directly(self):
        """测试本地路径直接返回"""
        from hllm.utils.model_downloader import ensure_model

        with patch('hllm.utils.model_downloader.os.path.isdir') as mock_isdir:
            mock_isdir.return_value = True
            
            result = ensure_model("/local/model/path")
            
            assert result == "/local/model/path"
            # 不应调用 download_model
            mock_isdir.assert_called_once()

    def test_remote_model_calls_download(self):
        """测试远程模型调用下载"""
        from hllm.utils.model_downloader import ensure_model

        with patch('hllm.utils.model_downloader.os.path.isdir') as mock_isdir, \
             patch('hllm.utils.model_downloader.download_model') as mock_download:
            mock_isdir.return_value = False
            mock_download.return_value = "/cached/model/path"
            
            result = ensure_model("hf://model/id")
            
            assert result == "/cached/model/path"
            mock_download.assert_called_once_with(
                "hf://model/id",
                cache_dir=None
            )


class TestRegisterModelMapping:
    """测试 register_model_mapping 函数"""

    def test_register_new_mapping(self):
        """测试注册新映射"""
        from hllm.utils.model_downloader import register_model_mapping, get_modelscope_id

        register_model_mapping("custom/model", "custom/ms/model")
        
        result = get_modelscope_id("custom/model")
        assert result == "custom/ms/model"


class TestDownloadModelLocal:
    """测试 download_model 本地路径处理"""

    def test_local_path_returns_early(self):
        """测试本地路径提前返回"""
        from hllm.utils.model_downloader import download_model

        with patch('hllm.utils.model_downloader.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = download_model("/local/model")
            
            assert result == "/local/model"


class TestDownloadFromHF:
    """测试 download_from_hf 函数"""

    def test_with_mirror(self):
        """测试使用镜像"""
        from hllm.utils.model_downloader import download_from_hf

        with patch('os.environ', {}), \
             patch('huggingface_hub.snapshot_download') as mock_snapshot:
            mock_snapshot.return_value = "/path/to/model"
            
            result = download_from_hf("test/model", use_mirror=True)
            
            assert result == "/path/to/model"
            mock_snapshot.assert_called_once()
            call_kwargs = mock_snapshot.call_args[1]
            assert call_kwargs["repo_id"] == "test/model"

    def test_without_mirror(self):
        """测试不使用镜像"""
        from hllm.utils.model_downloader import download_from_hf

        with patch('os.environ', {}), \
             patch('huggingface_hub.snapshot_download') as mock_snapshot:
            mock_snapshot.return_value = "/path/to/model"
            
            result = download_from_hf("test/model", use_mirror=False)
            
            assert result == "/path/to/model"
            mock_snapshot.assert_called_once()

    def test_missing_huggingface_hub(self):
        """测试缺少 huggingface_hub 抛出 ImportError"""
        from hllm.utils.model_downloader import download_from_hf

        with patch.dict('sys.modules', {'huggingface_hub': None}):
            with pytest.raises(ImportError, match="huggingface-hub"):
                download_from_hf("test/model", use_mirror=False)


class TestDownloadModelFallback:
    """测试 download_model 降级逻辑"""

    def test_all_sources_fail_raises_runtime_error(self):
        """测试所有源都失败时抛出 RuntimeError"""
        from hllm.utils.model_downloader import download_model

        with patch('hllm.utils.model_downloader.os.path.exists') as mock_exists, \
             patch('hllm.utils.model_downloader.download_from_modelscope') as mock_ms, \
             patch('hllm.utils.model_downloader.download_from_hf') as mock_hf:
            mock_exists.return_value = False
            mock_ms.side_effect = Exception("ModelScope error")
            mock_hf.side_effect = Exception("HF error")
            
            with pytest.raises(RuntimeError, match="无法下载模型"):
                download_model("failing/model")

    def test_modelscope_fallback(self):
        """测试 ModelScope 降级"""
        from hllm.utils.model_downloader import download_model

        with patch('hllm.utils.model_downloader.os.path.exists') as mock_exists, \
             patch('hllm.utils.model_downloader.download_from_modelscope') as mock_ms, \
             patch('hllm.utils.model_downloader.download_from_hf') as mock_hf:
            mock_exists.return_value = False
            mock_ms.side_effect = Exception("ModelScope error")
            mock_hf.return_value = "/fallback/path"
            
            result = download_model("test/model", prefer_modelscope=False)
            
            assert result == "/fallback/path"
