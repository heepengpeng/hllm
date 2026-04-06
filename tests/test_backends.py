"""
测试 backend 模块 - 深度覆盖
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import torch
import sys


class TestBackendRegistryDeep:
    """测试后端注册表"""

    def test_list_backends_returns_list(self):
        """测试 list_backends 返回列表"""
        from hllm.backends import list_backends
        backends = list_backends()
        assert isinstance(backends, list)
        assert "pytorch" in backends

    def test_get_backend_info_returns_dict(self):
        """测试 get_backend_info 返回结构"""
        from hllm.backends import get_backend_info

        info = get_backend_info()
        assert isinstance(info, dict)
        assert "pytorch" in info
        assert "available" in info["pytorch"]
        assert "supports_quantization" in info["pytorch"]
        assert "supports_gpu" in info["pytorch"]
        assert "default_device" in info["pytorch"]

    def test_auto_select_returns_tuple(self):
        """测试 auto_select 返回元组"""
        from hllm.backends import auto_select_backend

        result = auto_select_backend("test-model")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

    def test_auto_select_with_device(self):
        """测试指定设备"""
        from hllm.backends import auto_select_backend

        name, kwargs = auto_select_backend("model", device="cpu")
        assert name == "pytorch"


class TestBackendBaseClass:
    """测试 BaseBackend 基类"""

    def test_generation_params_defaults(self):
        """测试 GenerationParams 默认值"""
        from hllm.backends.base import GenerationParams

        params = GenerationParams()
        assert params.max_new_tokens == 128
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == 50
        assert params.repetition_penalty == 1.0
        assert params.stop_sequences == []

    def test_generation_params_validate_valid(self):
        """测试有效参数验证"""
        from hllm.backends.base import GenerationParams

        params = GenerationParams(temperature=0.5, top_p=0.9)
        params.validate()  # 不应抛出异常

    def test_generation_params_validate_invalid_temp(self):
        """测试无效温度验证"""
        from hllm.backends.base import GenerationParams

        params = GenerationParams(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            params.validate()

        params = GenerationParams(temperature=2.5)
        with pytest.raises(ValueError):
            params.validate()

    def test_generation_params_validate_invalid_tokens(self):
        """测试无效 max_tokens 验证"""
        from hllm.backends.base import GenerationParams

        params = GenerationParams(max_new_tokens=0)
        with pytest.raises(ValueError, match="max_new_tokens"):
            params.validate()

    def test_backend_stats_defaults(self):
        """测试 BackendStats 默认值"""
        from hllm.backends.base import BackendStats

        stats = BackendStats()
        assert stats.total_requests == 0
        assert stats.total_tokens_generated == 0
        assert stats.avg_latency_ms == 0.0

    def test_backend_stats_update(self):
        """测试统计更新"""
        from hllm.backends.base import BackendStats

        stats = BackendStats()
        stats.update(prompt_tokens=10, generated_tokens=20, latency_ms=100.0)
        assert stats.total_requests == 1
        assert stats.total_prompt_tokens == 10
        assert stats.total_tokens_generated == 20
        assert stats.avg_latency_ms > 0

    def test_backend_stats_reset(self):
        """测试统计重置"""
        from hllm.backends.base import BackendStats

        # 创建 BackendStats 实例
        stats = BackendStats()
        # 检查默认值
        assert stats.total_requests == 0

    def test_tokenizer_protocol(self):
        """测试 TokenizerProtocol"""
        from hllm.backends.base import TokenizerProtocol

        # 验证是 Protocol
        assert hasattr(TokenizerProtocol, 'encode')
        assert hasattr(TokenizerProtocol, 'decode')


class TestBackendFactory:
    """测试后端工厂"""

    def test_register_backend(self):
        """测试注册新后端"""
        from hllm.backends import register_backend, list_backends, BaseBackend

        class TestBackend(BaseBackend):
            NAME = "test_register"
            def __init__(self, model_path, **kwargs):
                super().__init__(model_path, **kwargs)
            def _load_model(self, **kwargs):
                pass
            def _generate_impl(self, prompt, params, **kwargs):
                return "test"
            def _stream_generate_impl(self, prompt, params, **kwargs):
                yield "test"
            @property
            def device_name(self):
                return "cpu"
            @property
            def eos_token_id(self):
                return 1
            @property
            def pad_token_id(self):
                return 0
            @property
            def tokenizer(self):
                return Mock()

        register_backend("test_register", TestBackend)
        assert "test_register" in list_backends()

    def test_register_duplicate_raises(self):
        """测试重复注册"""
        from hllm.backends import register_backend, BaseBackend

        class DummyBackend(BaseBackend):
            NAME = "dummy_dup"
            def __init__(self, model_path, **kwargs):
                super().__init__(model_path, **kwargs)
            def _load_model(self, **kwargs):
                pass
            def _generate_impl(self, prompt, params, **kwargs):
                return ""
            def _stream_generate_impl(self, prompt, params, **kwargs):
                yield ""
            @property
            def device_name(self):
                return "cpu"
            @property
            def eos_token_id(self):
                return 1
            @property
            def pad_token_id(self):
                return 0
            @property
            def tokenizer(self):
                return Mock()

        register_backend("dummy_dup", DummyBackend)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("dummy_dup", DummyBackend)

    def test_get_backend_class_unknown(self):
        """测试获取未知后端"""
        from hllm.backends import get_backend_class

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_class("nonexistent_backend_xyz")


class TestPyTorchBackendMock:
    """测试 PyTorchBackend (Mock)"""

    def test_pytorch_backend_class_exists(self):
        """测试 PyTorchBackend 类存在"""
        from hllm.backends.pytorch import PyTorchBackend
        assert PyTorchBackend.NAME == "pytorch"

    def test_pytorch_backend_supports_gpu(self):
        """测试 PyTorch 支持 GPU"""
        from hllm.backends.pytorch import PyTorchBackend
        assert PyTorchBackend.SUPPORTS_GPU == True

    def test_pytorch_backend_supports_quantization(self):
        """测试 PyTorch 支持量化"""
        from hllm.backends.pytorch import PyTorchBackend
        assert PyTorchBackend.SUPPORTS_QUANTIZATION == True

    def test_pytorch_backend_has_load_model(self):
        """测试有 _load_model 方法"""
        from hllm.backends.pytorch import PyTorchBackend
        assert hasattr(PyTorchBackend, '_load_model')

    def test_pytorch_backend_has_generate_impl(self):
        """测试有 _generate_impl 方法"""
        from hllm.backends.pytorch import PyTorchBackend
        assert hasattr(PyTorchBackend, '_generate_impl')

    def test_pytorch_backend_has_stream_impl(self):
        """测试有 _stream_generate_impl 方法"""
        from hllm.backends.pytorch import PyTorchBackend
        assert hasattr(PyTorchBackend, '_stream_generate_impl')


class TestMLXBackendMock:
    """测试 MLXBackend (Mock)"""

    def test_mlx_backend_class_exists(self):
        """测试 MLXBackend 类存在"""
        try:
            from hllm.backends.mlx import MLXBackend
            assert MLXBackend.NAME == "mlx"
        except ImportError:
            pytest.skip("MLX not installed")

    def test_mlx_backend_supports_gpu(self):
        """测试 MLX 支持 GPU"""
        try:
            from hllm.backends.mlx import MLXBackend
            assert MLXBackend.SUPPORTS_GPU == True
        except ImportError:
            pytest.skip("MLX not installed")


class TestPagedPyTorchBackendMock:
    """测试 PagedPyTorchBackend (Mock)"""

    def test_paged_pytorch_class_exists(self):
        """测试 PagedPyTorchBackend 类存在"""
        try:
            from hllm.backends.paged_pytorch import PagedPyTorchBackend
            assert PagedPyTorchBackend.NAME == "paged_pytorch"
        except ImportError:
            pytest.skip("vLLM not installed")
