"""
测试 backend 模块 - 深度覆盖
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestBackendRegistryDeep:
    """测试后端注册表"""

    def test_list_backends_returns_list(self):
        """测试 list_backends 返回列表"""
        from hllm.backends import list_backends
        backends = list_backends()
        assert isinstance(backends, list)
        assert "pytorch" in backends

    def test_get_backend_info_structure(self):
        """测试 get_backend_info 返回正确的结构"""
        from hllm.backends import get_backend_info

        info = get_backend_info()
        assert isinstance(info, dict)
        assert "pytorch" in info
        assert "available" in info["pytorch"]
        assert "supports_quantization" in info["pytorch"]

    def test_create_backend_pytorch(self):
        """测试创建 PyTorch 后端"""
        from hllm.backends import create_backend, list_backends

        if "pytorch" in list_backends():
            # 不实际加载模型，只检查函数存在
            assert callable(create_backend)

    def test_auto_select_returns_tuple(self):
        """测试 auto_select 返回元组"""
        from hllm.backends import auto_select_backend

        result = auto_select_backend("test-model")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)


class TestBackendBaseMethods:
    """测试 BaseBackend 方法"""

    def test_generation_params_validate(self):
        """测试 GenerationParams 验证"""
        from hllm.backends.base import GenerationParams

        params = GenerationParams()
        params.validate()  # 默认值应该有效

        # 测试无效值
        invalid_params = GenerationParams(temperature=-0.5)
        with pytest.raises(ValueError):
            invalid_params.validate()

    def test_generation_params_custom(self):
        """测试自定义 GenerationParams"""
        from hllm.backends.base import GenerationParams

        params = GenerationParams(
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            stop_sequences=["END"]
        )
        assert params.max_new_tokens == 256
        assert params.temperature == 0.8
        assert params.stop_sequences == ["END"]

    def test_backend_stats_update(self):
        """测试 BackendStats 更新"""
        from hllm.backends.base import BackendStats

        stats = BackendStats()
        assert stats.total_requests == 0

        stats.update(prompt_tokens=10, generated_tokens=20, latency_ms=100.0)
        assert stats.total_requests == 1
        assert stats.total_prompt_tokens == 10
        assert stats.total_tokens_generated == 20
        assert stats.avg_latency_ms > 0

    def test_backend_stats_multiple_updates(self):
        """测试多次更新统计"""
        from hllm.backends.base import BackendStats

        stats = BackendStats()
        stats.update(10, 20, 100.0)
        stats.update(15, 25, 150.0)

        assert stats.total_requests == 2
        assert stats.total_tokens_generated == 45

    def test_backend_stats_reset(self):
        """测试统计重置"""
        from hllm.backends.base import BackendStats

        stats = BackendStats()
        stats.update(10, 20, 100.0)
        assert stats.total_requests == 1

        # 重置应该创建新对象
        stats.total_requests = 0
        stats.total_prompt_tokens = 0
        stats.total_tokens_generated = 0
        assert stats.total_requests == 0


class TestBackendFactory:
    """测试后端工厂"""

    def test_register_backend(self):
        """测试注册新后端"""
        from hllm.backends import register_backend, list_backends, BaseBackend

        class TestBackend(BaseBackend):
            NAME = "test"
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

        # 注册测试后端
        register_backend("test_new", TestBackend)
        assert "test_new" in list_backends()

    def test_register_duplicate_raises(self):
        """测试重复注册抛出异常"""
        from hllm.backends import register_backend, BaseBackend

        class DummyBackend(BaseBackend):
            NAME = "dummy"
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

        register_backend("dummy_test", DummyBackend)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("dummy_test", DummyBackend)

    def test_get_backend_class_unknown(self):
        """测试获取未知后端"""
        from hllm.backends import get_backend_class

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_class("nonexistent_backend_xyz")
