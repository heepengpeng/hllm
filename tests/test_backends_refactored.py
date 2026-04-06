"""Backend 抽象层重构测试

测试新的 BaseBackend 抽象接口和各个后端实现。
"""

import pytest
from unittest.mock import Mock, patch

from hllm.backends.base import (
    BaseBackend,
    GenerationParams,
    BackendStats,
    TokenizerProtocol,
)
from hllm.backends import (
    create_backend,
    list_backends,
    get_backend_class,
    auto_select_backend,
    register_backend,
)


class TestGenerationParams:
    """GenerationParams 测试"""

    def test_default_values(self):
        """测试默认值"""
        params = GenerationParams()
        assert params.max_new_tokens == 128
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == 50
        assert params.repetition_penalty == 1.0
        assert params.stop_sequences == []

    def test_custom_values(self):
        """测试自定义值"""
        params = GenerationParams(
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            stop_sequences=["END", "STOP"]
        )
        assert params.max_new_tokens == 256
        assert params.temperature == 0.8
        assert params.stop_sequences == ["END", "STOP"]

    def test_validation_valid(self):
        """测试有效参数验证"""
        params = GenerationParams()
        params.validate()  # 不应抛出异常

    def test_validation_invalid_temperature(self):
        """测试无效 temperature"""
        params = GenerationParams(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature must be in"):
            params.validate()

        params = GenerationParams(temperature=2.1)
        with pytest.raises(ValueError, match="temperature must be in"):
            params.validate()

    def test_validation_invalid_max_tokens(self):
        """测试无效 max_new_tokens"""
        params = GenerationParams(max_new_tokens=0)
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            params.validate()


class TestBackendStats:
    """BackendStats 测试"""

    def test_default_values(self):
        """测试默认值"""
        stats = BackendStats()
        assert stats.total_requests == 0
        assert stats.total_tokens_generated == 0
        assert stats.avg_latency_ms == 0.0

    def test_update(self):
        """测试更新统计"""
        stats = BackendStats()
        stats.update(prompt_tokens=10, generated_tokens=20, latency_ms=100.0)

        assert stats.total_requests == 1
        assert stats.total_prompt_tokens == 10
        assert stats.total_tokens_generated == 20
        assert stats.avg_latency_ms > 0

    def test_multiple_updates(self):
        """测试多次更新"""
        stats = BackendStats()
        stats.update(10, 20, 100.0)
        stats.update(15, 25, 150.0)

        assert stats.total_requests == 2
        assert stats.total_tokens_generated == 45


class MockBackend(BaseBackend):
    """用于测试的 Mock 后端"""

    NAME = "mock"
    SUPPORTS_QUANTIZATION = True

    def __init__(self, model_path: str, **kwargs):
        self._device = kwargs.get("device", "cpu")
        self._tokenizer_mock = Mock()
        self._tokenizer_mock.eos_token_id = 1
        self._tokenizer_mock.pad_token_id = 0
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        """模拟加载模型"""
        pass

    def _generate_impl(self, prompt: str, params: GenerationParams, **kwargs) -> str:
        """模拟生成"""
        return f"Generated: {prompt}"

    def _stream_generate_impl(self, prompt: str, params: GenerationParams, **kwargs):
        """模拟流式生成"""
        yield "Hello"
        yield " World"

    @property
    def device_name(self) -> str:
        return self._device

    @property
    def eos_token_id(self):
        return 1

    @property
    def pad_token_id(self):
        return 0

    @property
    def tokenizer(self):
        return self._tokenizer_mock


class TestBaseBackend:
    """BaseBackend 抽象类测试"""

    def test_init(self):
        """测试初始化"""
        backend = MockBackend("test-model")
        assert backend.model_path == "test-model"
        assert backend.is_loaded
        assert backend.NAME == "mock"

    def test_generate(self):
        """测试生成"""
        backend = MockBackend("test-model")
        result = backend.generate("Hello", max_new_tokens=50)
        assert "Generated:" in result
        assert backend.stats.total_requests == 1

    def test_stream_generate(self):
        """测试流式生成"""
        backend = MockBackend("test-model")
        tokens = list(backend.stream_generate("Hello"))
        assert tokens == ["Hello", " World"]
        assert backend.stats.total_requests == 1

    def test_get_info(self):
        """测试获取信息"""
        backend = MockBackend("test-model", device="cuda")
        info = backend.get_info()
        assert info["name"] == "mock"
        assert info["device"] == "cuda"
        assert info["supports_quantization"] is True

    def test_context_manager(self):
        """测试上下文管理器"""
        with MockBackend("test-model") as backend:
            assert backend.is_loaded
        # 退出上下文后应该调用 cleanup
        assert not backend.is_loaded

    def test_reset_stats(self):
        """测试重置统计"""
        backend = MockBackend("test-model")
        backend.generate("Hello")
        assert backend.stats.total_requests == 1

        backend.reset_stats()
        assert backend.stats.total_requests == 0


class TestBackendRegistry:
    """后端注册机制测试"""

    def test_list_backends(self):
        """测试列出后端"""
        backends = list_backends()
        assert isinstance(backends, list)
        # 应该至少包含 pytorch 和 mlx（如果安装了依赖）

    def test_register_backend(self):
        """测试注册后端"""
        register_backend("mock_test", MockBackend)
        assert "mock_test" in list_backends()

    def test_register_duplicate(self):
        """测试重复注册"""
        register_backend("mock_dup", MockBackend)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("mock_dup", MockBackend)

    def test_get_backend_class(self):
        """测试获取后端类"""
        # 测试获取存在的后端
        if "pytorch" in list_backends():
            cls = get_backend_class("pytorch")
            assert cls is not None

        # 测试获取不存在的后端
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_class("nonexistent")

    def test_auto_select_backend(self):
        """测试自动选择后端"""
        backend_name, kwargs = auto_select_backend("test-model")
        assert isinstance(backend_name, str)
        assert backend_name in list_backends()
        assert kwargs.get("model_path") == "test-model"

        # 测试指定设备
        backend_name, kwargs = auto_select_backend("test-model", device="cpu")
        assert backend_name in list_backends()


@pytest.mark.skipif(
    "pytorch" not in list_backends(),
    reason="PyTorch not installed"
)
class TestPyTorchBackend:
    """PyTorchBackend 集成测试（可选）"""

    def test_backend_info(self):
        """测试后端信息"""
        cls = get_backend_class("pytorch")
        assert cls.NAME == "pytorch"
        assert cls.SUPPORTS_GPU is True


@pytest.mark.skipif(
    "mlx" not in list_backends(),
    reason="MLX not installed"
)
class TestMLXBackend:
    """MLXBackend 集成测试（可选）"""

    def test_backend_info(self):
        """测试后端信息"""
        cls = get_backend_class("mlx")
        assert cls.NAME == "mlx"
        assert cls.SUPPORTS_GPU is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
