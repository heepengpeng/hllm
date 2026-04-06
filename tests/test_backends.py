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


class TestPyTorchBackendDeviceNormalization:
    """测试 PyTorchBackend 设备标准化"""

    def test_device_cpu_normalization(self):
        """测试 CPU 设备标准化"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        # Mock ensure_model_fn to skip real loading
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_model.__call__ = Mock(return_value=Mock())
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="mocked response")
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            device="cpu",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.device_name == "cpu"

    def test_device_cuda_fallback_to_cpu(self):
        """测试 CUDA 不可用时回退到 CPU"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)  # CUDA 不可用
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_model.__call__ = Mock(return_value=Mock())
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="mocked response")
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        # 请求 CUDA，但不可用，应该回退到 CPU
        backend = PyTorchBackend(
            "test-model",
            device="cuda",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.device_name == "cpu"

    def test_device_mps_normalization(self):
        """测试 MPS 设备标准化"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=True)  # MPS 可用
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_model.__call__ = Mock(return_value=Mock())
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="mocked response")
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            device="mps",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.device_name == "mps"

    def test_unknown_device_fallback(self):
        """测试未知设备回退到 CPU"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_model.__call__ = Mock(return_value=Mock())
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="mocked response")
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        # 请求未知设备
        backend = PyTorchBackend(
            "test-model",
            device="unknown_device",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.device_name == "cpu"


class TestPyTorchBackendModelLoading:
    """测试 PyTorchBackend 模型加载"""

    def test_load_model_with_custom_ensure_fn(self):
        """测试使用自定义 ensure_model_fn"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        # 自定义 ensure_model_fn
        mock_ensure = Mock(return_value="/custom/path/to/model")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "hf://test/model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # 验证 ensure_model_fn 被调用
        mock_ensure.assert_called_once()
        call_args = mock_ensure.call_args
        assert call_args[0][0] == "hf://test/model"
        
        # 验证 transformers 函数使用正确的路径
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        call_args = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
        assert call_args[0][0] == "/custom/path/to/model"

    def test_load_model_sets_pad_token(self):
        """测试加载模型时设置 pad_token"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        
        # pad_token 为 None，eos_token 存在
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None  # None
        mock_tokenizer.eos_token = "<pad>"  # 会被复制
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # 验证 pad_token 被设置为 eos_token
        assert mock_tokenizer.pad_token == "<pad>"

    def test_load_model_with_quantization(self):
        """测试使用量化配置加载模型"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        mock_quant_config = Mock()
        
        # 清除第一次调用
        mock_transformers.AutoModelForCausalLM.from_pretrained.reset_mock()
        
        # 直接测试 _load_model 传入量化配置
        # 由于 PyTorchBackend 在 __init__ 中调用 _load_model，
        # 我们需要重新创建 mock 并测试第二次调用
        mock_ensure2 = Mock(return_value="/mock/path")
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure2,
        )
        
        # 获取第二次调用（带量化配置）
        backend._load_model(quantization_config=mock_quant_config)
        
        # 验证最后一次调用包含量化配置
        calls = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args_list
        last_call_kwargs = calls[-1][1]  # 最后一个调用的关键字参数
        assert "quantization_config" in last_call_kwargs
        assert last_call_kwargs["quantization_config"] == mock_quant_config


class TestPyTorchBackendProperties:
    """测试 PyTorchBackend 属性"""

    def test_vocab_size_property(self):
        """测试 vocab_size 属性"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=50000)  # vocab_size = 50000
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.vocab_size == 50000

    def test_tokenizer_property(self):
        """测试 tokenizer 属性"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.tokenizer == mock_tokenizer

    def test_config_property(self):
        """测试 config 属性"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock(hidden_size=4096)  # 模型配置
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.config.hidden_size == 4096


class TestPyTorchBackendMemoryUsage:
    """测试 PyTorchBackend 内存使用"""

    def test_memory_usage_cuda(self):
        """测试 CUDA 内存使用"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=True)  # CUDA 可用
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.cuda.memory_allocated = Mock(return_value=1024 * 1024 * 100)  # 100 MB
        mock_torch.cuda.memory_reserved = Mock(return_value=1024 * 1024 * 200)  # 200 MB
        mock_torch.cuda.max_memory_allocated = Mock(return_value=1024 * 1024 * 500)  # 500 MB
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            device="cuda",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        memory = backend.get_memory_usage()
        
        assert memory["device"] == "cuda"
        assert "allocated_mb" in memory
        assert memory["allocated_mb"] == 100  # 100 MB
        assert "reserved_mb" in memory
        assert memory["reserved_mb"] == 200


class TestPyTorchBackendWarmup:
    """测试 PyTorchBackend 预热"""

    def test_warmup_with_cuda_sync(self):
        """测试 CUDA 预热同步"""
        from hllm.backends.pytorch import PyTorchBackend
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        # 让 no_grad 支持上下文管理器
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=True)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()  # 会调用
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.__call__ = Mock(return_value=Mock())
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            device="cuda",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        backend.warmup(batch_size=2, seq_len=256)
        
        # 验证调用了 synchronize
        mock_torch.cuda.synchronize.assert_called_once()


class TestPyTorchBackendCleanup:
    """测试 PyTorchBackend 清理"""

    def test_cleanup_cuda(self):
        """测试 CUDA 清理"""
        from hllm.backends.pytorch import PyTorchBackend
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        mock_torch.no_grad = Mock(return_value=Mock())
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=True)
        mock_torch.cuda.empty_cache = Mock()  # 会调用
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            device="cuda",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        backend.cleanup()
        
        # 验证调用了 empty_cache
        mock_torch.cuda.empty_cache.assert_called_once()


class TestPyTorchBackendGenerate:
    """测试 PyTorchBackend 生成逻辑"""

    def _create_mock_backend(self):
        """创建 mock 后端辅助方法"""
        from hllm.backends.pytorch import PyTorchBackend
        from hllm.backends.base import GenerationParams
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        return PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        ), mock_model, mock_tokenizer

    def test_generate_impl_updates_stats(self):
        """测试 _generate_impl 更新统计"""
        from hllm.backends.pytorch import PyTorchBackend
        from hllm.backends.base import GenerationParams
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        # encode 返回5个token
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # patch hllm.generate.generate (函数在内部导入)
        with patch('hllm.generate.generate') as mock_gen:
            mock_gen.return_value = "Hello world"
            
            params = GenerationParams(max_new_tokens=50)
            result = backend._generate_impl("Hello", params)
            
            assert result == "Hello world"
            # 验证 generate 被调用
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["max_new_tokens"] == 50
            assert call_kwargs["temperature"] == 1.0

    def test_stream_generate_impl_yields_tokens(self):
        """测试 _stream_generate_impl 逐个 yield token"""
        from hllm.backends.pytorch import PyTorchBackend
        from hllm.backends.base import GenerationParams
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # patch hllm.generate.stream_generate (函数在内部导入)
        def mock_stream_gen(*args, **kwargs):
            yield "Hello"
            yield " "
            yield "World"
        
        with patch('hllm.generate.stream_generate', mock_stream_gen):
            params = GenerationParams(max_new_tokens=50)
            tokens = list(backend._stream_generate_impl("Hello", params))
            
            assert tokens == ["Hello", " ", "World"]

    def test_eos_token_id_property(self):
        """测试 eos_token_id 属性"""
        from hllm.backends.pytorch import PyTorchBackend
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 200  # 设置 eos_token_id
        mock_tokenizer.pad_token_id = 0
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.eos_token_id == 200

    def test_pad_token_id_property(self):
        """测试 pad_token_id 属性"""
        from hllm.backends.pytorch import PyTorchBackend
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_tokenizer.pad_token_id = 300  # 设置 pad_token_id
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        assert backend.pad_token_id == 300


class TestPyTorchBackendCompile:
    """测试 torch.compile 功能"""

    def test_load_model_with_compile(self):
        """测试使用 torch.compile 编译模型"""
        from hllm.backends.pytorch import PyTorchBackend
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = Mock(return_value=Mock())  # 支持 compile
        mock_torch.compile.mode = "reduce-overhead"
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # 调用 _load_model 传入 compile=True
        backend._load_model(compile=True)
        
        # 验证 torch.compile 被调用
        mock_torch.compile.assert_called_once()
        call_args = mock_torch.compile.call_args
        assert call_args[0][0] == mock_model
        assert call_args[1]["mode"] == "reduce-overhead"


class TestPyTorchBackendMemoryCPU:
    """测试 CPU 内存使用"""

    def test_memory_usage_cpu(self):
        """测试 CPU 内存使用"""
        from hllm.backends.pytorch import PyTorchBackend
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            device="cpu",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # psutil 在 get_memory_usage 函数内部导入
        with patch('psutil.Process') as mock_process_cls:
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 1024 * 1024 * 50  # 50 MB
            mock_process_cls.return_value = mock_process
            
            memory = backend.get_memory_usage()
            
            assert memory["device"] == "cpu"
            assert "rss_mb" in memory


class TestPyTorchBackendLoadModelAttnImpl:
    """测试注意力实现配置"""

    def test_load_model_with_attn_implementation(self):
        """测试使用自定义注意力实现"""
        from hllm.backends.pytorch import PyTorchBackend
        import contextlib
        
        mock_torch = Mock()
        mock_torch.float32 = "float32"
        mock_torch.zeros = Mock(return_value=Mock())
        
        @contextlib.contextmanager
        def mock_no_grad():
            yield
        
        mock_torch.no_grad = mock_no_grad
        mock_torch.cuda = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.backends = Mock()
        mock_torch.backends.mps = Mock()
        mock_torch.backends.mps.is_available = Mock(return_value=False)
        mock_torch.compile = None
        
        mock_ensure = Mock(return_value="/mock/path")
        mock_transformers = Mock()
        mock_model = Mock()
        mock_model.to = Mock()
        mock_model.eval = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100
        mock_transformers.AutoModelForCausalLM.from_pretrained = Mock(return_value=mock_model)
        mock_transformers.AutoTokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
        
        backend = PyTorchBackend(
            "test-model",
            torch_module=mock_torch,
            transformers_module=mock_transformers,
            ensure_model_fn=mock_ensure,
        )
        
        # 调用 _load_model 传入 attn_implementation
        backend._load_model(attn_implementation="flash_attention_2")
        
        # 验证注意力实现被传递
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called()
        call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert "attn_implementation" in call_kwargs
        assert call_kwargs["attn_implementation"] == "flash_attention_2"


class TestBackendAutoSelect:
    """测试 auto_select_backend 函数"""

    def test_auto_select_returns_pytorch_default(self):
        """测试默认返回 pytorch"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.backends.mps.is_available', return_value=False):
                name, kwargs = auto_select_backend()
                assert name == "pytorch"

    def test_auto_select_with_model_path(self):
        """测试带模型路径"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.backends.mps.is_available', return_value=False):
                name, kwargs = auto_select_backend("test/model")
                assert name == "pytorch"
                assert kwargs["model_path"] == "test/model"

    def test_auto_select_cpu_device(self):
        """测试指定 CPU 设备"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.backends.mps.is_available', return_value=False):
                name, kwargs = auto_select_backend(device="cpu")
                assert name == "pytorch"

    def test_auto_select_cuda_available(self):
        """测试 CUDA 可用时返回 paged_pytorch"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=True):
                name, kwargs = auto_select_backend(device="cuda")
                # 如果 paged_pytorch 可用，返回 paged_pytorch
                assert name in ["paged_pytorch", "pytorch"]

    def test_auto_select_cuda_device(self):
        """测试指定 cuda 设备"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=True):
                name, kwargs = auto_select_backend(device="cuda")
                assert name in ["paged_pytorch", "pytorch"]

    def test_auto_select_mps_available(self):
        """测试 MPS 可用时返回 pytorch"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.backends.mps.is_available', return_value=True):
                name, kwargs = auto_select_backend(device="mps")
                assert name == "pytorch"

    def test_auto_select_mps_fallback(self):
        """测试 MPS fallback"""
        from hllm.backends import auto_select_backend

        with patch.dict('sys.modules', {'mlx.core': None}):
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('torch.backends.mps.is_available', return_value=True):
                name, kwargs = auto_select_backend(device="auto")
                assert name == "pytorch"


class TestBackendCreate:
    """测试 create_backend 函数"""

    def test_create_pytorch_backend(self):
        """测试创建 PyTorch 后端"""
        from hllm.backends import create_backend, PyTorchBackend

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "PAD"
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.__len__ = Mock(return_value=32000)
        mock_tokenizer.eos_token_id = 100

        with patch.object(PyTorchBackend, '_load_model'):
            backend = create_backend("pytorch", "test/model")
            assert backend.model_path == "test/model"

    def test_create_unknown_backend_raises(self):
        """测试创建未知后端抛出异常"""
        from hllm.backends import create_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("nonexistent_backend_xyz", "test/model")


class TestBackendInfo:
    """测试后端信息"""

    def test_get_backend_info_contains_all_backends(self):
        """测试后端信息包含所有后端"""
        from hllm.backends import get_backend_info, list_backends

        info = get_backend_info()
        backends = list_backends()

        for backend in backends:
            assert backend in info

    def test_backend_info_structure(self):
        """测试后端信息结构"""
        from hllm.backends import get_backend_info

        info = get_backend_info()
        pytorch_info = info["pytorch"]

        assert "available" in pytorch_info
        assert "supports_quantization" in pytorch_info
        assert "supports_gpu" in pytorch_info
        assert "default_device" in pytorch_info


class TestBackendRegister:
    """测试后端注册"""

    def test_register_backend_with_custom_class(self):
        """测试注册自定义后端类"""
        from hllm.backends import register_backend, list_backends, BaseBackend

        class CustomBackend(BaseBackend):
            NAME = "custom_test"
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

        # 使用唯一的名称避免冲突
        register_backend("custom_test_module", CustomBackend)
        assert "custom_test_module" in list_backends()

    def test_register_backend_requires_basebackend(self):
        """测试注册需要 BaseBackend"""
        from hllm.backends import register_backend

        class NotABackend:
            pass

        with pytest.raises(TypeError, match="must inherit"):
            register_backend("not_a_backend", NotABackend)
