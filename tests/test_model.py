"""
测试 model 模块
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestHLLMClass:
    """测试 HLLM 类"""

    def test_hllm_class_exists(self):
        """测试 HLLM 类存在"""
        from hllm import HLLM
        assert callable(HLLM)

    def test_hllm_has_generate_method(self):
        """测试 HLLM 有 generate 方法"""
        from hllm.model import HLLM
        assert hasattr(HLLM, 'generate')

    def test_hllm_has_stream_generate_method(self):
        """测试 HLLM 有 stream_generate 方法"""
        from hllm.model import HLLM
        assert hasattr(HLLM, 'stream_generate')


class TestHLLMMethods:
    """测试 HLLM 方法签名"""

    def test_generate_method_signature(self):
        """测试 generate 方法签名"""
        from hllm.model import HLLM
        import inspect

        sig = inspect.signature(HLLM.generate)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "max_new_tokens" in params
        assert "temperature" in params

    def test_stream_generate_method_signature(self):
        """测试 stream_generate 方法签名"""
        from hllm.model import HLLM
        import inspect

        sig = inspect.signature(HLLM.stream_generate)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "max_new_tokens" in params


class TestHLLMAttributes:
    """测试 HLLM 属性"""

    def test_hllm_has_generate_method(self):
        """测试 HLLM 有 generate 方法"""
        from hllm.model import HLLM
        assert callable(HLLM.generate)

    def test_hllm_has_stream_generate_method(self):
        """测试 HLLM 有 stream_generate 方法"""
        from hllm.model import HLLM
        assert callable(HLLM.stream_generate)

    def test_hllm_has_get_info(self):
        """测试 HLLM 有 get_info 方法"""
        from hllm.model import HLLM
        assert hasattr(HLLM, 'get_info')

    def test_hllm_has_tokenizer_property(self):
        """测试 HLLM 有 tokenizer 属性"""
        from hllm.model import HLLM
        assert hasattr(HLLM, 'tokenizer')


class TestHLLMInit:
    """测试 HLLM 初始化参数"""

    def test_init_parameters(self):
        """测试初始化参数"""
        from hllm.model import HLLM
        import inspect

        sig = inspect.signature(HLLM.__init__)
        params = list(sig.parameters.keys())
        assert "model_path" in params
        assert "backend" in params
        assert "device" in params


class TestHLLMWithMockBackend:
    """测试 HLLM 与 Mock 后端"""

    def test_generate_delegates_to_backend(self):
        """测试 generate 委托给后端"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        # 创建 mock 后端
        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "Hello from mock"

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch")
            result = hllm.generate("Hello", max_new_tokens=50)

            assert result == "Hello from mock"
            mock_backend.generate.assert_called_once_with(
                prompt="Hello",
                max_new_tokens=50,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.0
            )

    def test_generate_with_custom_params(self):
        """测试带自定义参数的 generate"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "Custom response"

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch")
            result = hllm.generate(
                "Hello",
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2
            )

            call_kwargs = mock_backend.generate.call_args[1]
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["top_k"] == 40
            assert call_kwargs["repetition_penalty"] == 1.2

    def test_stream_generate_delegates_to_backend(self):
        """测试 stream_generate 委托给后端"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.stream_generate.return_value = iter(["Hello", " ", "World"])

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch")
            tokens = list(hllm.stream_generate("Hello"))

            assert tokens == ["Hello", " ", "World"]
            mock_backend.stream_generate.assert_called_once()

    def test_auto_backend_selection(self):
        """测试自动后端选择"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "Auto response"

        with patch('hllm.backends.auto_select_backend', return_value=("pytorch", {})), \
             patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="auto")
            result = hllm.generate("Hello")

            assert result == "Auto response"
            assert hllm.backend_name == "pytorch"

    def test_tokenizer_property(self):
        """测试 tokenizer 属性"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_tokenizer = Mock()
        mock_backend.tokenizer = mock_tokenizer

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch")
            assert hllm.tokenizer == mock_tokenizer

    def test_get_info_delegates_to_backend(self):
        """测试 get_info 委托给后端"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.get_info.return_value = {"name": "pytorch", "device": "cpu"}

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch")
            info = hllm.get_info()

            assert info["name"] == "pytorch"
            mock_backend.get_info.assert_called_once()

    def test_model_path_stored(self):
        """测试 model_path 被存储"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "test"

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("microsoft/Phi-3-mini", backend="pytorch")
            assert hllm.model_path == "microsoft/Phi-3-mini"

    def test_backend_name_stored(self):
        """测试 backend_name 被存储"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "test"

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="mlx")
            assert hllm.backend_name == "mlx"


class TestHLLMDeviceOption:
    """测试设备选项"""

    def test_pytorch_backend_with_device(self):
        """测试 PyTorch 后端指定设备"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "test"

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch", device="cuda")
            hllm.generate("Hello")

            call_kwargs = mock_backend.generate.call_args[1]
            assert "device" not in call_kwargs  # device 在后端初始化时传递


class TestHLLMTrustRemoteCode:
    """测试 trust_remote_code 选项"""

    def test_trust_remote_code_passed(self):
        """测试 trust_remote_code 被传递"""
        from hllm.model import HLLM
        from hllm.backends.base import BaseBackend

        mock_backend = Mock(spec=BaseBackend)
        mock_backend.generate.return_value = "test"

        with patch('hllm.backends.create_backend', return_value=mock_backend):
            hllm = HLLM("test/model", backend="pytorch", trust_remote_code=True)
            hllm.generate("Hello")

            # 验证 trust_remote_code 传递给后端
            create_call_kwargs = mock_backend.generate.call_args[1]
            # trust_remote_code 在初始化时传递，不是 generate 时
