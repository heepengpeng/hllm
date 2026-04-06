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
