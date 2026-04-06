"""
测试 server 模块 - Mock 覆盖
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestServerConfig:
    """测试服务器配置"""

    def test_server_config_class_exists(self):
        """测试 ServerConfig 类存在"""
        from hllm.config import ServerConfig
        assert callable(ServerConfig)

    def test_server_config_defaults(self):
        """测试默认配置"""
        from hllm.config import ServerConfig

        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000

    def test_server_config_custom(self):
        """测试自定义配置"""
        from hllm.config import ServerConfig

        config = ServerConfig(host="localhost", port=9000)
        assert config.host == "localhost"
        assert config.port == 9000


class TestServerModule:
    """测试 server 模块"""

    def test_server_module_import(self):
        """测试 server 模块可导入"""
        try:
            import hllm.server
            assert True
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_server_has_main(self):
        """测试有 main 函数"""
        try:
            from hllm.server import main
            assert callable(main)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_server_has_app(self):
        """测试有 app 对象"""
        try:
            from hllm.server import app
            assert app is not None
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerImports:
    """测试服务器导入"""

    def test_server_imports_fastapi(self):
        """测试 FastAPI 导入"""
        try:
            from hllm.server import FastAPI
            assert callable(FastAPI)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_server_has_uvicorn(self):
        """测试有 uvicorn 导入"""
        try:
            from hllm.server import uvicorn
            assert uvicorn is not None
        except ImportError:
            pytest.skip("uvicorn not installed")


class TestServerRoutes:
    """测试服务器路由"""

    def test_server_routes_exist(self):
        """测试路由存在"""
        try:
            from hllm.server import app
            assert hasattr(app, 'routes') or hasattr(app, 'add_api_route')
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerMiddleware:
    """测试服务器中间件"""

    def test_server_has_middleware(self):
        """测试有中间件支持"""
        try:
            from hllm.server import app
            assert hasattr(app, 'middleware') or hasattr(app, 'add_middleware')
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerPydanticModels:
    """测试 server Pydantic 模型"""

    def test_chat_message_model(self):
        """测试 ChatMessage 模型"""
        try:
            from hllm.server import ChatMessage
            msg = ChatMessage(role="user", content="Hello")
            assert msg.role == "user"
            assert msg.content == "Hello"
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_message_validation(self):
        """测试 ChatMessage 角色验证"""
        try:
            from hllm.server import ChatMessage
            with pytest.raises(ValueError):
                ChatMessage(role="invalid_role", content="test")
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_defaults(self):
        """测试 ChatCompletionRequest 默认值"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            req = ChatCompletionRequest(messages=[msg])
            assert req.model == "hllm-model"
            assert req.max_tokens == 100
            assert req.temperature == 0.7
            assert req.top_p == 0.9
            assert req.top_k == 50
            assert req.stream == False
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_custom(self):
        """测试 ChatCompletionRequest 自定义值"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            req = ChatCompletionRequest(
                model="custom-model",
                messages=[msg],
                max_tokens=200,
                temperature=1.0,
                top_p=0.95,
                top_k=100,
                stream=True
            )
            assert req.model == "custom-model"
            assert req.max_tokens == 200
            assert req.temperature == 1.0
            assert req.stream == True
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_validation(self):
        """测试 ChatCompletionRequest 参数验证"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            # max_tokens 超出范围
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], max_tokens=5000)
            # temperature 超出范围
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], temperature=3.0)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_completion_request_model(self):
        """测试 CompletionRequest 模型"""
        try:
            from hllm.server import CompletionRequest
            req = CompletionRequest(prompt="Hello world")
            assert req.model == "hllm-model"
            assert req.max_tokens == 100
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_completion_request_string_prompt(self):
        """测试字符串 prompt"""
        try:
            from hllm.server import CompletionRequest
            req = CompletionRequest(prompt="Test prompt")
            assert req.prompt == "Test prompt"
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_completion_request_list_prompt(self):
        """测试列表 prompt"""
        try:
            from hllm.server import CompletionRequest
            req = CompletionRequest(prompt=["Line 1", "Line 2"])
            assert req.prompt == ["Line 1", "Line 2"]
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_response_model(self):
        """测试 ChatCompletionResponse 模型"""
        try:
            from hllm.server import (
                ChatMessage,
                ChatCompletionResponseChoice,
                ChatCompletionResponse
            )
            choice = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hi")
            )
            resp = ChatCompletionResponse(
                id="test-123",
                created=1234567890,
                model="test-model",
                choices=[choice],
                usage={"prompt_tokens": 5, "completion_tokens": 2}
            )
            assert resp.id == "test-123"
            assert resp.object == "chat.completion"
            assert len(resp.choices) == 1
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_stream_response_model(self):
        """测试流式响应模型"""
        try:
            from hllm.server import ChatCompletionStreamChoice, ChatCompletionStreamResponse
            choice = ChatCompletionStreamChoice(index=0, delta={"content": "Hi"})
            resp = ChatCompletionStreamResponse(
                id="test-456",
                created=1234567890,
                model="test-model",
                choices=[choice]
            )
            assert resp.id == "test-456"
            assert resp.object == "chat.completion.chunk"
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerGetModel:
    """测试 get_model 函数"""

    def test_get_model_raises_when_not_initialized(self):
        """测试模型未初始化时抛出异常"""
        try:
            from hllm.server import get_model
            with pytest.raises(RuntimeError, match="not initialized"):
                get_model()
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerLogging:
    """测试服务器日志"""

    def test_server_logger_exists(self):
        """测试 logger 存在"""
        try:
            from hllm.server import logger
            assert logger is not None
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerPydanticValidation:
    """测试 Pydantic 模型验证"""

    def test_chat_message_content_required(self):
        """测试 content 是必需的"""
        try:
            from hllm.server import ChatMessage
            with pytest.raises(ValueError):
                ChatMessage(role="user")  # 缺少 content
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_max_tokens_range(self):
        """测试 max_tokens 范围验证"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            # 太小的值
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], max_tokens=0)
            # 太大的值
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], max_tokens=5000)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_temperature_range(self):
        """测试 temperature 范围验证"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            # 太小
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], temperature=-0.1)
            # 太大
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], temperature=2.5)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_top_p_range(self):
        """测试 top_p 范围验证"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            # 太小
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], top_p=-0.1)
            # 太大
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], top_p=1.5)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_top_k_non_negative(self):
        """测试 top_k 必须非负"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            with pytest.raises(ValueError):
                ChatCompletionRequest(messages=[msg], top_k=-1)
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_stop_string(self):
        """测试 stop 可以是字符串"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            req = ChatCompletionRequest(messages=[msg], stop="END")
            assert req.stop == "END"
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_request_stop_list(self):
        """测试 stop 可以是列表"""
        try:
            from hllm.server import ChatMessage, ChatCompletionRequest
            msg = ChatMessage(role="user", content="Hello")
            req = ChatCompletionRequest(messages=[msg], stop=["END", "STOP"])
            assert req.stop == ["END", "STOP"]
        except ImportError:
            pytest.skip("FastAPI not installed")


class TestServerResponseModels:
    """测试响应模型"""

    def test_chat_completion_response_defaults(self):
        """测试 ChatCompletionResponse 默认值"""
        try:
            from hllm.server import ChatCompletionResponse
            resp = ChatCompletionResponse(
                id="test-123",
                created=1234567890,
                model="test-model",
                choices=[],
                usage={}
            )
            assert resp.object == "chat.completion"
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_stream_response_defaults(self):
        """测试流式响应默认值"""
        try:
            from hllm.server import ChatCompletionStreamResponse
            resp = ChatCompletionStreamResponse(
                id="test-456",
                created=1234567890,
                model="test-model",
                choices=[]
            )
            assert resp.object == "chat.completion.chunk"
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_chat_completion_stream_choice_defaults(self):
        """测试流式选择默认值"""
        try:
            from hllm.server import ChatCompletionStreamChoice
            choice = ChatCompletionStreamChoice(index=0, delta={})
            assert choice.finish_reason is None
        except ImportError:
            pytest.skip("FastAPI not installed")
