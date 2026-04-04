"""
HLLM REST API 服务端测试
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock

# 跳过测试如果 flask 未安装
try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


@unittest.skipUnless(HAS_FLASK, "Flask not installed")
class TestServer(unittest.TestCase):
    """测试 REST API 服务端"""

    def setUp(self):
        """设置测试环境"""
        # 创建 mock 模型
        self.mock_model = Mock()
        self.mock_model.model_path = "test-model"
        self.mock_model.device = "cpu"
        self.mock_model.config = Mock()
        self.mock_model.config.max_position_embeddings = 2048
        self.mock_model.config.vocab_size = 32000
        self.mock_model.tokenizer = Mock()
        self.mock_model.tokenizer.encode.return_value = MagicMock(shape=(1, 5))
        self.mock_model.tokenizer.decode.return_value = "Hello world"
        self.mock_model.tokenizer.apply_chat_template.return_value = "<user>Hello</user>"
        self.mock_model.generate.return_value = "Generated text"

        # 创建 mock 流式生成器
        def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token
        self.mock_model.stream_generate.return_value = mock_stream()

        # 导入并配置服务器
        from hllm.server import app, _error_response
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # 设置全局模型
        import hllm.server as server_module
        server_module._model = self.mock_model

    def tearDown(self):
        """清理测试环境"""
        import hllm.server as server_module
        server_module._model = None

    def test_health_endpoint(self):
        """测试健康检查端点"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "test-model")
        self.assertEqual(data["device"], "cpu")

    def test_models_endpoint(self):
        """测试模型信息端点"""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data["model"], "test-model")
        self.assertEqual(data["device"], "cpu")
        self.assertEqual(data["max_length"], 2048)
        self.assertEqual(data["vocab_size"], 32000)

    def test_generate_endpoint_success(self):
        """测试生成端点 - 成功场景"""
        payload = {
            "prompt": "Hello",
            "max_new_tokens": 50,
            "temperature": 0.7
        }
        response = self.client.post(
            "/generate",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["text"], "Generated text")
        self.assertIn("usage", data)

        # 验证模型被调用
        self.mock_model.generate.assert_called_once()
        call_kwargs = self.mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs["prompt"], "Hello")
        self.assertEqual(call_kwargs["max_new_tokens"], 50)
        self.assertEqual(call_kwargs["temperature"], 0.7)

    def test_generate_endpoint_missing_prompt(self):
        """测试生成端点 - 缺少 prompt"""
        payload = {"max_new_tokens": 50}
        response = self.client.post(
            "/generate",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["error"]["code"], "INVALID_REQUEST")

    def test_generate_endpoint_empty_body(self):
        """测试生成端点 - 空请求体"""
        response = self.client.post("/generate")
        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        self.assertEqual(data["error"]["code"], "INVALID_REQUEST")

    def test_chat_endpoint_success(self):
        """测试对话端点 - 成功场景"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_new_tokens": 100
        }
        response = self.client.post(
            "/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["message"]["role"], "assistant")
        self.assertEqual(data["message"]["content"], "Generated text")
        self.assertIn("usage", data)

    def test_chat_endpoint_missing_messages(self):
        """测试对话端点 - 缺少 messages"""
        payload = {"max_new_tokens": 100}
        response = self.client.post(
            "/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["error"]["code"], "INVALID_REQUEST")

    def test_chat_endpoint_invalid_messages(self):
        """测试对话端点 - 无效的 messages 格式"""
        payload = {"messages": "invalid"}
        response = self.client.post(
            "/chat",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["error"]["code"], "INVALID_REQUEST")

    def test_generate_stream_endpoint(self):
        """测试流式生成端点"""
        payload = {
            "prompt": "Hello",
            "max_new_tokens": 10
        }
        response = self.client.post(
            "/generate/stream",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")

        # 解析 SSE 数据
        data = response.data.decode("utf-8")
        self.assertIn("data:", data)
        self.assertIn("[DONE]", data)


@unittest.skipUnless(HAS_FLASK, "Flask not installed")
class TestServerModelNotInitialized(unittest.TestCase):
    """测试模型未初始化时的错误处理"""

    def setUp(self):
        from hllm.server import app
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # 确保模型为 None
        import hllm.server as server_module
        server_module._model = None

    def test_health_without_model(self):
        """测试未初始化模型时访问健康端点"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        
        data = json.loads(response.data)
        self.assertEqual(data["error"]["code"], "SERVICE_UNAVAILABLE")


class TestErrorResponse(unittest.TestCase):
    """测试错误响应函数"""

    @unittest.skipUnless(HAS_FLASK, "Flask not installed")
    def test_error_response_format(self):
        """测试错误响应格式"""
        from flask import Flask
        from hllm.server import _error_response

        # 创建应用上下文
        app = Flask(__name__)
        with app.app_context():
            response = _error_response("TEST_CODE", "Test message", 418)
            data = json.loads(response.data)

            self.assertEqual(response.status_code, 418)
            self.assertEqual(data["error"]["code"], "TEST_CODE")
            self.assertEqual(data["error"]["message"], "Test message")


if __name__ == "__main__":
    unittest.main()
