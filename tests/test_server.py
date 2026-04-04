"""
HLLM REST API 服务端测试 (FastAPI + OpenAI Compatible)
"""

import json
import unittest
from unittest.mock import Mock, patch

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI, "FastAPI not installed")
class TestServer(unittest.TestCase):
    """测试 REST API 服务端"""

    def setUp(self):
        """设置测试环境"""
        # 创建 mock 模型
        self.mock_model = Mock()
        self.mock_model.model_path = "test-model"
        self.mock_model.device = "cpu"
        self.mock_model.backend_name = "pytorch"
        self.mock_model.config = Mock()
        self.mock_model.config.max_position_embeddings = 2048
        self.mock_model.config.vocab_size = 32000
        self.mock_model.tokenizer = Mock()
        # encode 应该返回一个列表，以便 len() 可以工作
        self.mock_model.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_model.tokenizer.decode.return_value = "Hello world"
        self.mock_model.tokenizer.apply_chat_template.return_value = "<user>Hello</user>"
        self.mock_model.generate.return_value = "Generated text"
        self.mock_model.get_info.return_value = {"name": "pytorch", "device": "cpu"}

        # 创建 mock 流式生成器
        def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token
        self.mock_model.stream_generate.return_value = mock_stream()

        # 导入 FastAPI 应用并设置模型
        from hllm.server import app, _model as model_ref
        self.app = app
        self.client = TestClient(app)

        # 设置全局模型
        import hllm.server as server_module
        server_module._model = self.mock_model

    def tearDown(self):
        """清理测试环境"""
        import hllm.server as server_module
        server_module._model = None

    def test_list_models(self):
        """测试模型列表端点"""
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["id"], "hllm-model")

    def test_chat_completions_success(self):
        """测试对话补全端点 - 成功场景"""
        payload = {
            "model": "hllm-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        response = self.client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "hllm-model")
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(data["choices"][0]["message"]["content"], "Generated text")
        self.assertIn("usage", data)

        # 验证模型被调用
        self.mock_model.generate.assert_called_once()

    def test_chat_completions_validation_error(self):
        """测试对话补全端点 - 参数验证错误"""
        payload = {
            "model": "hllm-model",
            "messages": "invalid"  # 应该是列表
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_chat_completions_missing_messages(self):
        """测试对话补全端点 - 缺少 messages"""
        payload = {
            "model": "hllm-model"
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_completions_success(self):
        """测试文本补全端点 - 成功场景"""
        payload = {
            "model": "hllm-model",
            "prompt": "Hello",
            "max_tokens": 50
        }
        response = self.client.post("/v1/completions", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["object"], "text_completion")
        self.assertEqual(data["model"], "hllm-model")
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["text"], "Generated text")

    def test_completions_with_list_prompt(self):
        """测试文本补全端点 - 列表格式的 prompt"""
        payload = {
            "model": "hllm-model",
            "prompt": ["Hello", "World"],
            "max_tokens": 50
        }
        response = self.client.post("/v1/completions", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_chat_completions_stream(self):
        """测试流式对话补全"""
        payload = {
            "model": "hllm-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        response = self.client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")

        # 解析 SSE 数据
        content = response.content.decode("utf-8")
        self.assertIn("data:", content)
        self.assertIn("[DONE]", content)

    def test_health_endpoint(self):
        """测试健康检查端点"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "test-model")


@unittest.skipUnless(HAS_FASTAPI, "FastAPI not installed")
class TestServerModelNotInitialized(unittest.TestCase):
    """测试模型未初始化时的错误处理"""

    def setUp(self):
        from hllm.server import app
        self.client = TestClient(app)

        # 确保模型为 None
        import hllm.server as server_module
        server_module._model = None

    def test_health_without_model(self):
        """测试未初始化模型时访问健康端点"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)

        data = response.json()
        self.assertIn("detail", data)

    def test_chat_completions_without_model(self):
        """测试未初始化模型时访问对话端点"""
        payload = {
            "model": "hllm-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 500)


@unittest.skipUnless(HAS_FASTAPI, "FastAPI not installed")
class TestOpenAICompatibility(unittest.TestCase):
    """测试 OpenAI 兼容性"""

    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.model_path = "test-model"
        self.mock_model.device = "cpu"
        self.mock_model.backend_name = "pytorch"
        self.mock_model.tokenizer = Mock()
        self.mock_model.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_model.tokenizer.decode.return_value = "Response"
        self.mock_model.tokenizer.apply_chat_template.return_value = "Formatted prompt"
        self.mock_model.generate.return_value = "Response text"
        self.mock_model.get_info.return_value = {"name": "pytorch", "device": "cpu"}

        from hllm.server import app
        self.client = TestClient(app)

        import hllm.server as server_module
        server_module._model = self.mock_model

    def tearDown(self):
        import hllm.server as server_module
        server_module._model = None

    def test_response_has_required_fields(self):
        """测试响应包含 OpenAI 所需字段"""
        payload = {
            "model": "hllm-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        data = response.json()

        # 检查必需字段
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            self.assertIn(field, data)

        # 检查 choices 结构
        self.assertIsInstance(data["choices"], list)
        if data["choices"]:
            choice = data["choices"][0]
            self.assertIn("index", choice)
            self.assertIn("message", choice)
            self.assertIn("finish_reason", choice)
            self.assertIn("role", choice["message"])
            self.assertIn("content", choice["message"])

    def test_chat_roles(self):
        """测试支持的聊天角色"""
        payload = {
            "model": "hllm-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ]
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
