"""
HLLM REST API Server

提供基于 HTTP 的 LLM 推理服务。
"""

import argparse
import json
import logging
from typing import Optional, List, Dict, Any

from flask import Flask, request, jsonify, Response
from flask.typing import ResponseReturnValue

from .model import HLLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局模型实例
_model: Optional[HLLM] = None


def get_model() -> HLLM:
    """获取全局模型实例"""
    if _model is None:
        raise RuntimeError("Model not initialized")
    return _model


@app.route("/health", methods=["GET"])
def health() -> ResponseReturnValue:
    """健康检查端点"""
    try:
        model = get_model()
        return jsonify({
            "status": "ok",
            "model": model.model_path,
            "device": model.device
        })
    except RuntimeError as e:
        return _error_response("SERVICE_UNAVAILABLE", str(e), 503)


@app.route("/models", methods=["GET"])
def get_models() -> ResponseReturnValue:
    """获取模型信息"""
    model = get_model()
    config = model.config
    return jsonify({
        "model": model.model_path,
        "device": model.device,
        "max_length": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None)
    })


@app.route("/generate", methods=["POST"])
def generate() -> ResponseReturnValue:
    """文本生成端点"""
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return _error_response("INVALID_REQUEST", "Request body is required", 400)

        prompt = data.get("prompt")
        if not prompt:
            return _error_response("INVALID_REQUEST", "Missing required field: prompt", 400)

        # 获取可选参数
        max_new_tokens = data.get("max_new_tokens", 100)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)
        stream = data.get("stream", False)

        model = get_model()

        if stream:
            return _stream_response(prompt, max_new_tokens, temperature, top_p, top_k)

        # 非流式生成
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = input_ids.shape[1]

        text = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        output_ids = model.tokenizer.encode(text, return_tensors="pt")
        completion_tokens = output_ids.shape[1]

        return jsonify({
            "text": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

    except Exception as e:
        logger.exception("Error in generate endpoint")
        return _error_response("INTERNAL_ERROR", str(e), 500)


@app.route("/chat", methods=["POST"])
def chat() -> ResponseReturnValue:
    """对话生成端点"""
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return _error_response("INVALID_REQUEST", "Request body is required", 400)

        messages = data.get("messages")
        if not messages or not isinstance(messages, list):
            return _error_response("INVALID_REQUEST", "Missing or invalid field: messages", 400)

        # 获取可选参数
        max_new_tokens = data.get("max_new_tokens", 200)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)

        model = get_model()

        # 使用 chat template 格式化消息
        prompt = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = input_ids.shape[1]

        text = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        output_ids = model.tokenizer.encode(text, return_tensors="pt")
        completion_tokens = output_ids.shape[1]

        return jsonify({
            "message": {
                "role": "assistant",
                "content": text
            },
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

    except Exception as e:
        logger.exception("Error in chat endpoint")
        return _error_response("INTERNAL_ERROR", str(e), 500)


@app.route("/generate/stream", methods=["POST"])
def generate_stream() -> ResponseReturnValue:
    """流式生成端点（SSE）"""
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return _error_response("INVALID_REQUEST", "Request body is required", 400)

        prompt = data.get("prompt")
        if not prompt:
            return _error_response("INVALID_REQUEST", "Missing required field: prompt", 400)

        max_new_tokens = data.get("max_new_tokens", 100)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)

        return _stream_response(prompt, max_new_tokens, temperature, top_p, top_k)

    except Exception as e:
        logger.exception("Error in generate_stream endpoint")
        return _error_response("INTERNAL_ERROR", str(e), 500)


def _stream_response(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int
) -> Response:
    """生成 SSE 流式响应"""
    model = get_model()

    def generate_sse():
        index = 0
        for token in model.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        ):
            data = json.dumps({"token": token, "index": index})
            yield f"data: {data}\n\n"
            index += 1
        yield "data: [DONE]\n\n"

    return Response(
        generate_sse(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


def _error_response(code: str, message: str, status_code: int) -> ResponseReturnValue:
    """生成错误响应"""
    response = jsonify({
        "error": {
            "code": code,
            "message": message
        }
    })
    response.status_code = status_code
    return response


class Server:
    """HLLM REST API 服务器"""

    def __init__(
        self,
        model: HLLM,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        初始化服务器

        Args:
            model: HLLM 模型实例
            host: 监听地址
            port: 监听端口
        """
        global _model
        _model = model
        self.host = host
        self.port = port

    def start(self, debug: bool = False) -> None:
        """
        启动服务器

        Args:
            debug: 是否开启调试模式
        """
        logger.info(f"Starting HLLM server on {self.host}:{self.port}")
        app.run(host=self.host, port=self.port, debug=debug, threaded=True)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="HLLM REST API Server")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}...")
    model = HLLM(model_path=args.model, device=args.device)

    server = Server(model, host=args.host, port=args.port)
    server.start(debug=args.debug)


if __name__ == "__main__":
    main()
