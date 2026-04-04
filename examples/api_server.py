#!/usr/bin/env python3
"""
HLLM OpenAI Compatible API Server Example

启动与 OpenAI API 兼容的 LLM 推理服务。

用法:
    python api_server.py --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000

然后可以通过 OpenAI 客户端或任意兼容工具访问:
    curl http://localhost:8000/v1/models
    curl -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "hllm-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 50
        }'

OpenAI 官方客户端示例:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="hllm-model",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
"""

import argparse
import logging

from hllm import HLLM
from hllm.server import Server


def main():
    parser = argparse.ArgumentParser(
        description="HLLM OpenAI Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --model ./TinyLlama-1.1B-Chat-v1.0
  %(prog)s --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000 --device cpu
  %(prog)s --model microsoft/Phi-3-mini-4k-instruct --port 8000
        """
    )
    parser.add_argument(
        "--model",
        required=True,
        help="模型路径（HuggingFace 路径或本地路径）"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="运行设备 (默认: cpu)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口 (默认: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式）"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("HLLM OpenAI Compatible API Server")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info("=" * 60)

    # 加载模型
    logger.info("Loading model...")
    model = HLLM(model_path=args.model, device=args.device)
    logger.info("Model loaded successfully!")

    # 启动服务器
    server = Server(model, host=args.host, port=args.port)
    logger.info(f"API Endpoint: http://{args.host}:{args.port}/v1")
    logger.info(f"Health Check: http://{args.host}:{args.port}/health")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        server.start(reload=args.reload)
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")


if __name__ == "__main__":
    main()
