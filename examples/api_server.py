#!/usr/bin/env python3
"""
HLLM REST API 服务端示例

启动 LLM 推理服务，提供 HTTP API。

用法:
    python api_server.py --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000

然后可以通过 API 访问:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello", "max_new_tokens": 50}'
"""

import argparse
import logging

from hllm import HLLM
from hllm.server import Server


def main():
    parser = argparse.ArgumentParser(
        description="HLLM REST API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --model ./TinyLlama-1.1B-Chat-v1.0
  %(prog)s --model ./TinyLlama-1.1B-Chat-v1.0 --port 8000 --device cpu
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
        "--debug",
        action="store_true",
        help="启用调试模式"
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("HLLM REST API Server")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info("=" * 50)

    # 加载模型
    logger.info("Loading model...")
    model = HLLM(model_path=args.model, device=args.device)
    logger.info("Model loaded successfully!")

    # 启动服务器
    server = Server(model, host=args.host, port=args.port)
    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 50)

    try:
        server.start(debug=args.debug)
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")


if __name__ == "__main__":
    main()
