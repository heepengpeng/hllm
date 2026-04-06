"""
HLLM OpenAI Compatible REST API Server (FastAPI)

提供与 OpenAI API 兼容的 HTTP 服务。
"""

import logging
import time
import uuid
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field

from .model import HLLM
from .config import get_config, ServerConfig, ModelConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局模型实例
_model: Optional[HLLM] = None


def get_model() -> HLLM:
    """获取全局模型实例"""
    if _model is None:
        raise RuntimeError("Model not initialized")
    return _model


# ============== Pydantic Models ==============

class ChatMessage(BaseModel):
    """聊天消息"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """对话补全请求"""
    model: str = "hllm-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=100, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionResponseChoice(BaseModel):
    """对话补全响应选项"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    """对话补全响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict


class ChatCompletionStreamChoice(BaseModel):
    """流式对话补全选项"""
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """流式对话补全响应"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class CompletionRequest(BaseModel):
    """文本补全请求"""
    model: str = "hllm-model"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=100, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionResponseChoice(BaseModel):
    """文本补全响应选项"""
    text: str
    index: int
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    """文本补全响应"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: dict


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "hllm"


class ModelListResponse(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: List[ModelInfo]


class ErrorResponse(BaseModel):
    """错误响应"""
    error: dict


# ============== FastAPI App ==============

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="HLLM OpenAI Compatible API",
        description="A lightweight LLM inference framework with OpenAI compatible API",
        version="0.2.0"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        """获取模型列表"""
        return ModelListResponse(
            data=[
                ModelInfo(
                    id="hllm-model",
                    created=int(time.time())
                )
            ]
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        """对话补全（OpenAI 兼容）"""
        try:
            model = get_model()

            # 使用 chat template 格式化消息
            messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
            # 使用 tokenizer 的 chat template
            if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'apply_chat_template'):
                prompt = model.tokenizer.apply_chat_template(
                    messages_dict,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 简单拼接
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_dict])
                prompt += "\nassistant:"

            if request.stream:
                return StreamingResponse(
                    _stream_chat_completion(request, prompt),
                    media_type="text/event-stream"
                )

            # 计算 prompt tokens
            prompt_tokens = len(model.tokenizer.encode(prompt))

            # 生成
            start_time = time.time()
            text = model.generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
            completion_time = time.time() - start_time

            completion_tokens = len(model.tokenizer.encode(text))

            logger.info(f"Chat completion: {prompt_tokens} prompt tokens, "
                       f"{completion_tokens} completion tokens, "
                       f"{completion_time:.2f}s")

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text),
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

        except Exception as e:
            logger.exception("Error in chat_completions")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest):
        """文本补全（OpenAI 兼容）"""
        try:
            model = get_model()

            # 处理 prompt
            if isinstance(request.prompt, list):
                prompt = request.prompt[0] if request.prompt else ""
            else:
                prompt = request.prompt

            if request.stream:
                return StreamingResponse(
                    _stream_completion(request, prompt),
                    media_type="text/event-stream"
                )

            # 计算 prompt tokens
            prompt_tokens = len(model.tokenizer.encode(prompt))

            # 生成
            text = model.generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )

            completion_tokens = len(model.tokenizer.encode(text))

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionResponseChoice(
                        text=text,
                        index=0,
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

        except Exception as e:
            logger.exception("Error in completions")
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_chat_completion(request: ChatCompletionRequest, prompt: str):
        """流式对话补全生成器"""
        model = get_model()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # 发送角色
        yield f"data: {ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None
            )]
        ).model_dump_json()}\n\n"

        # 流式生成
        for token in model.stream_generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        ):
            yield f"data: {ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": token},
                    finish_reason=None
                )]
            ).model_dump_json()}\n\n"

        # 发送结束标记
        yield f"data: {ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        ).model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    async def _stream_completion(request: CompletionRequest, prompt: str):
        """流式文本补全生成器"""
        model = get_model()
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # 流式生成
        for token in model.stream_generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        ):
            yield f"data: {ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": token},
                    finish_reason=None
                )]
            ).model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    @app.get("/health")
    async def health():
        """健康检查"""
        try:
            model = get_model()
            return {
                "status": "ok",
                "model": model.model_path,
                "backend": model.backend_name,
                "info": model.get_info()
            }
        except RuntimeError:
            raise HTTPException(status_code=503, detail="Model not initialized")

except ImportError:
    app = None
    logger.warning("FastAPI not installed, server module unavailable")


# ============== Server Class ==============

class Server:
    """HLLM OpenAI Compatible REST API Server"""

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

    def start(self, reload: bool = False) -> None:
        """
        启动服务器

        Args:
            reload: 是否启用热重载
        """
        if app is None:
            raise RuntimeError("FastAPI not installed. Install with: pip install fastapi uvicorn")

        import uvicorn
        logger.info(f"Starting HLLM OpenAI compatible server on {self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port, reload=reload)


def main(
    model: Optional[str] = None,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
) -> None:
    """
    命令行入口
    
    Args:
        model: 模型路径或 HuggingFace model ID
        backend: 推理后端 (auto/pytorch/mlx/paged_pytorch)
        device: 设备 (cpu/cuda/mps)
        host: 监听地址
        port: 监听端口
        reload: 是否启用热重载
        
    Example:
        >>> from hllm.server import main
        >>> main(model="microsoft/Phi-3-mini-4k-instruct", port=8080)
        
        或使用命令行:
        $ python -m hllm.server --model microsoft/Phi-3-mini-4k-instruct --port 8080
    """
    # 加载配置（支持环境变量和配置文件）
    config = get_config()
    
    # 命令行参数覆盖配置文件
    model_path = model or config.model.path
    backend_name = backend or config.model.backend
    device_name = device or config.model.device
    host_addr = host or config.server.host
    port_num = port or config.server.port
    
    logger.info(f"Loading model from {model_path}...")
    logger.info(f"Backend: {backend_name}")
    logger.info(f"Server: {host_addr}:{port_num}")

    # 构建模型参数
    model_kwargs: dict = {"backend": backend_name}
    if device_name:
        model_kwargs["device"] = device_name

    model_instance = HLLM(model_path=model_path, **model_kwargs)
    logger.info(f"Model loaded. Backend info: {model_instance.get_info()}")

    server = Server(model=model_instance, host=host_addr, port=port_num)
    server.start(reload=reload or config.server.reload)


def cli_main() -> None:
    """CLI 入口点"""
    import argparse
    
    config = get_config()
    
    parser = argparse.ArgumentParser(description="HLLM OpenAI Compatible REST API Server")
    parser.add_argument(
        "--model", 
        default=config.model.path,
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--backend", 
        default=config.model.backend,
        choices=["auto", "pytorch", "mlx", "paged_pytorch"],
        help="Inference backend"
    )
    parser.add_argument(
        "--device", 
        default=config.model.device,
        help="Device for PyTorch backend (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--host", 
        default=config.server.host,
        help="Host to bind"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=config.server.port,
        help="Port to bind"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        default=config.server.reload,
        help="Enable auto-reload"
    )
    parser.add_argument(
        "--config",
        help="Path to config file (YAML)"
    )

    args = parser.parse_args()
    
    # 如果指定了配置文件，重新加载
    if args.config:
        from .config import reload_config
        import os
        os.environ["HLLM_CONFIG_FILE"] = args.config
        reload_config()

    main(
        model=args.model,
        backend=args.backend,
        device=args.device,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    cli_main()
