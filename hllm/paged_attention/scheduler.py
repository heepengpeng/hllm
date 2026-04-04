"""Scheduler: 管理请求的调度"""

import torch
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """推理请求"""
    req_id: int
    seq_id: int
    input_ids: torch.Tensor
    max_new_tokens: int
    generated_tokens: int = 0
    is_prefill: bool = True  # 是否是 prefill 阶段
    
    def is_finished(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens


class Scheduler:
    """
    简单的请求调度器
    
    功能：
    1. 管理 waiting/running 队列
    2. 决定当前 batch 包含哪些请求
    3. 支持 continuous batching
    """
    
    def __init__(
        self,
        max_batch_size: int = 16,
        max_num_seqs: int = 256
    ):
        self.max_batch_size = max_batch_size
        self.max_num_seqs = max_num_seqs
        
        # 请求队列
        self.waiting_queue: deque[Request] = deque()
        self.running_queue: List[Request] = []
        
        self.next_req_id = 0
        
        logger.info(f"Scheduler initialized: max_batch_size={max_batch_size}")
    
    def add_request(
        self,
        seq_id: int,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> int:
        """
        添加新请求到等待队列
        
        Args:
            seq_id: 序列 ID
            input_ids: 输入 token IDs
            max_new_tokens: 最大生成 token 数
            
        Returns:
            req_id: 请求 ID
        """
        req_id = self.next_req_id
        self.next_req_id += 1
        
        req = Request(
            req_id=req_id,
            seq_id=seq_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens
        )
        
        self.waiting_queue.append(req)
        logger.debug(f"Added request {req_id} to waiting queue")
        
        return req_id
    
    def schedule(self) -> Tuple[List[Request], bool]:
        """
        调度请求，决定当前 batch
        
        Returns:
            (batch_requests, is_prompt): 当前 batch 的请求列表和是否是 prefill 阶段
        """
        # 如果有 running 的请求，优先继续处理（continuous batching）
        if self.running_queue:
            # 尝试从 waiting 队列添加新请求
            while (
                len(self.running_queue) < self.max_batch_size
                and self.waiting_queue
            ):
                req = self.waiting_queue.popleft()
                self.running_queue.append(req)
            
            return self.running_queue, False
        
        # 没有 running 的请求，从 waiting 队列取
        if self.waiting_queue:
            batch_size = min(len(self.waiting_queue), self.max_batch_size)
            batch = []
            for _ in range(batch_size):
                req = self.waiting_queue.popleft()
                self.running_queue.append(req)
                batch.append(req)
            
            return batch, True
        
        return [], False
    
    def update_request(self, req_id: int, generated_token: int):
        """更新请求状态"""
        for req in self.running_queue:
            if req.req_id == req_id:
                req.generated_tokens += 1
                req.is_prefill = False
                break
    
    def finish_request(self, req_id: int):
        """完成请求，从 running 队列移除"""
        self.running_queue = [r for r in self.running_queue if r.req_id != req_id]
        logger.debug(f"Finished request {req_id}")
    
    def has_work(self) -> bool:
        """检查是否还有待处理的工作"""
        return len(self.waiting_queue) > 0 or len(self.running_queue) > 0
    
    def get_num_waiting(self) -> int:
        """获取等待队列长度"""
        return len(self.waiting_queue)
    
    def get_num_running(self) -> int:
        """获取运行中队列长度"""
        return len(self.running_queue)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_queue),
            "max_batch_size": self.max_batch_size
        }
