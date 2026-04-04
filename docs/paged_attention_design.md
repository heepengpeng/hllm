# PagedAttention 设计文档

## 概述

PagedAttention 是 vLLM 的核心优化技术，本实现将其集成到 HLLM 框架中，为 NVIDIA GPU 提供高效的 LLM 推理能力。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        PagedAttention 架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Request   │───▶│  Scheduler  │───▶│   BlockManager      │  │
│  │    Queue    │    │  (调度器)    │    │    (内存管理器)      │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                              │                      │            │
│                              ▼                      ▼            │
│                     ┌─────────────────────────────────────┐     │
│                     │      PagedAttention Engine          │     │
│                     │  ┌─────────┐    ┌───────────────┐  │     │
│                     │  │  Block  │───▶│  Block Table  │  │     │
│                     │  │ Allocator│   │  (块映射表)    │  │     │
│                     │  └─────────┘    └───────────────┘  │     │
│                     │         │                  │        │     │
│                     │         ▼                  ▼        │     │
│                     │  ┌────────────────────────────────┐ │     │
│                     │  │    KV Cache (Physical Blocks)  │ │     │
│                     │  │  [Block 0][Block 1]...[Block N]│ │     │
│                     │  └────────────────────────────────┘ │     │
│                     └─────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│                     ┌─────────────────┐                         │
│                     │  Attention Compute│                        │
│                     │  (Flash Attention)│                        │
│                     └─────────────────┘                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. BlockManager - 内存块管理器

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks          # 总物理块数
        self.block_size = block_size          # 每块 token 数 (默认16)
        self.free_blocks = set(...)           # 空闲块集合
        self.block_tables = {}                # seq_id -> block_list 映射
```

**职责:**
- 物理内存块分配与回收
- Copy-on-Write (CoW) 机制实现
- 块共享与引用计数管理
- 内存碎片整理

**关键算法:**

#### 块分配策略
```python
def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
    """
    为序列分配物理块
    
    算法:
    1. 计算所需块数: ceil(num_tokens / block_size)
    2. 从 free_blocks 分配
    3. 更新 block_tables[seq_id]
    4. 返回分配的块 ID 列表
    """
    num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
    if len(self.free_blocks) < num_blocks_needed:
        raise MemoryError("Out of memory")
    
    allocated = list(self.free_blocks)[:num_blocks_needed]
    self.free_blocks -= set(allocated)
    self.block_tables[seq_id] = allocated
    return allocated
```

#### Copy-on-Write 机制
```python
def fork(self, parent_seq_id: int, child_seq_id: int):
    """
    复制序列的 block table，实现 CoW
    
    用于 beam search 或并行解码时共享 KV cache
    """
    parent_blocks = self.block_tables[parent_seq_id]
    self.block_tables[child_seq_id] = parent_blocks.copy()
    
    # 增加引用计数
    for block_id in parent_blocks:
        self.ref_count[block_id] += 1

def write(self, seq_id: int, block_idx: int) -> int:
    """
    写入前检查，如果块被共享则复制
    
    返回: 可写入的块 ID (可能是新分配的)
    """
    block_id = self.block_tables[seq_id][block_idx]
    
    if self.ref_count[block_id] > 1:
        # 块被共享，需要复制
        new_block = self._allocate_new_block()
        self._copy_block(block_id, new_block)
        self.ref_count[block_id] -= 1
        self.block_tables[seq_id][block_idx] = new_block
        return new_block
    
    return block_id
```

### 2. PagedAttention - 注意力计算引擎

```python
class PagedAttention:
    def __init__(self, num_heads: int, head_dim: int, block_size: int = 16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
```

**核心创新:**

传统 Attention:
```python
# 连续内存布局
K_cache = torch.zeros([batch_size, max_seq_len, num_heads, head_dim])
V_cache = torch.zeros([batch_size, max_seq_len, num_heads, head_dim])
```

PagedAttention:
```python
# 分块内存布局
kv_cache = torch.zeros([num_blocks, block_size, num_heads, head_dim])
block_table = [[0, 5, 3], [1, 2]]  # 每个序列的块映射

# 通过 block table 间接访问
def get_kv_cache(seq_idx, position):
    block_idx = position // block_size
    offset = position % block_size
    block_id = block_table[seq_idx][block_idx]
    return kv_cache[block_id, offset]
```

**计算流程:**

```python
def forward(self, query, key_cache, value_cache, block_table, context_lens):
    """
    Args:
        query: [batch_size, num_heads, head_dim]
        key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: [batch_size, max_blocks_per_seq]
        context_lens: [batch_size] - 每个序列的实际长度
    
    Returns:
        output: [batch_size, num_heads, head_dim]
    """
    batch_size = query.shape[0]
    
    # 1. 根据 block_table 收集 KV cache
    # key_cache_gathered: [batch_size, max_seq_len, num_kv_heads, head_dim]
    key_cache_gathered = self._gather_kv_cache(key_cache, block_table, context_lens)
    value_cache_gathered = self._gather_kv_cache(value_cache, block_table, context_lens)
    
    # 2. 计算注意力 (支持 Flash Attention)
    output = self._flash_attention(query, key_cache_gathered, value_cache_gathered)
    
    return output
```

### 3. Scheduler - 请求调度器

```python
class Scheduler:
    def __init__(self, max_num_seqs: int = 256, max_model_len: int = 4096):
        self.waiting = deque()      # 等待队列
        self.running = []           # 运行中的序列
        self.swapped = []           # 被换出的序列 (显存不足)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
```

**调度策略:**

#### 连续批处理 (Continuous Batching)
```python
def schedule(self) -> Tuple[List[Sequence], List[Sequence], List[Sequence]]:
    """
    调度策略:
    1. 优先处理运行中的序列 (self.running)
    2. 如果有空余槽位，从 waiting 队列取新请求
    3. 如果显存不足，将部分序列 swap 到 CPU
    
    Returns:
        (scheduled_running, scheduled_new, scheduled_swap)
    """
    # 1. 保留所有运行中的序列
    scheduled = self.running.copy()
    
    # 2. 尝试加入新请求
    while len(scheduled) < self.max_num_seqs and self.waiting:
        seq = self.waiting.popleft()
        
        # 检查是否可以分配块
        try:
            self.block_manager.allocate(seq.seq_id, len(seq.tokens))
            scheduled.append(seq)
        except MemoryError:
            # 显存不足，放回等待队列
            self.waiting.appendleft(seq)
            break
    
    # 3. 如果仍然显存不足，需要 swap out
    if not self._has_enough_memory(scheduled):
        scheduled = self._swap_out(scheduled)
    
    return scheduled
```

#### 迭代级调度
```python
def step(self) -> List[SequenceOutput]:
    """
    每次迭代执行一次调度:
    1. 选择要运行的序列
    2. 执行模型前向
    3. 采样生成新 token
    4. 更新序列状态
    5. 将完成的序列移出
    """
    # 1. 调度
    scheduled_seqs = self.schedule()
    
    # 2. 准备输入
    input_tokens = self._prepare_input(scheduled_seqs)
    block_tables = self._prepare_block_tables(scheduled_seqs)
    
    # 3. 模型推理
    logits = self.model.forward(input_tokens, block_tables)
    
    # 4. 采样
    new_tokens = self._sample(logits)
    
    # 5. 更新状态
    for seq, new_token in zip(scheduled_seqs, new_tokens):
        seq.append_token(new_token)
        
        # 为新 token 分配块 (如果需要)
        if len(seq.tokens) % self.block_size == 1:
            self.block_manager.append_block(seq.seq_id)
    
    # 6. 移除完成的序列
    done = [s for s in scheduled_seqs if s.is_done()]
    self.running = [s for s in scheduled_seqs if not s.is_done()]
    
    return [SequenceOutput(s.seq_id, s.output_tokens) for s in done]
```

## 内存管理机制

### 物理块布局

```
┌─────────────────────────────────────────────────────────────┐
│                    Physical KV Cache                         │
├─────────────────────────────────────────────────────────────┤
│ Block 0 │ Block 1 │ Block 2 │ ... │ Block N │ Free Blocks   │
├─────────┴─────────┴─────────┴─────┴─────────┴───────────────┤
│                                                             │
│  Block Table (Sequence 0): [0, 2, 5]                        │
│  Block Table (Sequence 1): [1, 3]                           │
│  Block Table (Sequence 2): [4]                              │
│                                                             │
│  Tokens:                                                    │
│    Seq 0: [T0-T15] in Block 0, [T16-T31] in Block 2, ...   │
│    Seq 1: [T0-T15] in Block 1, [T16-T22] in Block 3        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 内存分配策略

1. **预分配**: 根据 max_num_blocks 预先分配 GPU 显存
2. **按需分配**: 序列生成过程中动态分配新块
3. **回收复用**: 序列完成后立即回收块到 free pool
4. **碎片整理**: 定期合并连续空闲块

### 显存计算

```python
# 显存计算公式
memory_per_block = block_size * num_layers * num_kv_heads * head_dim * 2 * dtype_size
# 2 表示 K + V, dtype_size: fp16=2, fp32=4

total_memory = num_blocks * memory_per_block

# 示例: Llama-3.2-1B
# block_size=16, num_layers=16, num_kv_heads=8, head_dim=64
# memory_per_block = 16 * 16 * 8 * 64 * 2 * 2 = 512 KB
# num_blocks = 1024 (约 512 MB)
```

## 连续批处理 (Continuous Batching)

### 传统静态批处理

```
时间 →
│
│ [Req A: ============]
│ [Req B: =====]       (等待 A 完成)
│ [Req C: ==============] (等待 B 完成)
│
└─────────────────────────────────────
总时间 = A + B + C
GPU 利用率 = 低 (等待时闲置)
```

### PagedAttention 连续批处理

```
时间 →
│
│ Batch 1: [A0, B0, C0]  (同时启动)
│ Batch 2: [A1, B1, D0]  (C 完成，D 加入)
│ Batch 3: [A2, B2, D1]  
│ Batch 4: [A3, D2]      (B 完成)
│ ...
│
└─────────────────────────────────────
总时间 ≈ max(A, B, C, D) + overhead
GPU 利用率 = 高 (始终满批)
```

### 动态批处理实现

```python
class ContinuousBatchingEngine:
    def run(self):
        while self.has_requests():
            # 1. 尝试将 waiting 中的请求加入 running batch
            self._add_new_requests_to_batch()
            
            # 2. 执行一次模型推理 (所有 running 请求)
            outputs = self._forward()
            
            # 3. 更新每个请求的状态
            for seq, output in zip(self.running, outputs):
                seq.add_token(output)
                
                # 检查是否完成
                if seq.is_finished():
                    self._complete_sequence(seq)
                    self.running.remove(seq)
            
            # 4. 为新请求腾出位置
            self._admit_waiting_requests()
```

## 性能优化策略

### 1. Flash Attention 集成

```python
# 使用 Flash Attention 加速注意力计算
from flash_attn import flash_attn_func

def compute_attention(query, key, value):
    # Flash Attention: O(N) 内存 instead of O(N^2)
    return flash_attn_func(query, key, value, causal=True)
```

### 2. CUDA Graph 捕获

```python
# 捕获静态计算图，减少 CPU 开销
cuda_graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(cuda_graph):
    static_output = model(static_input)

# 重放 (零 CPU 开销)
cuda_graph.replay()
```

### 3. Kernel Fusion

```python
# 融合多个小 kernel 为大 kernel
# 例如: rope + attention + projection 合并为单个 kernel
```

## 与 vLLM 的对比

| 特性 | vLLM | HLLM PagedAttention |
|------|------|---------------------|
| 语言 | Python/C++ | Python |
| 易用性 | 中等 | 高 (纯 Python) |
| 性能 | 极高 | 高 |
| 定制化 | 困难 | 容易 |
| 学习曲线 | 陡峭 | 平缓 |
| 适用场景 | 生产环境 | 研究/原型/中等规模 |

**定位差异:**
- **vLLM**: 生产级高性能推理引擎
- **HLLM PagedAttention**: 教育/研究友好的轻量级实现

## 使用指南

### Python API

```python
from hllm import HLLM

# 自动启用 PagedAttention (CUDA 环境下)
model = HLLM("meta-llama/Llama-3.2-1B-Instruct")

# 手动指定
model = HLLM(
    "meta-llama/Llama-3.2-1B-Instruct",
    backend="paged_pytorch",
    device="cuda",
    block_size=16,          # 每块 token 数
    num_blocks=1024,        # 总块数
    max_num_seqs=256        # 最大并发序列数
)

# 生成
text = model.generate("Hello, how are you?", max_new_tokens=100)
```

### REST API

```bash
# 启动服务器
python -m hllm.server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend paged_pytorch \
    --device cuda \
    --block-size 16 \
    --num-blocks 1024

# 请求
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_new_tokens": 100
  }'
```

### 性能调优

```python
# 1. 调整 block_size (默认 16)
# 较小的 block_size: 更细粒度内存管理，更多开销
# 较大的 block_size: 更少开销，更多内部碎片

# 2. 调整 num_blocks
# 根据 GPU 显存计算: num_blocks = available_memory / memory_per_block

# 3. 调整 max_num_seqs
# 根据并发需求设置

# 示例: RTX 3080 Ti (12GB)
model = HLLM(
    "model-name",
    backend="paged_pytorch",
    block_size=16,
    num_blocks=2048,    # 约 1GB KV cache
    max_num_seqs=128
)
```

## 实现细节

### 文件结构

```
hllm/
├── paged_attention/
│   ├── __init__.py
│   ├── block_manager.py      # 内存块管理
│   ├── paged_attention.py    # 注意力计算
│   └── scheduler.py          # 请求调度
├── backends/
│   ├── paged_pytorch.py      # PagedAttention PyTorch 后端
│   └── __init__.py
└── docs/
    └── paged_attention_design.md  # 本文档
```

### 关键数据结构

```python
@dataclass
class Sequence:
    seq_id: int
    tokens: List[int]
    output_tokens: List[int]
    block_table: List[int]  # 物理块 ID 列表
    status: SequenceStatus  # WAITING/RUNNING/DONE

@dataclass
class Block:
    block_id: int
    ref_count: int          # 引用计数 (用于 CoW)
    is_allocated: bool
```

## 测试与验证

### 单元测试

```bash
# 测试 BlockManager
python -m pytest tests/test_block_manager.py -v

# 测试 PagedAttention
python -m pytest tests/test_paged_attention.py -v

# 测试 Scheduler
python -m pytest tests/test_scheduler.py -v
```

### 性能基准测试

```bash
# 对比 Standard PyTorch vs PagedAttention
python examples/benchmark_paged.py
```

## 未来改进

1. **Prefix Caching**: 共享公共前缀的 KV cache
2. **Speculative Decoding**: 使用小模型预测 token
3. **Chunked Prefill**: 长 prompt 分块处理
4. **Pipeline Parallelism**: 多 GPU 流水线并行
5. **Tensor Parallelism**: 多 GPU 张量并行

## 参考

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Continuous Batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)

---

*文档版本: 1.0*  
*更新日期: 2026-04-05*  
*作者: HLLM Team*
