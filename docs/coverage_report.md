# HLLM 测试覆盖率报告

## 概览

- **总体覆盖率**: 72%
- **测试通过**: 94 个
- **测试文件**: 14 个
- **目标覆盖率**: 80%

## 各模块覆盖率

| 模块 | 语句数 | 未覆盖 | 覆盖率 | 状态 |
|------|-------|-------|-------|------|
| hllm/tokenizer.py | 22 | 0 | **100%** | ✅ |
| hllm/client.py | 149 | 16 | **89%** | ✅ |
| hllm/backends/pytorch.py | 60 | 10 | **83%** | ✅ |
| hllm/model.py | 53 | 10 | **81%** | ✅ |
| hllm/server.py | 180 | 38 | **79%** | ✅ |
| hllm/backends/base.py | 43 | 9 | **79%** | ✅ |
| hllm/backends/__init__.py | 48 | 11 | **77%** | ✅ |
| hllm/__init__.py | 15 | 4 | **73%** | ✅ |
| hllm/generate.py | 86 | 59 | **31%** | ⚠️ |
| hllm/backends/mlx.py | 74 | 51 | **31%** | ⚠️ |

## 测试文件列表

| 测试文件 | 描述 |
|---------|------|
| test_hllm.py | HLLM 基础导入测试 |
| test_model.py | HLLM 类属性测试 |
| test_model_init.py | HLLM 初始化测试 |
| test_hllm_extended.py | HLLM 扩展功能测试 |
| test_tokenizer.py | Tokenizer 测试 |
| test_generate.py | 生成模块基础测试 |
| test_generate_simple.py | 生成模块简单测试 |
| test_generate_more.py | 生成参数测试 |
| test_generate_core.py | 生成核心测试 |
| test_generate_helpers.py | 生成辅助函数测试 |
| test_backends.py | 后端模块导入测试 |
| test_backends_extended.py | 后端扩展测试 |
| test_create_backend.py | 后端创建测试 |
| test_auto_select.py | 自动选择后端测试 |
| test_base_backend.py | BaseBackend 测试 |
| test_pytorch_backend.py | PyTorch 后端测试 |
| test_mlx_backend.py | MLX 后端测试 |
| test_server.py | REST API 服务端测试 |
| test_server_simple.py | 服务端简单测试 |
| test_client.py | OpenAI 兼容客户端测试 |
| test_init.py | 包初始化测试 |

## 覆盖率缺口分析

### 1. generate.py (31%)

**原因**: 需要实际的 PyTorch 模型才能完整测试

**未覆盖代码**:
- 主生成循环 (lines 45-99)
- 流式生成循环 (lines 130-184)

**建议**: 使用小型预训练模型进行集成测试

### 2. mlx.py (31%)

**原因**: MLX 是 Apple Silicon 专属可选依赖

**未覆盖代码**:
- MLX 模型加载 (lines 31-52)
- MLX 生成逻辑 (lines 65-88, 101-126)

**建议**: 在 Apple Silicon 设备上运行特定测试

## 运行测试

```bash
# 运行所有测试
python -m pytest tests/ --cov=hllm --cov-report=term

# 生成 HTML 报告
python -m pytest tests/ --cov=hllm --cov-report=html

# 查看覆盖率详情
python -m pytest tests/ --cov=hllm --cov-report=term-missing
```

## 提升覆盖率计划

1. **短期 (72% → 75%)**:
   - 添加更多边界条件测试
   - 补充错误处理测试

2. **中期 (75% → 80%)**:
   - 使用小型模型进行集成测试
   - 添加端到端测试

3. **长期 (80% → 90%)**:
   - 在 Apple Silicon 上测试 MLX 后端
   - 添加性能测试

---

*报告生成时间: 2025-04-04*
