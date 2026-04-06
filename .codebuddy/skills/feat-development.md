# Feat Development Skill

## 触发条件

当用户表达以下意图时激活：
- "开发新功能"
- "实现 xxx 功能"
- "添加 xxx 特性"
- "支持 xxx"
- "做一个 xxx"
- 任何涉及新增功能的请求

## 标准开发流程 (SOP)

### Phase 1: 需求澄清 (必须)

在开始任何实现前，先与用户确认：

```
1. 功能目标是什么？
2. 输入/输出接口定义？
3. 有哪些边界情况？
4. 性能/兼容性要求？
5. 是否需要配置项？
```

**禁止**: 直接开始写代码

---

### Phase 2: 设计文档 (必须)

在 `docs/designs/` 目录下创建设计文档：

**文件命名**: `docs/designs/YYYYMMDD-feature-name.md`

**文档模板**:

```markdown
# Feature: [功能名称]

## 概述

- 目标: [一句话描述]
- 优先级: P0/P1/P2
- 预计工作量: [小时/天]

## 需求分析

### 功能需求
- [ ] 需求1
- [ ] 需求2

### 非功能需求
- 性能: [要求]
- 兼容性: [要求]
- 可维护性: [要求]

## 设计方案

### 架构图
```
[ASCII 架构图]
```

### 接口定义

```python
# 公开 API
class NewFeature:
    def method(self, input: Type) -> OutputType:
        \"\"\"文档字符串\"\"\"
        pass
```

### 数据模型

```python
# 核心数据结构
@dataclass
class DataModel:
    field: type
```

## 实现计划

### Task Breakdown
- [ ] Task 1: [描述] (预计 x 小时)
- [ ] Task 2: [描述] (预计 x 小时)
- [ ] Task 3: [描述] (预计 x 小时)

### 依赖项
- 依赖1
- 依赖2

## 测试策略

### 单元测试覆盖
- [ ] 正常路径
- [ ] 边界条件
- [ ] 异常处理
- [ ] 性能测试 (如需要)

### 集成测试
- [ ] 端到端场景

## 风险与回滚

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| 风险1 | 高/中/低 | 高/中/低 | 措施 |

## 参考资料
- 相关文档链接
- 参考实现
```

**检查点**: 用户必须 approve 设计文档才能进入下一阶段

---

### Phase 3: 代码实现 (必须)

#### 3.1 代码规范

**文件组织**:
```
新功能应放在:
- hllm/[module]/[feature].py        # 核心实现
- hllm/[module]/__init__.py         # 导出公开 API
- tests/test_[feature].py           # 单元测试
- examples/example_[feature].py     # 使用示例 (可选)
```

**代码要求**:
- 类型注解 (Type Hints) 必须完整
- 文档字符串 (Docstrings) 遵循 Google Style
- 公开 API 必须有文档
- 配置项必须有默认值和验证

**示例**:

```python
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    \"\"\"功能配置。
    
    Attributes:
        enabled: 是否启用功能
        timeout: 超时时间(秒)
        max_retries: 最大重试次数
    \"\"\"
    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3

class NewFeature:
    \"\"\"新功能实现。
    
    Args:
        config: 功能配置
        
    Example:
        >>> feature = NewFeature(config=FeatureConfig())
        >>> result = feature.process(input_data)
    \"\"\"
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._validate_config()
    
    def process(self, input_data: str) -> str:
        \"\"\"处理输入数据。
        
        Args:
            input_data: 输入字符串
            
        Returns:
            处理后的字符串
            
        Raises:
            ValueError: 输入格式错误
            TimeoutError: 处理超时
        \"\"\"
        # 实现代码
        pass
    
    def _validate_config(self) -> None:
        \"\"\"验证配置有效性。\"\"\"
        if self.config.timeout <= 0:
            raise ValueError("timeout must be positive")
```

#### 3.2 实现检查清单

- [ ] 代码符合设计文档
- [ ] 所有公开方法有类型注解
- [ ] 所有公开方法有文档字符串
- [ ] 错误处理完善
- [ ] 日志记录适当
- [ ] 配置项可定制

---

### Phase 4: 单元测试 (必须)

**测试文件**: `tests/test_[feature].py`

**覆盖率要求**: 新代码覆盖率 >= 80%

**测试模板**:

```python
import pytest
from hllm.new_feature import NewFeature, FeatureConfig

class TestNewFeature:
    \"\"\"NewFeature 单元测试。\"\"\"
    
    def test_init_default_config(self):
        \"\"\"测试默认配置初始化。\"\"\"
        feature = NewFeature()
        assert feature.config.enabled is True
        assert feature.config.timeout == 30.0
    
    def test_init_custom_config(self):
        \"\"\"测试自定义配置。\"\"\"
        config = FeatureConfig(enabled=False, timeout=60.0)
        feature = NewFeature(config)
        assert feature.config.enabled is False
        assert feature.config.timeout == 60.0
    
    def test_process_normal_case(self):
        \"\"\"测试正常处理路径。\"\"\"
        feature = NewFeature()
        result = feature.process("input")
        assert result == "expected_output"
    
    def test_process_edge_case_empty_input(self):
        \"\"\"测试边界条件: 空输入。\"\"\"
        feature = NewFeature()
        with pytest.raises(ValueError, match="input cannot be empty"):
            feature.process("")
    
    def test_process_edge_case_long_input(self):
        \"\"\"测试边界条件: 超长输入。\"\"\"
        feature = NewFeature()
        long_input = "x" * 10000
        result = feature.process(long_input)
        assert len(result) > 0
    
    def test_process_invalid_input_type(self):
        \"\"\"测试异常处理: 错误输入类型。\"\"\"
        feature = NewFeature()
        with pytest.raises(TypeError):
            feature.process(123)  # type: ignore
    
    def test_validate_config_invalid_timeout(self):
        \"\"\"测试配置验证: 无效超时时间。\"\"\"
        config = FeatureConfig(timeout=-1)
        with pytest.raises(ValueError, match="timeout must be positive"):
            NewFeature(config)
```

**测试运行**:

```bash
# 运行新功能的测试
pytest tests/test_new_feature.py -v

# 检查覆盖率
pytest tests/test_new_feature.py --cov=hllm.new_feature --cov-report=term-missing

# 确保覆盖率 >= 80%
```

---

### Phase 5: 文档更新 (必须)

更新以下文档：

#### 5.1 README.md

在对应章节添加新功能说明：

```markdown
## 新功能

简要描述功能。

```python
# 使用示例
from hllm import NewFeature

feature = NewFeature()
result = feature.process("input")
```
```

#### 5.2 API 文档

如果新增 API，更新 `docs/api.md`

#### 5.3 CHANGELOG.md

添加变更记录：

```markdown
## [Unreleased]

### Added
- 新功能: [功能描述] (#PR号)
```

---

### Phase 6: 代码审查与提交 (必须)

#### 6.1 自查清单

提交前自检：

- [ ] 代码通过所有单元测试
- [ ] 覆盖率 >= 80%
- [ ] 类型检查通过 (pyright)
- [ ] 代码风格检查通过 (flake8/black)
- [ ] 设计文档已创建
- [ ] README 已更新
- [ ] CHANGELOG 已更新

#### 6.2 Git 提交规范

**分支命名**: `feat/feature-name`

**提交信息格式**:

```
feat: 简短描述

详细描述:
- 实现了什么
- 为什么这样实现
- 测试覆盖情况

Closes #issue-number
```

**示例**:

```bash
git commit -m "feat: 添加 PagedAttention 优化支持

- 实现 BlockManager 内存块管理
- 实现 PagedAttention 注意力计算
- 实现 Scheduler 连续批处理调度
- 添加单元测试，覆盖率 85%
- 更新 README 和 benchmark 报告

RTX 3080 Ti 性能提升 11% (58.3 -> 64.6 tok/s)"
```

---

### Phase 7: 发布 (按需)

#### 7.1 版本发布流程

如果是正式版本功能：

1. 更新 `pyproject.toml` 版本号
2. 更新 `CHANGELOG.md` 版本日期
3. 创建 Git Tag
4. 推送到 PyPI (如需要)

```bash
# 更新版本
poetry version minor  # 或 patch/major

# 创建 Tag
git tag -a v0.2.0 -m "Release v0.2.0: 添加 PagedAttention 支持"

# 推送
git push origin main --tags
```

#### 7.2 发布检查清单

- [ ] 版本号已更新
- [ ] CHANGELOG 已完善
- [ ] 所有测试通过
- [ ] 文档已部署
- [ ] Git Tag 已创建

---

## 快速参考

### 完整命令清单

```bash
# 1. 创建设计文档
mkdir -p docs/designs
touch docs/designs/$(date +%Y%m%d)-feature-name.md

# 2. 实现代码
# 编辑 hllm/module/feature.py

# 3. 编写测试
touch tests/test_feature.py

# 4. 运行测试
pytest tests/test_feature.py -v --cov=hllm.module

# 5. 类型检查
pyright hllm/module/feature.py

# 6. 代码风格
black hllm/module/feature.py tests/test_feature.py

# 7. 提交
git add .
git commit -m "feat: xxx"

# 8. 推送
git push origin feat/feature-name
```

### 常用模板

**设计文档模板**: 见 Phase 2
**代码模板**: 见 Phase 3
**测试模板**: 见 Phase 4

---

## 注意事项

1. **不要跳过设计文档**: 即使功能简单，也要写简要设计
2. **测试驱动**: 先写测试用例，再实现功能
3. **小步提交**: 每个逻辑单元独立提交
4. **文档同步**: 代码和文档同步更新
5. **及时沟通**: 遇到设计变更及时与用户确认

---

## 成功标准

一个完整的 feat 开发应该产出：

- ✅ 1 份设计文档 (docs/designs/)
- ✅ 1 个功能实现 (hllm/)
- ✅ 1 套单元测试 (tests/，覆盖率 >= 80%)
- ✅ 1 个使用示例 (examples/)
- ✅ 更新的 README
- ✅ 更新的 CHANGELOG
- ✅ 1 次代码提交
- ✅ (可选) 1 次版本发布

---

*Last Updated: 2026-04-06*
*Version: 1.0*
