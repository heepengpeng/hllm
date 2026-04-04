# PyPI 自动发布指南

## 配置步骤

### 1. 获取 PyPI API Token

1. 访问 [PyPI Account Settings](https://pypi.org/manage/account/)
2. 滚动到 "API tokens" 部分
3. 点击 "Add API token"
4. 输入 Token 名称，如 `GitHub Actions Release`
5. 选择 Scope:
   - 如果是第一次发布：`Entire account (all projects)`
   - 如果项目已存在：选择 `light-llm-hp` 项目
6. 点击 "Create token"
7. **复制生成的 token**（只显示一次！）

### 2. 配置 GitHub Secrets

1. 打开 GitHub 仓库页面
2. 点击 Settings -> Secrets and variables -> Actions
3. 点击 "New repository secret"
4. 添加以下 secret:
   - **Name**: `PYPI_API_TOKEN`
   - **Secret**: 刚才复制的 PyPI token

### 3. 使用方法

#### 方式一：推送 Tag 触发

```bash
# 1. 更新版本号（修改 pyproject.toml 中的 version）
vim pyproject.toml

# 2. 提交版本更新
git add pyproject.toml
git commit -m "chore: bump version to 0.4.3"

# 3. 创建 tag
git tag v0.4.3

# 4. 推送到 GitHub
git push origin main
git push origin v0.4.3
```

推送 tag 后，GitHub Actions 会自动：
- 构建包
- 发布到 PyPI
- 创建 GitHub Release

#### 方式二：手动触发

也可以手动触发工作流：
1. 进入 GitHub 仓库 Actions 页面
2. 选择 "Release to PyPI" 工作流
3. 点击 "Run workflow"

## 版本号规范

使用 [语义化版本](https://semver.org/lang/zh-CN/)：

- `MAJOR.MINOR.PATCH`
- 例如：`v0.4.2`

### 版本号示例

| 版本 | 说明 |
|------|------|
| v0.4.2 | 补丁版本（bug 修复） |
| v0.5.0 | 次要版本（新功能，向后兼容） |
| v1.0.0 | 主要版本（破坏性变更） |

## 检查发布状态

### 查看 Actions 运行状态

1. 打开 GitHub 仓库
2. 点击 Actions 标签
3. 查看 "Release to PyPI" 工作流运行状态

### 验证 PyPI 发布

发布成功后，可以在以下地址查看：
- https://pypi.org/project/light-llm-hp/
- https://pypi.org/project/light-llm-hp/X.Y.Z/ （具体版本）

### 验证安装

```bash
pip install --upgrade light-llm-hp
```

## 故障排除

### 问题 1: 版本号已存在

**错误信息**: `File already exists`

**解决**: PyPI 不允许重复上传相同版本号。需要更新版本号后重新发布。

### 问题 2: 认证失败

**错误信息**: `Invalid API Token`

**解决**: 
1. 检查 GitHub Secrets 中的 `PYPI_API_TOKEN` 是否正确
2. 重新生成 PyPI token 并更新

### 问题 3: 构建失败

**错误信息**: `Build failed`

**解决**: 
1. 本地测试构建：`python -m build`
2. 检查 `pyproject.toml` 配置
3. 确保所有文件已提交到 git

## 安全建议

1. **不要**将 PyPI token 提交到代码仓库
2. 使用 GitHub Environments 保护生产部署
3. 定期轮换 API token
4. 为 token 设置最小权限（仅指定项目）

## 参考文档

- [PyPI API Token 文档](https://pypi.org/help/#apitoken)
- [GitHub Actions 文档](https://docs.github.com/cn/actions)
- [Python Packaging Guide](https://packaging.python.org/)
