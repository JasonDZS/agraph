# 开发指南

欢迎参与 Knowledge Graph Toolkit 的开发！本指南将帮助你快速设置开发环境并了解开发流程。

## 目录

- [环境设置](#环境设置)
- [开发流程](#开发流程)
- [代码质量检查](#代码质量检查)
- [测试](#测试)
- [提交规范](#提交规范)
- [发布流程](#发布流程)
- [常见问题](#常见问题)

## 环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/JasonDZS/agraph.git
cd agraph
```

### 2. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或者
venv\Scripts\activate     # Windows

# 使用 conda
conda create -n agraph python=3.11
conda activate agraph

# 使用uv(推荐)
uv sync
```

### 3. 安装开发依赖

```bash
# 安装包和开发工具
pip install -e ".[dev,docs]"

# 安装 pre-commit 钩子
pre-commit install
```

### 4. 验证安装

```bash
# 测试导入
python -c "from agraph import KnowledgeGraph; print('安装成功!')"

# 运行测试确保环境正常
pytest --version
black --version
pylint --version
```

## 开发流程

### 1. 创建功能分支

```bash
git checkout -b feature/your-feature-name
```

### 2. 进行开发

- 编写代码
- 添加必要的类型提示
- 编写文档字符串
- 添加单元测试

### 3. 提交代码

```bash
# 添加文件
git add .

# 提交（会自动运行 pre-commit 检查）
git commit -m "feat: add your feature description"

# 推送到远程
git push origin feature/your-feature-name
```

### 4. 创建 Pull Request

- 确保所有 CI 检查通过
- 添加详细的 PR 描述
- 请求代码审查

## 代码质量检查

我们使用多种工具确保代码质量。你可以在本地运行这些检查：

### 一键检查所有工具

```bash
# 使用提供的脚本
./scripts/check_code.sh

# 或使用 Makefile
make check
```

### 逐个工具检查

#### 1. 代码格式检查 (Black)

```bash
# 检查格式
black --check --diff  --line-length=120 agraph/ examples/

# 自动格式化
black --line-length=120 agraph/ examples/
```

#### 2. 导入排序 (isort)

```bash
# 检查导入排序
isort --check-only --diff agraph/ examples/

# 自动排序
isort --line-length=120 agraph/ examples/
```

#### 3. 语法检查 (Flake8)

```bash
# 基本语法检查
flake8 agraph/ examples/ --max-line-length=120 --extend-ignore=E203,W503

# 详细检查
flake8 agraph/ examples/ --max-line-length=120 --extend-ignore=E203,W503 --count --statistics
```

#### 4. 类型检查 (MyPy)

```bash
# 类型检查
mypy agraph/ --ignore-missing-imports --follow-imports=silent

# 详细报告
mypy agraph/ --ignore-missing-imports --show-error-codes --pretty
```

#### 5. 代码规范检查 (Pylint)

```bash
# 基本检查
pylint agraph/

# 生成详细报告
pylint agraph/ --output-outformat=text > pylint_report.txt

# 只显示错误和警告
pylint agraph/ --errors-only
```

#### 6. 安全检查 (Bandit)

```bash
# 安全扫描
bandit -r agraph/

# 生成 JSON 报告
bandit -r agraph/ -f json -o bandit_report.json

# 排除特定检查
bandit -r agraph/ -s B101,B601
```

### 修复常见问题

```bash
# 使用提供的脚本自动修复
./scripts/fix_code.sh

# 或使用 Makefile
make fix

# 手动修复格式问题
black --line-length=120 agraph/ examples/
isort --line-length=120 agraph/ examples/
```

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_graph.py

# 运行特定测试方法
pytest tests/test_graph.py::TestKnowledgeGraph::test_add_entity

# 详细输出
pytest -v

# 并行运行
pytest -n auto
```

### 测试覆盖率

```bash
# 生成覆盖率报告
pytest --cov=agraph --cov-report=html --cov-report=term

# 查看 HTML 报告
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
```

### 测试特定 Python 版本

```bash
# 使用 tox 测试多个 Python 版本
pip install tox
tox

# 测试特定版本
tox -e py311
```

## 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交类型

- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 重构代码
- `test`: 添加或修改测试
- `chore`: 维护任务（如依赖更新）

### 提交示例

```bash
git commit -m "feat: add LightRAG integration support"
git commit -m "fix: resolve entity deduplication issue"
git commit -m "docs: update installation guide"
git commit -m "test: add integration tests for Neo4j storage"
git commit -m "refactor: improve graph builder interface"
```

## 发布流程

### 版本管理

我们使用 [Semantic Versioning](https://semver.org/)：

- `MAJOR.MINOR.PATCH`
- `1.0.0` → `1.0.1` (补丁)
- `1.0.0` → `1.1.0` (小版本)
- `1.0.0` → `2.0.0` (大版本)

### 发布步骤

1. **更新版本号**
   ```bash
   # 在 pyproject.toml 中更新 version
   version = "1.1.0"
   ```

2. **更新 CHANGELOG**
   ```bash
   # 在 CHANGELOG.md 中添加版本信息
   ## [1.1.0] - 2024-01-15
   ### Added
   - 新功能描述
   ### Fixed
   - 修复的问题
   ```

3. **创建发布标签**
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0
   ```

4. **GitHub 自动发布**
   - GitHub Actions 会自动构建并发布到 PyPI
   - 检查 Actions 页面确保发布成功

## 工具配置

### IDE 配置

#### VS Code

创建 `.vscode/settings.json`：

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=120"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. 配置代码格式化器为 Black
2. 启用 Pylint 和 MyPy 检查
3. 配置 import 排序使用 isort

### Git 钩子

Pre-commit 已配置以下钩子：

- 代码格式检查
- 导入排序
- 语法检查
- 类型检查
- 安全扫描

```bash
# 手动运行所有钩子
pre-commit run --all-files

# 跳过钩子提交（不推荐）
git commit --no-verify -m "message"
```

## 常见问题

### 1. Pre-commit 钩子失败

```bash
# 重新安装钩子
pre-commit uninstall
pre-commit install

# 更新钩子版本
pre-commit autoupdate
```

### 2. MyPy 类型检查错误

```bash
# 忽略特定错误
# type: ignore

# 为第三方库安装类型包
pip install types-requests types-pyyaml
```

### 3. Pylint 评分过低

```bash
# 查看详细报告
pylint agraph/ --reports=y

# 禁用特定检查
# pylint: disable=too-many-arguments
```

### 4. 测试失败

```bash
# 重新安装依赖
pip install -e .[dev] --force-reinstall

# 清理缓存
pytest --cache-clear

# 检查测试环境
pytest --collect-only
```

### 5. 导入问题

```bash
# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 重新安装包
pip uninstall knowledge-graph-toolkit
pip install -e .
```

## 快捷命令

### 使用 Makefile

```bash
# 查看所有可用命令
make help

# 安装开发环境
make install-dev

# 代码质量检查
make check

# 自动修复格式
make fix

# 运行测试
make test

# 生成覆盖率报告
make test-cov

# 构建包
make build
```

## 相关资源

- [Python 编码规范 (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Pytest 文档](https://docs.pytest.org/)
- [Black 文档](https://black.readthedocs.io/)
- [MyPy 文档](https://mypy.readthedocs.io/)
- [Pylint 文档](https://pylint.pycqa.org/)

## 获得帮助

如果遇到问题：

1. 查看本文档的常见问题部分
2. 搜索现有的 [Issues](https://github.com/yourusername/knowledge-graph-toolkit/issues)
3. 创建新的 Issue 并提供详细信息
4. 加入我们的讨论 [Discussions](https://github.com/yourusername/knowledge-graph-toolkit/discussions)

感谢你的贡献！
