# Knowledge Graph Toolkit - Makefile
# 提供常用的开发命令快捷方式

.PHONY: help install install-dev test test-cov lint format check fix clean docs build publish

# 默认目标
help:
	@echo "Knowledge Graph Toolkit - 开发命令"
	@echo "=================================="
	@echo ""
	@echo "安装相关:"
	@echo "  install      - 安装包"
	@echo "  install-dev  - 安装开发依赖"
	@echo ""
	@echo "代码质量:"
	@echo "  check        - 运行所有代码质量检查"
	@echo "  fix          - 自动修复代码格式问题"
	@echo "  format       - 格式化代码 (black + isort)"
	@echo "  lint         - 运行 linting 工具"
	@echo ""
	@echo ""
	@echo "文档和发布:"
	@echo "  docs         - 构建文档"
	@echo "  build        - 构建包"
	@echo "  clean        - 清理构建文件"
	@echo "  publish      - 发布到 PyPI (需要权限)"

# 安装
install:
	pip install -e .

install-dev:
	pip install -e .[dev,docs]
	pre-commit install

# 代码质量检查
check:
	@echo "🔍 运行所有代码质量检查..."
	./scripts/check_code.sh

fix:
	@echo "🔧 自动修复代码格式问题..."
	./scripts/fix_code.sh

format:
	@echo "📝 格式化代码..."
	black agraph/ examples/
	isort agraph/ examples/

lint:
	@echo "📋 运行代码检查工具..."
	flake8 agraph/ examples/ --max-line-length=100
	pylint agraph/ --exit-zero
	mypy agraph/ --ignore-missing-imports

# 文档
docs:
	@echo "📖 构建文档..."
	sphinx-build -b html docs/source docs/build
	@echo "📖 查看文档: open docs/build/index.html"

# 构建和发布
build:
	@echo "🏗️ 构建包..."
	python -m build

clean:
	@echo "🧹 清理构建文件..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf docs/build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

publish: build
	@echo "🚀 发布到 PyPI..."
	@echo "⚠️  确保你有发布权限！"
	twine check dist/*
	twine upload dist/*

# 开发工具
pre-commit:
	pre-commit run --all-files

update-deps:
	@echo "📦 更新依赖..."
	pip-compile --upgrade pyproject.toml
	pre-commit autoupdate
