.PHONY: help install install-dev test test-cov lint format check fix clean docs docs-init build publish

# Default target
help:
	@echo "Knowledge Graph Toolkit - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install package"
	@echo "  install-dev  - Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  test         - Run unit tests"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  check        - Run all code quality checks"
	@echo "  fix          - Auto-fix code formatting issues"
	@echo "  format       - Format code (black + isort + trailing whitespace removal)"
	@echo "  lint         - Run linting tools"
	@echo ""
	@echo "Documentation & Publishing:"
	@echo "  docs         - Auto-generate and build documentation from code"
	@echo "  docs-init    - Initialize Sphinx documentation structure"
	@echo "  docs-watch   - Start development server with auto-reload"
	@echo "  build        - Build package"
	@echo "  clean        - Clean build files"
	@echo "  publish      - Publish to PyPI (requires permission)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev,docs]
	pre-commit install

# Testing
test:
	@echo "🧪 Running unit tests..."
	python -m unittest discover tests -v

test-cov:
	@echo "🧪 Running tests with coverage report..."
	coverage run -m unittest discover tests
	coverage report
	coverage html
	@echo "📊 Coverage report: open htmlcov/index.html"

# Code quality checks
check: test lint
	@echo "✅ All code quality checks completed"

fix:
	@echo "🔧 Auto-fixing code formatting issues..."
	./scripts/fix_code.sh

format:
	@echo "📝 Formatting code..."
	@echo "🔧 Removing trailing whitespace..."
	find agraph/ tests/ -name "*.py" -type f -exec sed -i '' 's/[[:space:]]*$$//' {} +
	find . -name "*.md" -type f -exec sed -i '' 's/[[:space:]]*$$//' {} +
	find . -name "*.yaml" -name "*.yml" -type f -exec sed -i '' 's/[[:space:]]*$$//' {} +
	black agraph/
	isort agraph/ --line-length=100

lint:
	@echo "📋 Running code linting tools..."
	flake8 agraph/ --ignore=E501,E203,W503
	pylint agraph/ --rcfile=.pylintrc --exit-zero
	mypy agraph/ --ignore-missing-imports

# Documentation
docs:
	@echo "📖 Setting up Sphinx documentation structure..."
	@if [ ! -d "docs/source" ]; then \
		echo "Creating Sphinx documentation structure..."; \
		sphinx-quickstart docs --quiet --project="Knowledge Graph Toolkit" --author="agraph-dev" --release="0.1.0" --language="en" --makefile --no-batchfile --sep; \
		echo "Sphinx documentation structure created."; \
	fi
	@echo "📖 Cleaning old auto-generated files..."
	@find docs/source -name "agraph*.rst" -not -name "agraph.rst" -delete 2>/dev/null || true
	@rm -f docs/source/modules.rst 2>/dev/null || true
	@echo "📖 Auto-generating API documentation from source code..."
	sphinx-apidoc -o docs/source --separate --module-first --force agraph/
	@echo "📖 Building documentation..."
	sphinx-build -b html docs/source docs/build
	@echo "📖 View documentation: open docs/build/index.html"

docs-init:
	@echo "📖 Initializing Sphinx documentation..."
	sphinx-quickstart docs --quiet --project="Knowledge Graph Toolkit" --author="agraph-dev" --release="0.1.0" --language="en" --makefile --no-batchfile --sep
	@echo "📖 Sphinx documentation initialized in docs/"

docs-watch:
	@echo "📖 Starting documentation development server with auto-reload..."
	@echo "📖 Watching for changes in agraph/ and docs/source/"
	@echo "📖 Press Ctrl+C to stop..."
	@while true; do \
		echo "📖 Building documentation..."; \
		make docs > /dev/null 2>&1; \
		echo "📖 Documentation updated at $(date)"; \
		echo "📖 Sleeping for 3 seconds..."; \
		sleep 3; \
	done

# Build and publish
build:
	@echo "🏗️ Building package..."
	python -m build

clean:
	@echo "🧹 Cleaning build files..."
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
	@echo "🚀 Publishing to PyPI..."
	@echo "⚠️  Make sure you have publishing permissions!"
	twine check dist/*
	twine upload dist/*

# Development tools
pre-commit:
	pre-commit run --all-files

update-deps:
	@echo "📦 Updating dependencies..."
	pip-compile --upgrade pyproject.toml
	pre-commit autoupdate
