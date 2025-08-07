#!/bin/bash

# Knowledge Graph Toolkit - 代码质量检查脚本
# 本脚本会运行所有代码质量检查工具

set -e  # 遇到错误时退出

echo "🔍 开始运行代码质量检查..."
echo "================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查函数
run_check() {
    local name=$1
    local command=$2
    local emoji=$3

    echo -e "\n${emoji} ${BLUE}${name}...${NC}"
    echo "----------------------------------------"

    if eval $command; then
        echo -e "${GREEN}✅ ${name} 通过${NC}"
        return 0
    else
        echo -e "${RED}❌ ${name} 失败${NC}"
        return 1
    fi
}

# 初始化计数器
total_checks=0
passed_checks=0

# 1. 代码格式检查 (Black)
((total_checks++))
if run_check "代码格式检查 (Black)" "black --check --diff --line-length=120 agraph/ examples/" "📝"; then
    ((passed_checks++))
fi

# 2. 导入排序检查 (isort)
((total_checks++))
if run_check "导入排序检查 (isort)" "isort --check-only --diff agraph/ examples/" "🧹"; then
    ((passed_checks++))
fi

# 3. 语法检查 (Flake8)
((total_checks++))
if run_check "语法检查 (Flake8)" "flake8 agraph/ examples/ --max-line-length=120 --extend-ignore=E203,W503,E501 --count --statistics" "🔧"; then
    ((passed_checks++))
fi

# 4. 类型检查 (MyPy)
((total_checks++))
if run_check "类型检查 (MyPy)" "mypy agraph/ --ignore-missing-imports --follow-imports=silent --allow-untyped-defs" "🎯"; then
    ((passed_checks++))
fi

# 5. 代码规范检查 (Pylint)
((total_checks++))
if run_check "代码规范检查 (Pylint)" "pylint agraph/ --exit-zero --rcfile=.pylintrc" "📋"; then
    ((passed_checks++))
fi

# 6. 安全检查 (Bandit)
((total_checks++))
if run_check "安全检查 (Bandit)" "bandit -r agraph/ -ll --exit-zero" "🔒"; then
    ((passed_checks++))
fi

# 7. 运行测试
if command -v pytest &> /dev/null; then
    ((total_checks++))
    if run_check "单元测试 (Pytest)" "pytest --tb=short -q" "🧪"; then
        ((passed_checks++))
    fi
fi

# 显示总结
echo -e "\n================================"
echo -e "📊 ${BLUE}检查结果总结${NC}"
echo "================================"

if [ $passed_checks -eq $total_checks ]; then
    echo -e "${GREEN}🎉 所有检查都通过了！($passed_checks/$total_checks)${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  有 $((total_checks - passed_checks)) 个检查失败 ($passed_checks/$total_checks)${NC}"
    echo -e "\n${YELLOW}💡 修复建议：${NC}"
    echo "  - 运行 'black agraph/ examples/' 修复格式问题"
    echo "  - 运行 'isort agraph/ examples/' 修复导入顺序"
    echo "  - 查看上面的错误信息了解具体问题"
    exit 1
fi
