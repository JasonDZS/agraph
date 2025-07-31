#!/bin/bash

# Knowledge Graph Toolkit - 代码自动修复脚本
# 本脚本会自动修复常见的代码格式问题

set -e  # 遇到错误时退出

echo "🔧 开始自动修复代码格式问题..."
echo "================================"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. 代码格式化 (Black)
echo -e "\n📝 ${BLUE}使用 Black 格式化代码...${NC}"
black knowledge_graph/ examples/
echo -e "${GREEN}✅ Black 格式化完成${NC}"

# 2. 导入排序 (isort)
echo -e "\n🧹 ${BLUE}使用 isort 排序导入...${NC}"
isort knowledge_graph/ examples/
echo -e "${GREEN}✅ 导入排序完成${NC}"

# 3. 自动修复简单的 PEP 8 问题 (autopep8)
if command -v autopep8 &> /dev/null; then
    echo -e "\n🔧 ${BLUE}使用 autopep8 修复 PEP 8 问题...${NC}"
    autopep8 --in-place --aggressive --aggressive --recursive knowledge_graph/
    autopep8 --in-place --aggressive --aggressive --recursive examples/
    echo -e "${GREEN}✅ autopep8 修复完成${NC}"
else
    echo -e "\n💡 提示: 安装 autopep8 可以自动修复更多格式问题"
    echo "   pip install autopep8"
fi

echo -e "\n================================"
echo -e "${GREEN}🎉 代码自动修复完成！${NC}"
echo "================================"

echo -e "\n💡 ${BLUE}下一步建议：${NC}"
echo "  1. 运行 './scripts/check_code.sh' 验证修复结果"
echo "  2. 检查 git diff 查看具体修改"
echo "  3. 运行测试确保功能正常: pytest"
echo "  4. 提交修改: git add . && git commit -m 'style: fix code formatting'"