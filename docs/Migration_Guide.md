# 🔄 AGraph 迁移指南

## 概述

AGraph v0.2.0 引入了全新的统一架构和优化版本，提供了 **10-100x 的性能提升**。本指南帮助您从传统的 `KnowledgeGraph` 迁移到新的 `OptimizedKnowledgeGraph`。

## 🚨 弃用时间线

| 版本 | 状态 | 说明 |
|------|------|------|
| **v0.2.0** | 软弃用 | `KnowledgeGraph` 发出弃用警告，但仍可使用 |
| **v0.5.0** | 硬弃用 | `KnowledgeGraph` 不再推荐，文档移除 |
| **v1.0.0** | 完全移除 | `KnowledgeGraph` 彻底移除 |

## ✅ 为什么要迁移？

### 性能提升
- **实体查询**: 10-100x 性能提升 (O(n) → O(1))
- **关系查询**: 10-50x 性能提升 (索引化查询)
- **图统计**: 2-5x 性能提升 (缓存机制)
- **级联删除**: 显著性能提升 (索引化删除)

### 架构优势
- **统一架构**: Manager层解耦，DAO抽象
- **Result模式**: 统一的错误处理和验证
- **线程安全**: 全面的并发支持
- **扩展性**: 为未来功能扩展做准备

### 兼容性
- **100% API 兼容**: 无需修改任何代码
- **数据兼容**: 完全支持现有数据格式
- **功能兼容**: 所有功能保持一致

## 🔄 迁移方式

### 方式1: 零代码迁移 (推荐)

**最简单的迁移方式 - 只需要修改导入语句:**

```python
# 旧版本 (已弃用)
from agraph import KnowledgeGraph

kg = KnowledgeGraph(name="My Graph")

# 新版本 (推荐)
from agraph import OptimizedKnowledgeGraph

kg = OptimizedKnowledgeGraph(name="My Graph")
```

**就这么简单！** 其他代码完全不需要修改。

### 方式2: 类型别名迁移

如果您有大量代码使用 `KnowledgeGraph`，可以使用类型别名：

```python
from agraph import OptimizedKnowledgeGraph as KnowledgeGraph

# 现有代码无需任何修改
kg = KnowledgeGraph(name="My Graph")
kg.add_entity(entity)
# ... 其他代码保持不变
```

### 方式3: 渐进式迁移

对于大型项目，可以逐步迁移：

```python
# 第一步: 导入两个版本
from agraph import KnowledgeGraph, OptimizedKnowledgeGraph

# 第二步: 新功能使用优化版本
def create_new_graph():
    return OptimizedKnowledgeGraph(name="New Graph")

# 第三步: 逐步替换旧代码
def migrate_existing_function():
    # kg = KnowledgeGraph()  # 旧版本
    kg = OptimizedKnowledgeGraph()  # 新版本
    return kg
```

## 🔧 AGraph 生态系统自动升级

**好消息**: 如果您使用 AGraph 主类，**无需任何迁移**！

```python
# 这些代码已自动使用优化版本，无需修改
from agraph import AGraph

async with AGraph() as agraph:
    # 内部已自动使用 OptimizedKnowledgeGraph
    # 和统一架构，享受所有性能提升
    kg = await agraph.build_from_texts(texts)
```

## 📊 迁移验证

### 1. 功能验证

迁移后，运行这个简单测试确保一切正常：

```python
from agraph import OptimizedKnowledgeGraph, Entity, Relation

# 创建图谱
kg = OptimizedKnowledgeGraph(name="Test Graph")

# 添加实体
entity1 = Entity(name="Apple", entity_type="organization")
entity2 = Entity(name="iPhone", entity_type="product")
kg.add_entity(entity1)
kg.add_entity(entity2)

# 添加关系
relation = Relation(
    head_entity=entity1.entity_id,
    tail_entity=entity2.entity_id,
    relation_type="produces"
)
kg.add_relation(relation)

# 验证功能
assert len(kg.entities) == 2
assert len(kg.relations) == 1
print("✅ 迁移验证成功！")
```

### 2. 性能验证

对比性能提升：

```python
import time
from agraph import KnowledgeGraph, OptimizedKnowledgeGraph

def benchmark_graph_operations(GraphClass, name):
    start = time.time()
    kg = GraphClass(name=f"Benchmark {name}")

    # 添加大量实体
    for i in range(1000):
        entity = Entity(name=f"Entity_{i}", entity_type="person")
        kg.add_entity(entity)

    # 查询性能测试
    entities_by_type = kg.get_entities_by_type("person")

    end = time.time()
    print(f"{name}: {end - start:.4f} 秒, 找到 {len(entities_by_type)} 个实体")

# 对比测试
benchmark_graph_operations(KnowledgeGraph, "传统版本")
benchmark_graph_operations(OptimizedKnowledgeGraph, "优化版本")
```

## ⚠️ 注意事项

### 1. 弃用警告处理

迁移前，您可能会看到这样的警告：

```
FutureWarning: KnowledgeGraph is deprecated as of v0.2.0 and will be removed in v1.0.0.
Use OptimizedKnowledgeGraph for 10-100x performance improvements and unified architecture.
```

**解决方法**: 按照迁移指南替换即可。

### 2. 抑制警告 (不推荐)

如果暂时不想看到警告，可以临时抑制：

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="agraph")
```

**但强烈建议及时迁移而不是抑制警告。**

### 3. 测试环境迁移

在生产环境迁移前，请在测试环境验证：

1. 功能正确性
2. 性能提升效果
3. 内存使用情况
4. 并发安全性

## 🆘 获得帮助

### 常见问题

**Q: 迁移会影响现有数据吗？**
A: 不会。OptimizedKnowledgeGraph 完全兼容现有数据格式。

**Q: 性能提升有多明显？**
A: 查询操作通常有 10-100x 提升，具体取决于数据规模。

**Q: 需要修改多少代码？**
A: 最少只需要修改导入语句，其他代码可能完全不需要修改。

**Q: 如果迁移出现问题怎么办？**
A: OptimizedKnowledgeGraph 提供完全的向后兼容。如有问题，可以临时回退。

### 支持渠道

- **文档**: 查看完整的 API 文档
- **示例**: 参考 `examples/` 目录中的示例代码
- **问题反馈**: 创建 GitHub Issue

## 🎯 最佳实践

### 1. 立即开始迁移

建议立即开始迁移，享受性能提升：

```python
# ❌ 避免使用 (已弃用)
from agraph import KnowledgeGraph

# ✅ 推荐使用
from agraph import OptimizedKnowledgeGraph
```

### 2. 批量迁移脚本

对于大型项目，可以创建迁移脚本：

```bash
# 使用 sed 批量替换
find . -name "*.py" -exec sed -i 's/KnowledgeGraph/OptimizedKnowledgeGraph/g' {} +

# 更新导入语句
find . -name "*.py" -exec sed -i 's/from agraph import KnowledgeGraph/from agraph import OptimizedKnowledgeGraph/g' {} +
```

### 3. 代码审查

迁移完成后，建议进行代码审查：

- [ ] 确认所有 `KnowledgeGraph` 引用已更新
- [ ] 验证功能测试通过
- [ ] 确认性能基准测试显示提升
- [ ] 检查日志中无弃用警告

## 📈 迁移后的收益

### 立即收益
- ✅ **性能大幅提升**: 10-100x 查询性能提升
- ✅ **更好的并发支持**: 线程安全的操作
- ✅ **统一的错误处理**: Result-based API
- ✅ **实时性能监控**: 内置指标统计

### 长期收益
- 🔮 **未来功能支持**: 新功能将优先支持优化版本
- 🔮 **生态系统集成**: 与其他工具更好的集成
- 🔮 **社区支持**: 活跃的开发和维护
- 🔮 **扩展性**: 支持更大规模的知识图谱

---

## 🎉 总结

迁移到 OptimizedKnowledgeGraph 是一个**零风险、高收益**的决定：

- **零迁移成本**: 最少只需修改导入语句
- **显著性能提升**: 10-100x 的性能改善
- **完全兼容**: 现有代码和数据无需修改
- **未来保证**: 持续的功能更新和支持

**立即开始您的迁移之旅，享受 AGraph 统一架构带来的强大性能提升！** 🚀
