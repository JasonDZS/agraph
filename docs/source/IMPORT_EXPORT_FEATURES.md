# 知识图谱导入导出功能总览

## 功能概述

agraph 现已支持完整的知识图谱导入导出功能，提供 JSON 和 GraphML 两种格式，满足不同场景的需求。

## 🎯 核心功能

### 1. JSON 格式支持

- ✅ **完整数据保留**: 所有实体、关系、集群、文本块、元数据
- ✅ **文件导入导出**: `export_to_json()` / `import_from_json()`
- ✅ **字符串导入导出**: `export_to_json_string()` / `import_from_json_string()`
- ✅ **自动目录创建**: 导出时自动创建必要的目录结构
- ✅ **UTF-8 编码**: 完整支持中文等多语言字符
- ✅ **格式化选项**: 可定制缩进、排序等 JSON 格式选项

### 2. GraphML 格式支持

- ✅ **标准兼容**: 符合 W3C GraphML 标准
- ✅ **文件导入导出**: `export_to_graphml()` / `import_from_graphml()`
- ✅ **属性映射**: 智能处理复杂属性的序列化
- ✅ **权重处理**: 关系权重存储在 properties 中
- ✅ **类型容错**: 无效类型时自动回退到默认值
- ✅ **工具兼容**: 支持 Gephi、Cytoscape、NetworkX、igraph

### 3. 错误处理与验证

- ✅ **文件检查**: 自动检查文件存在性
- ✅ **数据验证**: 导入后自动验证数据完整性
- ✅ **异常处理**: 完整的错误处理机制
- ✅ **容错解析**: 属性解析失败时的优雅降级

## 🏗️ 架构设计

### ImportExportMixin 基类

```python
class ImportExportMixin(ABC):
    # JSON 格式支持
    def export_to_json(self, file_path, **kwargs)
    def import_from_json(cls, file_path, **kwargs)
    def export_to_json_string(self, **kwargs)
    def import_from_json_string(cls, json_string, **kwargs)

    # GraphML 格式接口（子类实现）
    def export_to_graphml(self, file_path, **kwargs)
    def import_from_graphml(cls, file_path, **kwargs)

    # 扩展接口
    def _export_data(self) -> Dict[str, Any]
    def _import_data(cls, data, **kwargs)
```

### KnowledgeGraph 集成

- 继承 `ImportExportMixin`，自动获得所有导入导出功能
- 实现 GraphML 的具体导入导出逻辑
- 利用现有的 `to_dict()` 和 `from_dict()` 方法

## 📊 格式比较

| 特性 | JSON | GraphML |
|------|------|---------|
| **数据完整性** | 100% | ~80% (无集群、文本块) |
| **文件大小** | 中等 | 较大 (XML 格式) |
| **解析速度** | 快 | 中等 |
| **工具支持** | agraph 原生 | 广泛的图形工具 |
| **可读性** | 好 | 好 (XML 结构) |
| **标准化** | JSON 标准 | W3C GraphML 标准 |

## 🔧 技术实现

### JSON 实现要点

- 使用 Python 标准 `json` 库
- 支持自定义序列化选项 (`indent`, `ensure_ascii`, `sort_keys`)
- 利用现有的 `SerializableMixin` 接口
- 自动处理时间戳的 ISO 格式转换

### GraphML 实现要点

- 集成 NetworkX 库进行 GraphML 处理
- MultiDiGraph 支持多重关系
- 属性序列化：复杂对象 → 字符串 → `ast.literal_eval()` 反序列化
- 图级元数据存储在 NetworkX 图属性中
- 智能类型处理和错误恢复

### 依赖管理

- 新增依赖：`networkx>=3.2`
- 向后兼容：不影响现有功能
- 可选依赖：GraphML 功能仅在需要时使用

## 🧪 测试覆盖

### 单元测试场景

- ✅ 基本导入导出功能
- ✅ 复杂数据结构处理
- ✅ 错误情况处理
- ✅ 文件不存在处理
- ✅ 无效数据格式处理
- ✅ 大型图谱处理
- ✅ 中文字符支持

### 集成测试场景

- ✅ 完整工作流测试
- ✅ 格式间转换测试
- ✅ 属性保留验证
- ✅ 性能基准测试

## 📚 文档体系

### 用户文档

1. **[快速开始](quick_start_import_export.md)** - 5分钟上手指南
2. **[完整教程](import_export_tutorial.md)** - 详细功能介绍和使用方法
3. **[GraphML 集成](graphml_integration_guide.md)** - 图形工具集成指南
4. **[功能概览](README_import_export.md)** - 功能总览和索引

### 技术文档

- API 文档自动生成
- 代码示例和最佳实践
- 性能优化建议
- 故障排除指南

## 🔗 工具集成

### 图形可视化工具

- **Gephi**: 交互式网络可视化
- **Cytoscape**: 生物信息学网络分析
- **NodeXL**: Excel 中的网络分析

### 编程库集成

- **NetworkX**: Python 网络分析库
- **igraph**: R/Python 图形分析库
- **graph-tool**: 高性能图形库

### 数据分析平台

- **Jupyter Notebook**: 交互式数据分析
- **R Studio**: 统计分析环境
- **MATLAB**: 科学计算平台

## 🚀 使用示例

### 基础用法

```python
# 创建和导出
kg = KnowledgeGraph(name="示例")
kg.export_to_json("graph.json")
kg.export_to_graphml("graph.graphml")

# 导入和使用
kg1 = KnowledgeGraph.import_from_json("graph.json")
kg2 = KnowledgeGraph.import_from_graphml("graph.graphml")
```

### 高级用法

```python
# 批量处理
for i, kg in enumerate(graphs):
    kg.export_to_json(f"output/graph_{i:03d}.json")

# 条件导出
filtered_kg = export_filtered_graph(
    kg, "filtered.json",
    entity_types=[EntityType.PERSON],
    min_confidence=0.8
)

# 工具集成
import networkx as nx
with tempfile.NamedTemporaryFile(suffix='.graphml') as tmp:
    kg.export_to_graphml(tmp.name)
    G = nx.read_graphml(tmp.name)
    centrality = nx.degree_centrality(G)
```

## ⚡ 性能特征

### 处理能力

- **小型图谱** (< 1K 节点): 毫秒级处理
- **中型图谱** (1K-10K 节点): 秒级处理
- **大型图谱** (10K+ 节点): 提供流式处理选项

### 内存使用

- JSON: 内存友好，支持流式处理
- GraphML: NetworkX 缓存，适中内存使用
- 大型图谱: 提供数据简化选项

## 🎯 应用场景

### 数据备份与迁移

```python
# 定期备份
kg.export_to_json(f"backup_{datetime.now().strftime('%Y%m%d')}.json")

# 系统迁移
old_kg = KnowledgeGraph.import_from_json("legacy_system.json")
new_system.import_graph(old_kg)
```

### 科学研究

```python
# 导出供 R 分析
kg.export_to_graphml("research_data.graphml")
# 在 R 中: library(igraph); g <- read_graph("research_data.graphml")
```

### 数据可视化

```python
# Gephi 可视化工作流
kg.export_to_graphml("network_viz.graphml")
# 1. 在 Gephi 中打开文件
# 2. 应用布局算法
# 3. 设置视觉属性
# 4. 导出高质量图像
```

## 🔮 未来计划

### 短期目标

- [ ] 支持更多 GraphML 属性类型
- [ ] 添加 GraphSON 格式支持
- [ ] 性能优化和内存使用改进

### 长期目标

- [ ] 支持 RDF/Turtle 格式
- [ ] 集成更多图形分析工具
- [ ] 云端图谱存储支持

## 📈 质量保证

### 代码质量

- ✅ 100% 类型注解覆盖
- ✅ Flake8 代码规范检查
- ✅ MyPy 静态类型检查
- ✅ Black 代码格式化
- ✅ 完整的错误处理

### 测试质量

- ✅ 单元测试覆盖
- ✅ 集成测试验证
- ✅ 性能基准测试
- ✅ 边界条件测试

## 💡 最佳实践

1. **格式选择**
   - 数据备份和系统集成：使用 JSON
   - 图形分析和可视化：使用 GraphML

2. **性能优化**
   - 大型图谱考虑数据分片
   - 使用流式处理减少内存占用
   - 定期清理临时文件

3. **错误处理**
   - 始终包含 try-catch 块
   - 验证导入后的数据完整性
   - 提供用户友好的错误信息

4. **数据管理**
   - 定期备份重要图谱数据
   - 使用版本控制管理图谱变更
   - 建立数据验证流程

---

**总结**: agraph 的导入导出功能为知识图谱的数据交换提供了完整的解决方案，支持多种格式，集成主流工具，具备良好的扩展性和可维护性。
