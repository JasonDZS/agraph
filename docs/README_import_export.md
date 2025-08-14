# 知识图谱导入导出功能

agraph 提供了强大的导入导出功能，支持多种格式，方便与其他系统和工具进行数据交换。

## 支持的格式

### JSON 格式

- ✅ **完整数据保留**: 保存所有实体、关系、集群、文本块和元数据
- ✅ **原生支持**: agraph 的标准格式
- ✅ **快速处理**: 高效的序列化和反序列化
- ✅ **编程友好**: 易于在代码中处理

**适用场景**: 数据备份、系统集成、配置文件

### GraphML 格式

- ✅ **标准化**: 基于 W3C GraphML 标准
- ✅ **工具兼容**: 支持 Gephi、Cytoscape、NetworkX、igraph
- ✅ **可视化**: 适合图形可视化和网络分析
- ⚠️ **部分数据**: 主要保存实体和关系，集群和文本块信息会丢失

**适用场景**: 图形可视化、网络分析、学术研究

## 快速开始

```python
from agraph import KnowledgeGraph, Entity, Relation, EntityType, RelationType

# 创建知识图谱
kg = KnowledgeGraph(name="示例图谱")

# 添加数据
entity1 = Entity(name="Python", entity_type=EntityType.CONCEPT)
entity2 = Entity(name="AI", entity_type=EntityType.CONCEPT)
kg.add_entity(entity1)
kg.add_entity(entity2)

relation = Relation(
    head_entity=entity1,
    tail_entity=entity2,
    relation_type=RelationType.BELONGS_TO
)
kg.add_relation(relation)

# 导出
kg.export_to_json("graph.json")        # JSON 格式
kg.export_to_graphml("graph.graphml")  # GraphML 格式

# 导入
kg_json = KnowledgeGraph.import_from_json("graph.json")
kg_graphml = KnowledgeGraph.import_from_graphml("graph.graphml")
```

## 文档指南

| 文档 | 描述 | 适合人群 |
|------|------|----------|
| [快速开始](quick_start_import_export.md) | 5分钟上手指南 | 初学者 |
| [完整教程](import_export_tutorial.md) | 详细的导入导出教程 | 所有用户 |
| [GraphML 集成](graphml_integration_guide.md) | 图形工具集成指南 | 数据分析师 |

## 主要功能

### 1. 文件导入导出

```python
# 导出到文件
kg.export_to_json("my_graph.json")
kg.export_to_graphml("my_graph.graphml")

# 从文件导入
kg = KnowledgeGraph.import_from_json("my_graph.json")
kg = KnowledgeGraph.import_from_graphml("my_graph.graphml")
```

### 2. 字符串导入导出

```python
# 导出为字符串
json_string = kg.export_to_json_string()

# 从字符串导入
kg = KnowledgeGraph.import_from_json_string(json_string)
```

### 3. 批量处理

```python
# 批量导出多个图谱
graphs = [kg1, kg2, kg3]
for i, graph in enumerate(graphs):
    graph.export_to_json(f"graph_{i}.json")
```

### 4. 错误处理

```python
try:
    kg = KnowledgeGraph.import_from_json("graph.json")
except FileNotFoundError:
    print("文件不存在")
except Exception as e:
    print(f"导入失败: {e}")
```

## 工具集成

### Gephi (图形可视化)

```python
# 为 Gephi 优化输出
kg.export_to_graphml("gephi_network.graphml")
# 在 Gephi 中: 文件 -> 打开 -> 选择 .graphml 文件
```

### NetworkX (Python 网络分析)

```python
import networkx as nx
import tempfile

# 转换为 NetworkX 图
with tempfile.NamedTemporaryFile(suffix='.graphml') as tmp:
    kg.export_to_graphml(tmp.name)
    G = nx.read_graphml(tmp.name)

# 进行网络分析
centrality = nx.degree_centrality(G)
communities = nx.community.greedy_modularity_communities(G.to_undirected())
```

### Cytoscape (生物网络分析)

```python
# 为 Cytoscape 准备数据
kg.export_to_graphml("cytoscape_network.graphml")
# 在 Cytoscape 中: File -> Import -> Network from File
```

## 性能特点

| 操作 | JSON | GraphML |
|------|------|---------|
| 导出速度 | 快 | 中等 |
| 导入速度 | 快 | 中等 |
| 文件大小 | 中等 | 较大 |
| 数据完整性 | 100% | ~80% |
| 工具兼容性 | agraph | 广泛 |

## 最佳实践

1. **备份数据**: 使用 JSON 格式保存完整数据
2. **可视化分析**: 使用 GraphML 格式与图形工具集成
3. **大型图谱**: 考虑数据简化和分批处理
4. **错误处理**: 始终包含异常处理代码
5. **验证数据**: 导入后验证数据完整性

## 示例项目

查看 `examples/` 目录中的完整示例：

- `json_export_example.py`: JSON 导入导出示例
- `graphml_analysis_example.py`: GraphML 图形分析示例
- `batch_processing_example.py`: 批量处理示例

## 技术支持

- 📚 [完整文档](import_export_tutorial.md)
- 🐛 [问题报告](https://github.com/your-org/agraph/issues)
- 💬 [讨论区](https://github.com/your-org/agraph/discussions)

---

*最后更新: 2024-01-01*
