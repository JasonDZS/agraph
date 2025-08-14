# 导入导出快速开始

本指南帮助您快速掌握 agraph 的导入导出功能。

## 5 分钟快速上手

### 1. 创建知识图谱

```python
from agraph import KnowledgeGraph, Entity, Relation, EntityType, RelationType

# 创建知识图谱
kg = KnowledgeGraph(name="快速示例", description="导入导出演示")

# 添加实体
python = Entity(name="Python", entity_type=EntityType.CONCEPT, description="编程语言")
ai = Entity(name="AI", entity_type=EntityType.CONCEPT, description="人工智能")

kg.add_entity(python)
kg.add_entity(ai)

# 添加关系
relation = Relation(
    head_entity=python,
    tail_entity=ai,
    relation_type=RelationType.BELONGS_TO,
    description="Python 用于 AI 开发"
)
kg.add_relation(relation)
```

### 2. 导出数据

```python
# 导出为 JSON（完整数据）
kg.export_to_json("my_graph.json")

# 导出为 GraphML（图形分析）
kg.export_to_graphml("my_graph.graphml")

print("导出完成！")
```

### 3. 导入数据

```python
# 从 JSON 导入
kg_json = KnowledgeGraph.import_from_json("my_graph.json")

# 从 GraphML 导入
kg_graphml = KnowledgeGraph.import_from_graphml("my_graph.graphml")

print(f"JSON 导入: {len(kg_json.entities)} 个实体, {len(kg_json.relations)} 个关系")
print(f"GraphML 导入: {len(kg_graphml.entities)} 个实体, {len(kg_graphml.relations)} 个关系")
```

## 格式选择

| 需求 | 推荐格式 | 原因 |
|------|---------|------|
| 备份数据 | JSON | 保留完整信息 |
| 图形可视化 | GraphML | 兼容 Gephi、Cytoscape |
| 网络分析 | GraphML | 兼容 NetworkX、igraph |
| 系统集成 | JSON | 编程友好 |

## 常用代码片段

### 检查文件

```python
import os
from pathlib import Path

# 检查文件是否存在
if Path("graph.json").exists():
    kg = KnowledgeGraph.import_from_json("graph.json")
    print("导入成功")
else:
    print("文件不存在")
```

### 错误处理

```python
try:
    kg = KnowledgeGraph.import_from_json("graph.json")
    print("导入成功")
except FileNotFoundError:
    print("文件未找到")
except Exception as e:
    print(f"导入失败: {e}")
```

### 批量操作

```python
# 批量导出
graphs = [kg1, kg2, kg3]
for i, graph in enumerate(graphs):
    graph.export_to_json(f"graph_{i}.json")
```

## 下一步

- 阅读完整的 [导入导出教程](import_export_tutorial.md)
- 了解 [向量数据库集成](vectordb_tutorial.md)
- 查看 [API 文档](source/agraph.base.graph.rst)
