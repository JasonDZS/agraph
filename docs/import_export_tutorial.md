# 知识图谱导入导出教程

本教程将指导您如何使用 agraph 的导入导出功能，支持 JSON 和 GraphML 格式，方便与其他系统和工具进行数据交换。

## 目录

1. [快速开始](#快速开始)
2. [JSON 格式导入导出](#json-格式导入导出)
3. [GraphML 格式导入导出](#graphml-格式导入导出)
4. [格式选择指南](#格式选择指南)
5. [高级用法](#高级用法)
6. [错误处理](#错误处理)
7. [性能优化](#性能优化)
8. [与其他工具集成](#与其他工具集成)

## 快速开始

### 基本导入导出示例

```python
from agraph import KnowledgeGraph, Entity, Relation, EntityType, RelationType

# 创建知识图谱
kg = KnowledgeGraph(
    name="我的知识图谱",
    description="一个简单的知识图谱示例"
)

# 添加实体
python_entity = Entity(
    name="Python",
    entity_type=EntityType.CONCEPT,
    description="一种高级编程语言",
    properties={"version": "3.11", "paradigm": "multi-paradigm"}
)

ai_entity = Entity(
    name="人工智能",
    entity_type=EntityType.CONCEPT,
    description="机器智能的研究领域"
)

kg.add_entity(python_entity)
kg.add_entity(ai_entity)

# 添加关系
relation = Relation(
    head_entity=python_entity,
    tail_entity=ai_entity,
    relation_type=RelationType.BELONGS_TO,
    description="Python 被广泛用于人工智能开发",
    properties={"usage": "high", "weight": 0.9}
)
kg.add_relation(relation)

# 导出到 JSON
kg.export_to_json("my_graph.json")

# 导出到 GraphML
kg.export_to_graphml("my_graph.graphml")

# 从文件导入
kg_from_json = KnowledgeGraph.import_from_json("my_graph.json")
kg_from_graphml = KnowledgeGraph.import_from_graphml("my_graph.graphml")
```

## JSON 格式导入导出

JSON 格式是 agraph 的原生格式，能够完整保存所有数据，包括实体、关系、集群、文本块和元数据。

### JSON 导出

#### 导出到文件

```python
# 基本导出
kg.export_to_json("graph.json")

# 自定义 JSON 格式选项
kg.export_to_json(
    "graph_formatted.json",
    indent=4,           # 缩进大小
    ensure_ascii=False, # 支持中文字符
    sort_keys=True      # 排序键值
)

# 导出到指定目录（自动创建目录）
kg.export_to_json("output/graphs/my_graph.json")
```

#### 导出为字符串

```python
# 导出为 JSON 字符串
json_string = kg.export_to_json_string()

# 紧凑格式（无缩进）
compact_json = kg.export_to_json_string(indent=None, separators=(',', ':'))
```

### JSON 导入

#### 从文件导入

```python
# 基本导入
kg = KnowledgeGraph.import_from_json("graph.json")

# 验证导入的数据
print(f"实体数量: {len(kg.entities)}")
print(f"关系数量: {len(kg.relations)}")
print(f"集群数量: {len(kg.clusters)}")
print(f"文本块数量: {len(kg.text_chunks)}")
```

#### 从字符串导入

```python
# 从 JSON 字符串导入
json_data = '''
{
    "id": "test-graph",
    "name": "测试图谱",
    "entities": {...},
    "relations": {...}
}
'''
kg = KnowledgeGraph.import_from_json_string(json_data)
```

### JSON 格式结构

```json
{
    "id": "图谱唯一标识符",
    "name": "图谱名称",
    "description": "图谱描述",
    "entities": {
        "entity_id": {
            "id": "实体ID",
            "name": "实体名称",
            "entity_type": "实体类型",
            "description": "实体描述",
            "confidence": 0.8,
            "properties": {},
            "text_chunks": [],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    },
    "relations": {
        "relation_id": {
            "id": "关系ID",
            "head_entity_id": "头实体ID",
            "tail_entity_id": "尾实体ID",
            "relation_type": "关系类型",
            "description": "关系描述",
            "confidence": 0.8,
            "properties": {},
            "text_chunks": [],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    },
    "clusters": {},
    "text_chunks": {},
    "metadata": {},
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
}
```

## GraphML 格式导入导出

GraphML 是一种标准的图形描述语言，基于 XML 格式，广泛用于图形分析和可视化工具。

### GraphML 导出

```python
# 基本导出
kg.export_to_graphml("graph.graphml")

# 导出到指定目录
kg.export_to_graphml("exports/network_analysis.graphml")
```

### GraphML 导入

```python
# 从 GraphML 文件导入
kg = KnowledgeGraph.import_from_graphml("graph.graphml")

# 检查导入结果
stats = kg.get_graph_statistics()
print(f"图谱统计: {stats}")
```

### GraphML 格式特点

#### 优势

- **标准化**: 基于 W3C 标准，兼容性好
- **工具支持**: 支持 Gephi、Cytoscape、igraph 等工具
- **可视化**: 便于图形可视化和网络分析
- **XML 结构**: 人类可读，便于调试

#### 限制

- **数据保留**: 不完全保留集群和文本块信息
- **属性序列化**: 复杂属性被转换为字符串
- **文件大小**: XML 格式相对冗长

### GraphML 数据映射

| agraph 概念 | GraphML 表示 | 说明 |
|------------|------------|------|
| Entity | Node | 实体作为图的节点 |
| Relation | Edge | 关系作为图的边 |
| Properties | Attributes | 属性作为节点/边属性 |
| Weight | Edge Attribute | 权重存储在关系的 properties 中 |
| Cluster | (不支持) | 当前版本不保留集群信息 |
| TextChunk | (不支持) | 当前版本不保留文本块信息 |

## 格式选择指南

### 选择 JSON 格式的场景

```python
# 1. 完整数据保存（推荐用于备份）
kg.export_to_json("backup/full_graph.json")

# 2. 与 agraph 系统间传输
data = kg.export_to_json_string()
# 通过 API 传输...

# 3. 配置文件和模板
template_kg = KnowledgeGraph(name="模板")
template_kg.export_to_json("templates/base_template.json")
```

### 选择 GraphML 格式的场景

```python
# 1. 图形可视化分析
kg.export_to_graphml("analysis/network.graphml")
# 然后在 Gephi 中打开进行可视化

# 2. 网络分析
kg.export_to_graphml("research/social_network.graphml")
# 使用 NetworkX 或 igraph 进行分析

# 3. 与其他图形工具交换数据
kg.export_to_graphml("export/for_cytoscape.graphml")
```

### 格式比较

| 特性 | JSON | GraphML |
|------|------|---------|
| 数据完整性 | ✅ 完整 | ⚠️ 部分（无集群、文本块） |
| 文件大小 | 中等 | 较大 |
| 工具支持 | agraph 原生 | 广泛的图形工具 |
| 可读性 | 好 | 好（XML） |
| 解析速度 | 快 | 中等 |
| 标准化 | JSON 标准 | GraphML 标准 |

## 高级用法

### 批量导入导出

```python
import os
from pathlib import Path

def batch_export_graphs(graphs, output_dir, format_type="json"):
    """批量导出多个知识图谱"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, kg in enumerate(graphs):
        if format_type == "json":
            file_path = output_dir / f"graph_{i:03d}.json"
            kg.export_to_json(file_path)
        elif format_type == "graphml":
            file_path = output_dir / f"graph_{i:03d}.graphml"
            kg.export_to_graphml(file_path)

    print(f"导出了 {len(graphs)} 个图谱到 {output_dir}")

def batch_import_graphs(input_dir, format_type="json"):
    """批量导入多个知识图谱"""
    input_dir = Path(input_dir)
    graphs = []

    if format_type == "json":
        pattern = "*.json"
        import_func = KnowledgeGraph.import_from_json
    else:
        pattern = "*.graphml"
        import_func = KnowledgeGraph.import_from_graphml

    for file_path in input_dir.glob(pattern):
        try:
            kg = import_func(file_path)
            graphs.append(kg)
            print(f"导入成功: {file_path.name}")
        except Exception as e:
            print(f"导入失败: {file_path.name}, 错误: {e}")

    return graphs

# 使用示例
graphs = [kg1, kg2, kg3]  # 多个知识图谱
batch_export_graphs(graphs, "output/batch", "json")
imported_graphs = batch_import_graphs("output/batch", "json")
```

### 条件导出

```python
def export_filtered_graph(kg, output_file, entity_types=None, min_confidence=0.0):
    """导出满足条件的子图"""
    # 创建新图谱
    filtered_kg = KnowledgeGraph(
        name=f"{kg.name}_filtered",
        description=f"过滤后的图谱: {kg.description}"
    )

    # 过滤实体
    for entity in kg.entities.values():
        if entity_types and entity.entity_type not in entity_types:
            continue
        if entity.confidence < min_confidence:
            continue
        filtered_kg.add_entity(entity)

    # 过滤关系
    for relation in kg.relations.values():
        if relation.head_entity.id not in filtered_kg.entities:
            continue
        if relation.tail_entity.id not in filtered_kg.entities:
            continue
        if relation.confidence < min_confidence:
            continue
        filtered_kg.add_relation(relation)

    # 导出过滤后的图谱
    filtered_kg.export_to_json(output_file)
    return filtered_kg

# 使用示例
filtered = export_filtered_graph(
    kg,
    "filtered_graph.json",
    entity_types=[EntityType.PERSON, EntityType.ORGANIZATION],
    min_confidence=0.7
)
```

### 增量导入导出

```python
def export_incremental(kg, base_file, incremental_file, timestamp):
    """导出增量数据"""
    incremental_kg = KnowledgeGraph(
        name=f"{kg.name}_incremental",
        description=f"增量数据，基于 {timestamp}"
    )

    # 导出在指定时间戳之后更新的数据
    for entity in kg.entities.values():
        if entity.updated_at > timestamp:
            incremental_kg.add_entity(entity)

    for relation in kg.relations.values():
        if relation.updated_at > timestamp:
            incremental_kg.add_relation(relation)

    incremental_kg.export_to_json(incremental_file)
    return incremental_kg

def merge_incremental(base_file, incremental_file, output_file):
    """合并增量数据"""
    base_kg = KnowledgeGraph.import_from_json(base_file)
    incremental_kg = KnowledgeGraph.import_from_json(incremental_file)

    # 合并数据
    base_kg.merge(incremental_kg)

    # 导出合并后的图谱
    base_kg.export_to_json(output_file)
    return base_kg
```

## 错误处理

### 常见错误和解决方案

```python
import logging
from pathlib import Path

def safe_export(kg, file_path, format_type="json"):
    """安全导出，包含错误处理"""
    try:
        file_path = Path(file_path)

        # 检查目录是否存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 验证图谱
        if not kg.is_valid():
            logging.warning("图谱验证失败，但仍尝试导出")

        # 执行导出
        if format_type == "json":
            kg.export_to_json(file_path)
        elif format_type == "graphml":
            kg.export_to_graphml(file_path)
        else:
            raise ValueError(f"不支持的格式: {format_type}")

        logging.info(f"导出成功: {file_path}")
        return True

    except Exception as e:
        logging.error(f"导出失败: {e}")
        return False

def safe_import(file_path, format_type="json"):
    """安全导入，包含错误处理"""
    try:
        file_path = Path(file_path)

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查文件大小
        if file_path.stat().st_size == 0:
            raise ValueError(f"文件为空: {file_path}")

        # 执行导入
        if format_type == "json":
            kg = KnowledgeGraph.import_from_json(file_path)
        elif format_type == "graphml":
            kg = KnowledgeGraph.import_from_graphml(file_path)
        else:
            raise ValueError(f"不支持的格式: {format_type}")

        # 验证导入的数据
        errors = kg.validate_integrity()
        if errors:
            logging.warning(f"数据完整性问题: {errors}")

        logging.info(f"导入成功: {file_path}")
        return kg

    except Exception as e:
        logging.error(f"导入失败: {e}")
        return None

# 使用示例
if safe_export(kg, "output/graph.json", "json"):
    imported_kg = safe_import("output/graph.json", "json")
    if imported_kg:
        print("导入导出成功完成")
```

### 数据验证

```python
def validate_before_export(kg):
    """导出前验证数据"""
    print("=== 数据验证报告 ===")

    # 基本统计
    stats = kg.get_graph_statistics()
    print(f"实体数量: {stats['total_entities']}")
    print(f"关系数量: {stats['total_relations']}")
    print(f"集群数量: {stats['total_clusters']}")
    print(f"文本块数量: {stats['total_text_chunks']}")

    # 完整性检查
    errors = kg.validate_integrity()
    if errors:
        print(f"\\n⚠️ 发现 {len(errors)} 个完整性问题:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\\n✅ 数据完整性检查通过")

    # 有效性检查
    invalid_entities = [e for e in kg.entities.values() if not e.is_valid()]
    invalid_relations = [r for r in kg.relations.values() if not r.is_valid()]

    if invalid_entities:
        print(f"\\n⚠️ 发现 {len(invalid_entities)} 个无效实体")
    if invalid_relations:
        print(f"\\n⚠️ 发现 {len(invalid_relations)} 个无效关系")

    return len(errors) == 0 and len(invalid_entities) == 0 and len(invalid_relations) == 0
```

## 性能优化

### 大规模数据导入导出

```python
import json
import time
from typing import Generator

def stream_export_large_graph(kg, file_path, chunk_size=1000):
    """流式导出大型图谱"""
    start_time = time.time()

    with open(file_path, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write('{\\n')
        f.write(f'  "id": "{kg.id}",\\n')
        f.write(f'  "name": "{kg.name}",\\n')
        f.write(f'  "description": "{kg.description}",\\n')

        # 分块写入实体
        f.write('  "entities": {\\n')
        entity_items = list(kg.entities.items())
        for i in range(0, len(entity_items), chunk_size):
            chunk = entity_items[i:i+chunk_size]
            for j, (eid, entity) in enumerate(chunk):
                entity_json = json.dumps(entity.to_dict(), ensure_ascii=False)
                comma = ',' if i + j < len(entity_items) - 1 else ''
                f.write(f'    "{eid}": {entity_json}{comma}\\n')

            # 显示进度
            progress = min(i + chunk_size, len(entity_items))
            print(f"实体导出进度: {progress}/{len(entity_items)}")

        f.write('  },\\n')

        # 分块写入关系
        f.write('  "relations": {\\n')
        relation_items = list(kg.relations.items())
        for i in range(0, len(relation_items), chunk_size):
            chunk = relation_items[i:i+chunk_size]
            for j, (rid, relation) in enumerate(chunk):
                relation_json = json.dumps(relation.to_dict(), ensure_ascii=False)
                comma = ',' if i + j < len(relation_items) - 1 else ''
                f.write(f'    "{rid}": {relation_json}{comma}\\n')

            progress = min(i + chunk_size, len(relation_items))
            print(f"关系导出进度: {progress}/{len(relation_items)}")

        f.write('  },\\n')

        # 写入其他字段
        f.write('  "clusters": {},\\n')
        f.write('  "text_chunks": {},\\n')
        f.write(f'  "metadata": {json.dumps(kg.metadata)},\\n')
        f.write(f'  "created_at": "{kg.created_at.isoformat()}",\\n')
        f.write(f'  "updated_at": "{kg.updated_at.isoformat()}"\\n')
        f.write('}')

    elapsed = time.time() - start_time
    print(f"导出完成，耗时: {elapsed:.2f} 秒")

def memory_efficient_import(file_path):
    """内存高效的导入方法"""
    print("开始内存高效导入...")

    # 这里可以实现流式解析 JSON
    # 或者分批处理大文件
    kg = KnowledgeGraph.import_from_json(file_path)

    print(f"导入完成，包含 {len(kg.entities)} 个实体，{len(kg.relations)} 个关系")
    return kg
```

### 并行处理

```python
import concurrent.futures
from pathlib import Path

def parallel_export_formats(kg, base_path):
    """并行导出多种格式"""
    def export_json():
        kg.export_to_json(f"{base_path}.json")
        return "JSON"

    def export_graphml():
        kg.export_to_graphml(f"{base_path}.graphml")
        return "GraphML"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(export_json),
            executor.submit(export_graphml)
        ]

        for future in concurrent.futures.as_completed(futures):
            format_name = future.result()
            print(f"{format_name} 导出完成")

# 使用示例
parallel_export_formats(kg, "output/my_graph")
```

## 与其他工具集成

### Gephi 集成

```python
def prepare_for_gephi(kg, output_file):
    """为 Gephi 准备 GraphML 文件"""
    # Gephi 偏好的节点和边属性
    gephi_kg = KnowledgeGraph(
        name=f"{kg.name}_for_gephi",
        description="为 Gephi 优化的版本"
    )

    # 添加实体，设置 Gephi 友好的属性
    for entity in kg.entities.values():
        entity.properties.update({
            "label": entity.name,  # Gephi 显示标签
            "type": str(entity.entity_type),
            "size": len(entity.description),  # 节点大小
        })
        gephi_kg.add_entity(entity)

    # 添加关系，设置权重
    for relation in kg.relations.values():
        relation.properties.update({
            "label": relation.description[:50],  # 边标签
            "weight": relation.properties.get("weight", 1.0),
        })
        gephi_kg.add_relation(relation)

    gephi_kg.export_to_graphml(output_file)
    print(f"Gephi 文件已生成: {output_file}")
    print("在 Gephi 中打开此文件进行网络可视化分析")

# 使用示例
prepare_for_gephi(kg, "gephi/network.graphml")
```

### NetworkX 集成

```python
import networkx as nx
import matplotlib.pyplot as plt

def to_networkx(kg):
    """转换为 NetworkX 图对象"""
    # 导出为 GraphML 字符串
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as tmp:
        kg.export_to_graphml(tmp.name)
        G = nx.read_graphml(tmp.name)

    return G

def analyze_with_networkx(kg):
    """使用 NetworkX 进行图分析"""
    G = to_networkx(kg)

    print("=== NetworkX 图分析 ===")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"是否连通: {nx.is_connected(G.to_undirected())}")
    print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

    # 中心性分析
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\\n度中心性最高的节点:")
    for node, cent in top_nodes:
        print(f"  {node}: {cent:.3f}")

    return G

def visualize_graph(kg, output_file="graph_visualization.png"):
    """可视化知识图谱"""
    G = to_networkx(kg)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.7)

    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)

    # 绘制标签
    labels = {node: G.nodes[node].get('name', node)[:10] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title(f"知识图谱: {kg.name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"图谱可视化已保存: {output_file}")

# 使用示例
G = analyze_with_networkx(kg)
visualize_graph(kg, "my_graph_visualization.png")
```

### 数据库集成

```python
import sqlite3
import json

def export_to_database(kg, db_path):
    """导出到 SQLite 数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graphs (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            data TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            graph_id TEXT,
            name TEXT,
            entity_type TEXT,
            description TEXT,
            confidence REAL,
            properties TEXT,
            created_at TEXT,
            FOREIGN KEY (graph_id) REFERENCES graphs (id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relations (
            id TEXT PRIMARY KEY,
            graph_id TEXT,
            head_entity_id TEXT,
            tail_entity_id TEXT,
            relation_type TEXT,
            description TEXT,
            confidence REAL,
            properties TEXT,
            created_at TEXT,
            FOREIGN KEY (graph_id) REFERENCES graphs (id)
        )
    ''')

    # 插入图谱信息
    cursor.execute('''
        INSERT OR REPLACE INTO graphs
        (id, name, description, data, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        kg.id, kg.name, kg.description,
        kg.export_to_json_string(),
        kg.created_at.isoformat(),
        kg.updated_at.isoformat()
    ))

    # 插入实体
    for entity in kg.entities.values():
        cursor.execute('''
            INSERT OR REPLACE INTO entities
            (id, graph_id, name, entity_type, description, confidence, properties, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entity.id, kg.id, entity.name, str(entity.entity_type),
            entity.description, entity.confidence,
            json.dumps(entity.properties), entity.created_at.isoformat()
        ))

    # 插入关系
    for relation in kg.relations.values():
        cursor.execute('''
            INSERT OR REPLACE INTO relations
            (id, graph_id, head_entity_id, tail_entity_id, relation_type,
             description, confidence, properties, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            relation.id, kg.id,
            relation.head_entity.id if relation.head_entity else None,
            relation.tail_entity.id if relation.tail_entity else None,
            str(relation.relation_type), relation.description,
            relation.confidence, json.dumps(relation.properties),
            relation.created_at.isoformat()
        ))

    conn.commit()
    conn.close()
    print(f"图谱已导出到数据库: {db_path}")

def import_from_database(db_path, graph_id):
    """从 SQLite 数据库导入"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询图谱数据
    cursor.execute('SELECT data FROM graphs WHERE id = ?', (graph_id,))
    result = cursor.fetchone()

    if result:
        json_data = result[0]
        kg = KnowledgeGraph.import_from_json_string(json_data)
        print(f"从数据库导入图谱: {kg.name}")
        return kg
    else:
        print(f"未找到图谱: {graph_id}")
        return None

# 使用示例
export_to_database(kg, "knowledge_graphs.db")
imported_kg = import_from_database("knowledge_graphs.db", kg.id)
```

## 总结

本教程介绍了 agraph 的完整导入导出功能：

1. **JSON 格式**: 完整保存所有数据，适合备份和系统间传输
2. **GraphML 格式**: 标准图形格式，适合可视化和分析
3. **高级功能**: 批量处理、条件导出、增量更新
4. **性能优化**: 大规模数据处理技巧
5. **工具集成**: 与 Gephi、NetworkX、数据库等的集成

根据您的具体需求选择合适的格式和方法，充分利用 agraph 的导入导出功能来管理和分析知识图谱数据。
