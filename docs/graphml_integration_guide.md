# GraphML 集成指南

本指南详细介绍如何使用 agraph 的 GraphML 功能与图形分析工具集成。

## GraphML 简介

GraphML 是一种基于 XML 的图形描述语言，广泛用于：

- 图形可视化软件（Gephi、Cytoscape）
- 网络分析库（NetworkX、igraph）
- 学术研究和数据分析

## 与 Gephi 集成

### 基本工作流

```python
from agraph import KnowledgeGraph, Entity, Relation, EntityType, RelationType

# 1. 创建知识图谱
kg = KnowledgeGraph(name="社交网络分析")

# 2. 添加节点（人物）
alice = Entity(name="Alice", entity_type=EntityType.PERSON,
               properties={"age": 25, "department": "Engineering"})
bob = Entity(name="Bob", entity_type=EntityType.PERSON,
             properties={"age": 30, "department": "Marketing"})
charlie = Entity(name="Charlie", entity_type=EntityType.PERSON,
                 properties={"age": 28, "department": "Engineering"})

kg.add_entity(alice)
kg.add_entity(bob)
kg.add_entity(charlie)

# 3. 添加关系（友谊）
friendship1 = Relation(
    head_entity=alice, tail_entity=bob,
    relation_type=RelationType.RELATED_TO,
    description="朋友关系",
    properties={"weight": 0.8, "years_known": 5}
)
friendship2 = Relation(
    head_entity=bob, tail_entity=charlie,
    relation_type=RelationType.RELATED_TO,
    description="同事关系",
    properties={"weight": 0.6, "years_known": 2}
)

kg.add_relation(friendship1)
kg.add_relation(friendship2)

# 4. 导出为 GraphML
kg.export_to_graphml("social_network.graphml")
```

### Gephi 优化设置

```python
def optimize_for_gephi(kg, output_file):
    \"\"\"为 Gephi 优化 GraphML 输出\"\"\"

    # 为节点添加 Gephi 特定属性
    for entity in kg.entities.values():
        # 设置节点标签（Gephi 显示名称）
        entity.properties["Label"] = entity.name

        # 设置节点大小（基于连接数或其他指标）
        degree = len([r for r in kg.relations.values()
                     if r.head_entity == entity or r.tail_entity == entity])
        entity.properties["Size"] = max(degree * 10, 10)

        # 设置节点颜色（基于类型）
        color_map = {
            "person": "#FF6B6B",
            "organization": "#4ECDC4",
            "concept": "#45B7D1",
            "location": "#96CEB4"
        }
        entity.properties["Color"] = color_map.get(str(entity.entity_type), "#CCCCCC")

    # 为边添加 Gephi 特定属性
    for relation in kg.relations.values():
        # 设置边标签
        relation.properties["Label"] = relation.description

        # 设置边权重（影响布局）
        weight = relation.properties.get("weight", 1.0)
        relation.properties["Weight"] = weight

        # 设置边类型
        relation.properties["Type"] = str(relation.relation_type)

    # 导出
    kg.export_to_graphml(output_file)
    print(f"Gephi 优化文件已生成: {output_file}")
    print("\\n在 Gephi 中的使用步骤:")
    print("1. 打开 Gephi")
    print("2. 文件 -> 打开 -> 选择生成的 .graphml 文件")
    print("3. 在外观面板中使用 'Size' 调整节点大小")
    print("4. 使用 'Color' 为节点着色")
    print("5. 在布局面板中选择 ForceAtlas 2 进行布局")

# 使用示例
optimize_for_gephi(kg, "gephi_optimized.graphml")
```

### Gephi 分析工作流

1. **导入数据**

   ```bash
   # 在 Gephi 中
   文件 -> 打开 -> 选择 .graphml 文件
   ```

2. **网络布局**
   - 推荐算法：ForceAtlas 2、Fruchterman Reingold
   - 调整参数：引力、排斥力、边权重影响

3. **视觉化**
   - 节点大小：基于度中心性或其他指标
   - 节点颜色：基于社区检测或节点类型
   - 边粗细：基于权重

4. **分析功能**
   - 统计面板：度分布、路径长度
   - 社区检测：Modularity 算法
   - 中心性分析：度中心性、介数中心性

## 与 NetworkX 集成

### 双向转换

```python
import networkx as nx
import matplotlib.pyplot as plt
import tempfile

def kg_to_networkx(kg):
    \"\"\"将 KnowledgeGraph 转换为 NetworkX 图\"\"\"
    # 通过临时文件转换
    with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as tmp:
        kg.export_to_graphml(tmp.name)
        G = nx.read_graphml(tmp.name)
    return G

def networkx_to_kg(G, name="从 NetworkX 导入"):
    \"\"\"将 NetworkX 图转换为 KnowledgeGraph\"\"\"
    with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as tmp:
        nx.write_graphml(G, tmp.name)
        kg = KnowledgeGraph.import_from_graphml(tmp.name)
        kg.name = name
    return kg

# 使用示例
G = kg_to_networkx(kg)
print(f"NetworkX 图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 在 NetworkX 中分析
centrality = nx.degree_centrality(G)
print("度中心性最高的节点:", max(centrality, key=centrality.get))

# 转换回 KnowledgeGraph
kg_back = networkx_to_kg(G, "NetworkX 处理后")
```

### 网络分析示例

```python
def analyze_network(kg):
    \"\"\"使用 NetworkX 进行网络分析\"\"\"
    G = kg_to_networkx(kg)

    print("=== 网络分析报告 ===")

    # 基本统计
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

    # 连通性
    if nx.is_connected(G.to_undirected()):
        print("图是连通的")
        diameter = nx.diameter(G.to_undirected())
        print(f"直径: {diameter}")
    else:
        components = list(nx.connected_components(G.to_undirected()))
        print(f"图不连通，有 {len(components)} 个连通分量")

    # 中心性分析
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)

    print("\\n=== 重要节点 ===")
    print("度中心性最高:")
    for node in sorted(degree_cent, key=degree_cent.get, reverse=True)[:3]:
        print(f"  {node}: {degree_cent[node]:.3f}")

    print("介数中心性最高:")
    for node in sorted(betweenness_cent, key=betweenness_cent.get, reverse=True)[:3]:
        print(f"  {node}: {betweenness_cent[node]:.3f}")

    # 社区检测
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.greedy_modularity_communities(G.to_undirected())
        print(f"\\n检测到 {len(communities)} 个社区")
        for i, community in enumerate(communities):
            print(f"  社区 {i+1}: {len(community)} 个节点")
    except ImportError:
        print("\\n需要安装 networkx[extra] 进行社区检测")

    return G

# 分析示例
analyzed_graph = analyze_network(kg)
```

### 可视化

```python
def visualize_knowledge_graph(kg, output_file="network_viz.png", layout="spring"):
    \"\"\"可视化知识图谱\"\"\"
    G = kg_to_networkx(kg)

    plt.figure(figsize=(12, 8))

    # 选择布局算法
    if layout == "spring":
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)

    # 计算节点大小（基于度）
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 100 + 100 for node in G.nodes()]

    # 绘制网络
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color='lightblue',
                          alpha=0.7)

    nx.draw_networkx_edges(G, pos,
                          alpha=0.5,
                          width=1,
                          edge_color='gray')

    # 添加标签
    labels = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        labels[node] = node_data.get('name', node)[:10]  # 截断长标签

    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title(f"知识图谱可视化: {kg.name}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"可视化图片已保存: {output_file}")

# 可视化示例
visualize_knowledge_graph(kg, "my_network.png", "spring")
```

## 与 Cytoscape 集成

### 准备 Cytoscape 文件

```python
def prepare_for_cytoscape(kg, output_file):
    \"\"\"为 Cytoscape 准备 GraphML 文件\"\"\"

    # Cytoscape 偏好特定的属性名称
    for entity in kg.entities.values():
        # 设置节点标签
        entity.properties["name"] = entity.name
        entity.properties["label"] = entity.name

        # 设置节点类型
        entity.properties["type"] = str(entity.entity_type)
        entity.properties["nodeType"] = str(entity.entity_type)

        # 设置描述
        entity.properties["description"] = entity.description

        # 设置其他属性
        entity.properties["confidence"] = entity.confidence

    for relation in kg.relations.values():
        # 设置边标签
        relation.properties["name"] = relation.description
        relation.properties["label"] = relation.description

        # 设置边类型
        relation.properties["interaction"] = str(relation.relation_type)
        relation.properties["edgeType"] = str(relation.relation_type)

        # 设置权重
        weight = relation.properties.get("weight", 1.0)
        relation.properties["weight"] = weight
        relation.properties["confidence"] = relation.confidence

    kg.export_to_graphml(output_file)

    print(f"Cytoscape 文件已生成: {output_file}")
    print("\\n在 Cytoscape 中的使用步骤:")
    print("1. 打开 Cytoscape")
    print("2. File -> Import -> Network from File")
    print("3. 选择生成的 .graphml 文件")
    print("4. 在 Style 面板中设置视觉映射")
    print("5. 使用 Layout 菜单选择布局算法")

# 使用示例
prepare_for_cytoscape(kg, "cytoscape_network.graphml")
```

## 与 igraph 集成

### R 语言集成

```python
def export_for_r_igraph(kg, output_file):
    \"\"\"为 R igraph 包准备数据\"\"\"
    kg.export_to_graphml(output_file)

    # 生成 R 脚本
    r_script = f\"\"\"
# R 脚本：加载和分析知识图谱
library(igraph)

# 读取 GraphML 文件
g <- read_graph("{output_file}", format = "graphml")

# 基本信息
cat("节点数:", vcount(g), "\\n")
cat("边数:", ecount(g), "\\n")
cat("是否连通:", is_connected(g), "\\n")

# 中心性分析
degree_cent <- degree(g, mode = "all")
betweenness_cent <- betweenness(g)
closeness_cent <- closeness(g)

# 打印最重要的节点
cat("\\n度中心性最高的节点:\\n")
print(head(sort(degree_cent, decreasing = TRUE), 5))

# 社区检测
communities <- cluster_louvain(g)
cat("\\n检测到", length(communities), "个社区\\n")

# 可视化
png("network_plot.png", width = 800, height = 600)
plot(g,
     vertex.label = V(g)$name,
     vertex.size = degree_cent * 2 + 5,
     vertex.color = membership(communities),
     edge.arrow.size = 0.5,
     layout = layout_with_fr(g))
title("知识图谱网络分析")
dev.off()

cat("\\n可视化图片已保存: network_plot.png\\n")
\"\"\"

    with open(f"{output_file}.R", "w", encoding="utf-8") as f:
        f.write(r_script)

    print(f"R 脚本已生成: {output_file}.R")
    print("运行命令: Rscript", f"{output_file}.R")

# 使用示例
export_for_r_igraph(kg, "analysis.graphml")
```

## 最佳实践

### 性能优化

```python
def optimize_large_graph_export(kg, output_file, max_nodes=1000):
    \"\"\"优化大型图谱的 GraphML 导出\"\"\"

    if len(kg.entities) <= max_nodes:
        # 小图谱直接导出
        kg.export_to_graphml(output_file)
        return

    # 大图谱需要简化
    print(f"图谱过大 ({len(kg.entities)} 节点)，进行简化...")

    # 创建简化版本
    simplified_kg = KnowledgeGraph(
        name=f"{kg.name}_simplified",
        description=f"简化版本，最多 {max_nodes} 个节点"
    )

    # 选择最重要的节点（基于连接数）
    entity_degrees = {}
    for entity_id, entity in kg.entities.items():
        degree = len([r for r in kg.relations.values()
                     if (r.head_entity and r.head_entity.id == entity_id) or
                        (r.tail_entity and r.tail_entity.id == entity_id)])
        entity_degrees[entity_id] = degree

    # 取前 N 个最重要的节点
    top_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    selected_entity_ids = {eid for eid, _ in top_entities}

    # 添加选中的实体
    for entity_id in selected_entity_ids:
        simplified_kg.add_entity(kg.entities[entity_id])

    # 添加连接这些实体的关系
    for relation in kg.relations.values():
        if (relation.head_entity and relation.head_entity.id in selected_entity_ids and
            relation.tail_entity and relation.tail_entity.id in selected_entity_ids):
            simplified_kg.add_relation(relation)

    print(f"简化后: {len(simplified_kg.entities)} 节点, {len(simplified_kg.relations)} 边")
    simplified_kg.export_to_graphml(output_file)

# 使用示例
optimize_large_graph_export(kg, "large_graph_simplified.graphml", max_nodes=500)
```

### 属性映射

```python
def customize_graphml_attributes(kg, node_mappings=None, edge_mappings=None):
    \"\"\"自定义 GraphML 属性映射\"\"\"

    # 默认节点属性映射
    default_node_mappings = {
        "label": lambda e: e.name,
        "type": lambda e: str(e.entity_type),
        "size": lambda e: len(e.description),
        "color": lambda e: hash(str(e.entity_type)) % 16777215  # 随机颜色
    }

    # 默认边属性映射
    default_edge_mappings = {
        "label": lambda r: r.description,
        "type": lambda r: str(r.relation_type),
        "weight": lambda r: r.properties.get("weight", 1.0),
        "style": lambda r: "solid" if r.confidence > 0.7 else "dashed"
    }

    node_mappings = node_mappings or default_node_mappings
    edge_mappings = edge_mappings or default_edge_mappings

    # 应用节点映射
    for entity in kg.entities.values():
        for attr_name, mapping_func in node_mappings.items():
            try:
                entity.properties[attr_name] = mapping_func(entity)
            except Exception as e:
                print(f"节点属性映射失败 {attr_name}: {e}")

    # 应用边映射
    for relation in kg.relations.values():
        for attr_name, mapping_func in edge_mappings.items():
            try:
                relation.properties[attr_name] = mapping_func(relation)
            except Exception as e:
                print(f"边属性映射失败 {attr_name}: {e}")

# 自定义映射示例
custom_node_mappings = {
    "importance": lambda e: e.confidence * 10,
    "category": lambda e: "重要" if e.confidence > 0.8 else "一般",
    "display_name": lambda e: f"{e.name} ({str(e.entity_type)})"
}

customize_graphml_attributes(kg, custom_node_mappings)
kg.export_to_graphml("custom_attributes.graphml")
```

## 故障排除

### 常见问题

1. **中文字符显示问题**

   ```python
   # 确保使用 UTF-8 编码
   kg.export_to_graphml("graph.graphml")
   # GraphML 自动使用 UTF-8 编码
   ```

2. **大型图谱内存问题**

   ```python
   # 使用简化版本
   optimize_large_graph_export(kg, "simplified.graphml", max_nodes=1000)
   ```

3. **属性类型不兼容**

   ```python
   # 确保属性值是基本类型
   for entity in kg.entities.values():
       for key, value in entity.properties.items():
           if isinstance(value, (list, dict, set)):
               entity.properties[key] = str(value)
   ```

### 验证导出文件

```python
def validate_graphml_export(file_path):
    \"\"\"验证 GraphML 文件\"\"\"
    try:
        # 尝试重新导入
        kg_test = KnowledgeGraph.import_from_graphml(file_path)
        print(f"✅ 文件有效: {len(kg_test.entities)} 节点, {len(kg_test.relations)} 边")
        return True
    except Exception as e:
        print(f"❌ 文件无效: {e}")
        return False

# 验证示例
if validate_graphml_export("my_graph.graphml"):
    print("可以安全地在其他工具中使用")
```

## 总结

GraphML 集成为 agraph 提供了强大的图形分析能力：

1. **Gephi**: 交互式网络可视化
2. **NetworkX**: Python 网络分析
3. **Cytoscape**: 生物信息学网络分析
4. **igraph**: R/Python 统计分析

选择合适的工具组合，可以充分发挥知识图谱的分析价值。
