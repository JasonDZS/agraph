# LightRAG Builder 使用教程

本教程将详细介绍如何使用 agraph 项目中的各种 LightRAG 构建器来构建和管理知识图谱。这些构建器遵循接口隔离原则（ISP），为不同的使用场景提供了专门的接口。

## 概述

agraph 提供了多种 LightRAG 构建器，每种都专门针对特定的用例设计：

- **MinimalLightRAGBuilder**: 最小化构建器，只提供基本的图谱构建功能
- **FlexibleLightRAGBuilder**: 灵活构建器，支持构建和更新功能
- **StreamingLightRAGBuilder**: 流式构建器，支持实时增量更新
- **BatchLightRAGBuilder**: 批量构建器，优化多数据源处理
- **LightRAGSearchBuilder**: 搜索专用构建器，专门用于搜索和导出
- **LightRAGBuilder**: 全功能构建器，包含所有功能（谨慎使用）

## 准备工作

首先确保安装了必要的依赖：

```bash
# 安装开发依赖
make install-dev

# 或者直接使用 pip
pip install -e .
```

创建工作目录：

```python
from pathlib import Path
Path("./workdir").mkdir(exist_ok=True)
```

## 1. 最小化构建器（MinimalLightRAGBuilder）

### 适用场景
- 只需要基本图构建功能
- 不需要更新、验证、导出等高级功能
- 轻量级应用

### 使用示例

```python
import asyncio
from agraph.builders.lightrag_builder import MinimalLightRAGBuilder

async def minimal_example():
    # 创建最小化构建器
    builder = MinimalLightRAGBuilder("./workdir/minimal_lightrag_storage")

    # 准备文档
    documents = [
        """
        北京是中华人民共和国的首都，位于华北地区。作为中国的政治、文化、国际交往、
        科技创新中心，北京有着3000多年建城史和860多年建都史。
        """,
        """
        清华大学是中国著名的高等学府，位于北京市海淀区。学校创建于1911年，
        是中国九校联盟成员，被誉为"红色工程师的摇篮"。
        """
    ]

    try:
        # 构建知识图谱
        graph = await builder.build_graph(texts=documents, graph_name="示例图谱")

        print(f"构建完成! 实体数量: {len(graph.entities)}, 关系数量: {len(graph.relations)}")

        # 查看实体信息
        for entity in list(graph.entities.values())[:3]:
            print(f"实体: {entity.name} ({entity.entity_type.value})")
            print(f"描述: {entity.description[:100]}...")

        return graph

    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        return None

# 运行示例
asyncio.run(minimal_example())
```

## 2. 灵活构建器（FlexibleLightRAGBuilder）

### 适用场景
- 需要构建图谱并支持后续更新
- 不需要验证、合并等高级功能
- 中等复杂度应用

### 使用示例

```python
from agraph.builders.lightrag_builder import FlexibleLightRAGBuilder

async def flexible_example():
    # 创建灵活构建器
    builder = FlexibleLightRAGBuilder("./workdir/flexible_lightrag_storage")

    # 初始文档
    initial_documents = [
        """
        北京是中华人民共和国的首都，位于华北地区。作为中国的政治、文化、国际交往、
        科技创新中心，北京有着3000多年建城史和860多年建都史。
        """,
        """
        清华大学是中国著名的高等学府，位于北京市海淀区。学校创建于1911年，
        是中国九校联盟成员，被誉为"红色工程师的摇篮"。
        """
    ]

    try:
        # 构建初始图谱
        graph = await builder.build_graph(texts=initial_documents, graph_name="可更新图谱")
        print(f"初始图谱: {len(graph.entities)} 实体, {len(graph.relations)} 关系")

        # 添加新文档更新图谱
        new_documents = [
            """
            上海是中华人民共和国的直辖市，位于长江三角洲地区。作为中国的经济中心，
            上海是全球著名的金融中心之一。
            """
        ]

        updated_graph = await builder.update_graph_with_texts(new_documents, "更新后的图谱")
        print(f"更新后图谱: {len(updated_graph.entities)} 实体, {len(updated_graph.relations)} 关系")

        return builder

    except Exception as e:
        print(f"灵活构建器示例失败: {e}")
        return None
    finally:
        builder.cleanup()

# 运行示例
asyncio.run(flexible_example())
```

## 3. 流式构建器（StreamingLightRAGBuilder）

### 适用场景
- 需要实时处理文档流
- 支持增量更新
- 不需要复杂的验证和合并功能

### 使用示例

```python
from agraph.builders.lightrag_builder import StreamingLightRAGBuilder

async def streaming_example():
    # 创建流式构建器
    streaming_builder = StreamingLightRAGBuilder("./workdir/streaming_lightrag_storage")

    # 初始文档
    initial_docs = [
        "人工智能是计算机科学的一个分支，致力于创建智能机器。",
        "机器学习是人工智能的核心技术之一。"
    ]

    try:
        # 构建初始图谱
        graph = await streaming_builder.build_graph(texts=initial_docs, graph_name="流式图谱")
        print(f"初始图谱: {len(graph.entities)} 实体, {len(graph.relations)} 关系")

        # 模拟实时文档流
        document_batches = [
            ["深度学习是机器学习的一个重要子领域。"],
            ["自然语言处理技术正在快速发展。", "计算机视觉在图像识别中应用广泛。"],
            ["强化学习通过奖励机制训练智能体。"]
        ]

        for i, batch in enumerate(document_batches):
            print(f"处理第 {i+1} 批文档: {len(batch)} 个文档")
            updated_graph = await streaming_builder.add_documents(batch)
            print(f"更新后: {len(updated_graph.entities)} 实体, {len(updated_graph.relations)} 关系")

    except Exception as e:
        print(f"流式构建器示例失败: {e}")
    finally:
        streaming_builder.cleanup()

# 运行示例
asyncio.run(streaming_example())
```

## 4. 批量构建器（BatchLightRAGBuilder）

### 适用场景
- 需要同时处理多个数据源并合并
- 不需要增量更新或验证功能
- 大规模数据处理

### 使用示例

```python
from agraph.builders.lightrag_builder import BatchLightRAGBuilder

async def batch_example():
    # 创建批量构建器
    batch_builder = BatchLightRAGBuilder("./workdir/batch_lightrag_storage")

    # 准备不同类型的数据源
    sources = [
        {
            "type": "text",
            "data": [
                "量子计算是利用量子力学现象进行计算的技术。",
                "量子比特是量子计算的基本单位。"
            ]
        },
        {
            "type": "text",
            "data": [
                "区块链是一种分布式账本技术。",
                "比特币是最著名的区块链应用。"
            ]
        },
        {
            "type": "mixed",
            "data": {
                "texts": [
                    "云计算提供了弹性和可扩展的计算资源。",
                    "边缘计算将计算能力推向网络边缘。"
                ]
            }
        }
    ]

    try:
        # 批量处理多个数据源
        merged_graph = await batch_builder.build_from_multiple_sources(
            sources, "批量处理图谱"
        )

        print(f"批量处理完成: {len(merged_graph.entities)} 实体, {len(merged_graph.relations)} 关系")

    except Exception as e:
        print(f"批量构建器示例失败: {e}")
    finally:
        batch_builder.cleanup()

# 运行示例
asyncio.run(batch_example())
```

## 5. 搜索专用构建器（LightRAGSearchBuilder）

### 适用场景
- 已有图谱数据
- 只需要搜索和导出功能
- 不需要构建功能

### 使用示例

```python
from agraph.builders.lightrag_builder import LightRAGSearchBuilder

async def search_example():
    # 创建搜索专用构建器（复用之前的数据）
    search_builder = LightRAGSearchBuilder("./workdir/flexible_lightrag_storage")

    try:
        # 测试不同类型的搜索
        queries = [
            ("北京的基本信息是什么？", "hybrid"),
            ("清华大学有什么特点？", "local"),
            ("上海是什么样的城市？", "global")
        ]

        for query, search_type in queries:
            try:
                print(f"查询: {query} (类型: {search_type})")
                result = await search_builder.search_graph(query, search_type)
                print(f"结果: {result.get('result', '无结果')[:150]}...\n")
            except Exception as e:
                print(f"搜索失败: {e}\n")

        # 获取统计信息
        stats = search_builder.get_statistics()
        print(f"图谱统计: {stats.get('entities_count', 0)} 实体, {stats.get('relations_count', 0)} 关系")

    except Exception as e:
        print(f"搜索示例失败: {e}")
    finally:
        search_builder.cleanup()

# 运行示例
asyncio.run(search_example())
```

## 6. 全功能构建器（LightRAGBuilder）

### 适用场景
⚠️ **注意：这违反了ISP原则，只有真正需要所有功能时才使用！**

- 需要所有功能的复杂应用
- 大多数客户端不应使用这个类

### 使用示例

```python
from agraph.builders.lightrag_builder import LightRAGBuilder

async def comprehensive_example():
    # 创建全功能构建器
    comprehensive_builder = LightRAGBuilder("./workdir/comprehensive_lightrag_storage")

    documents = [
        "物联网连接了数十亿的智能设备。",
        "5G网络提供了超高速的无线连接。",
        "边缘AI将人工智能推向设备端。"
    ]

    try:
        # 构建图谱
        graph = await comprehensive_builder.build_graph(texts=documents, graph_name="全功能图谱")
        print(f"构建完成: {len(graph.entities)} 实体, {len(graph.relations)} 关系")

        # 可用功能演示
        print("可用功能:")
        print("✓ 构建功能 (BasicGraphBuilder)")
        print("✓ 更新功能 (UpdatableGraphBuilder)")
        print("✓ 验证功能 (GraphValidator)")
        print("✓ 合并功能 (GraphMerger)")
        print("✓ 导出功能 (GraphExporter)")
        print("✓ 统计功能 (GraphStatistics)")

        # 获取统计信息
        stats = comprehensive_builder.get_statistics()
        print(f"统计信息: {stats.get('entities_count', 0)} 实体, {stats.get('relations_count', 0)} 关系")

    except Exception as e:
        print(f"全功能构建器示例失败: {e}")
    finally:
        comprehensive_builder.cleanup()

# 运行示例
asyncio.run(comprehensive_example())
```

## 完整示例运行

```python
import asyncio
from pathlib import Path

async def main():
    """运行所有示例"""
    print("🚀 LightRAG Builders 使用教程示例")
    print("=" * 60)

    # 确保工作目录存在
    Path("./workdir").mkdir(exist_ok=True)

    try:
        # 依次运行各个示例
        await minimal_example()
        await flexible_example()
        await streaming_example()
        await batch_example()
        await search_example()
        await comprehensive_example()

    except Exception as e:
        print(f"示例执行过程中出现错误: {e}")

    print("\n🎉 所有示例执行完成!")
    print("   选择适合你需求的构建器，享受接口隔离原则带来的好处!")

if __name__ == "__main__":
    asyncio.run(main())
```

## 最佳实践

1. **选择合适的构建器**: 根据实际需求选择最小化的接口，避免使用全功能构建器
2. **资源管理**: 始终在 finally 块中调用 `cleanup()` 方法
3. **错误处理**: 使用 try-catch 块处理可能的异常
4. **工作目录**: 为不同的构建器使用不同的工作目录
5. **异步编程**: 所有构建器操作都是异步的，需要使用 `await` 关键字

## 配置说明

### 环境变量
确保设置了必要的环境变量：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 工作目录结构
```
workdir/
├── minimal_lightrag_storage/
├── flexible_lightrag_storage/
├── streaming_lightrag_storage/
├── batch_lightrag_storage/
├── comprehensive_lightrag_storage/
└── search_lightrag_storage/
```

## 故障排除

1. **API 密钥错误**: 确保 OPENAI_API_KEY 环境变量已正确设置
2. **权限问题**: 确保对工作目录有读写权限
3. **依赖缺失**: 运行 `make install-dev` 安装所有依赖
4. **端口冲突**: 如果使用 Neo4j，确保端口 7687 可用

通过本教程，你应该能够根据具体需求选择和使用合适的 LightRAG 构建器来构建和管理知识图谱。
