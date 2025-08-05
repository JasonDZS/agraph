"""
Improved LightRAG Builders Usage Examples

This example demonstrates how to use the new ISP-compliant LightRAG builders
for different use cases, showing the benefits of Interface Segregation Principle.
"""

import asyncio
from pathlib import Path

from agraph.builders.lightrag_builder import LightRAGBuilder  # Comprehensive builder
from agraph.builders.lightrag_builder import (
    BatchLightRAGBuilder,
    FlexibleLightRAGBuilder,
    LightRAGSearchBuilder,
    MinimalLightRAGBuilder,
    StreamingLightRAGBuilder,
)


async def minimal_builder_example():
    """最小化构建器示例 - 演示ISP原则：只需要基本构建功能的客户端"""
    print("=== 最小化LightRAG构建器示例 - 只有核心构建功能 ===")
    print("适用场景：只需要基本图构建，不需要更新、验证、导出等功能\n")

    # 1. 创建最小化构建器 - 只实现BasicGraphBuilder接口
    builder = MinimalLightRAGBuilder("./workdir/minimal_lightrag_storage")

    # 2. 准备示例文档
    documents = [
        """
        北京是中华人民共和国的首都，位于华北地区。作为中国的政治、文化、国际交往、
        科技创新中心，北京有着3000多年建城史和860多年建都史。北京市下辖16个区，
        总面积16410.54平方千米。2022年末，北京市常住人口2184.3万人。
        """,
        """
        清华大学是中国著名的高等学府，位于北京市海淀区。学校创建于1911年，
        是中国九校联盟成员，被誉为"红色工程师的摇篮"。清华大学在工程技术、
        自然科学、经济管理、人文社科等多个学科领域都有很强的实力。
        """,
        """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种
        新的能以人类智能相似的方式做出反应的智能机器。人工智能包括机器学习、
        深度学习、自然语言处理、计算机视觉等多个子领域。
        """,
    ]

    try:
        # 3. 构建知识图谱 - 最小化接口，只有build_graph方法
        print("正在构建知识图谱...")
        graph = await builder.build_graph(texts=documents, graph_name="最小化示例图谱")

        print(f"构建完成! 实体数量: {len(graph.entities)}, 关系数量: {len(graph.relations)}")

        # 4. 显示部分实体信息
        print("\n=== 实体信息 ===")
        for i, entity in enumerate(list(graph.entities.values())[:5]):
            print(f"{i+1}. {entity.name} ({entity.entity_type.value})")
            print(f"   描述: {entity.description[:100]}...")
            print()

        # 5. 显示部分关系信息
        print("=== 关系信息 ===")
        for i, relation in enumerate(list(graph.relations.values())[:3]):
            print(f"{i+1}. {relation.head_entity.name} -> {relation.tail_entity.name}")
            print(f"   关系类型: {relation.relation_type.value}")
            print(f"   置信度: {relation.confidence}")
            print()

        print("✅ 最小化构建器示例完成 - 轻量级，专注核心功能\n")
        return graph

    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        return None
    finally:
        # 在这里不清理资源，因为我们要返回builder供后续使用
        pass


async def flexible_builder_example():
    """灵活构建器示例 - 演示ISP原则：需要构建+更新功能的客户端"""
    print("=== 灵活LightRAG构建器示例 - 支持构建和更新 ===")
    print("适用场景：需要构建图谱并支持后续更新，但不需要验证、合并等高级功能\n")

    # 1. 创建灵活构建器 - 实现UpdatableGraphBuilder接口
    builder = FlexibleLightRAGBuilder("./workdir/flexible_lightrag_storage")

    # 2. 准备初始文档
    initial_documents = [
        """
        北京是中华人民共和国的首都，位于华北地区。作为中国的政治、文化、国际交往、
        科技创新中心，北京有着3000多年建城史和860多年建都史。
        """,
        """
        清华大学是中国著名的高等学府，位于北京市海淀区。学校创建于1911年，
        是中国九校联盟成员，被誉为"红色工程师的摇篮"。
        """,
    ]

    try:
        # 3. 构建初始图谱
        print("构建初始图谱...")
        graph = await builder.build_graph(texts=initial_documents, graph_name="可更新示例图谱")
        print(f"初始图谱: {len(graph.entities)} 实体, {len(graph.relations)} 关系")

        # 4. 演示更新功能 - 这是UpdatableGraphBuilder接口的特色
        new_documents = [
            """
            上海是中华人民共和国的直辖市，位于长江三角洲地区。作为中国的经济中心，
            上海是全球著名的金融中心之一。
            """
        ]

        print("\n添加新文档更新图谱...")
        updated_graph = await builder.update_graph_with_texts(new_documents, "更新后的示例图谱")
        print(f"更新后图谱: {len(updated_graph.entities)} 实体, {len(updated_graph.relations)} 关系")

        print("✅ 灵活构建器示例完成 - 支持构建和更新，接口适度\n")
        return builder

    except Exception as e:
        print(f"灵活构建器示例失败: {e}")
        return None
    finally:
        builder.cleanup()


async def search_example():
    """搜索专用构建器示例 - 演示ISP原则：只需要搜索功能的客户端"""
    print("=== LightRAG搜索构建器示例 - 专门用于搜索和导出 ===")
    print("适用场景：已有图谱数据，只需要搜索和导出功能，不需要构建功能\n")

    # 1. 创建搜索专用构建器 - 只实现GraphExporter接口
    search_builder = LightRAGSearchBuilder("./workdir/flexible_lightrag_storage")  # 复用之前的数据

    try:
        # 2. 测试不同类型的搜索 - 搜索构建器的核心功能
        queries = [
            ("北京的基本信息是什么？", "hybrid"),
            ("清华大学有什么特点？", "local"),
            ("上海是什么样的城市？", "global"),
        ]

        for query, search_type in queries:
            try:
                print(f"查询: {query} (类型: {search_type})")
                result = await search_builder.search_graph(query, search_type)
                print(f"结果: {result.get('result', '无结果')[:150]}...\n")
            except Exception as e:
                print(f"搜索失败: {e}\n")

        # 3. 演示导出功能 - GraphExporter接口的功能
        print("测试导出功能...")
        stats = search_builder.get_statistics()
        print(f"图谱统计: {stats.get('entities_count', 0)} 实体, {stats.get('relations_count', 0)} 关系")

        print("✅ 搜索构建器示例完成 - 专注搜索和导出，不包含构建功能\n")

    except Exception as e:
        print(f"搜索示例失败: {e}")
    finally:
        search_builder.cleanup()


async def streaming_builder_example():
    """流式构建器示例 - 演示ISP原则：需要实时增量更新的客户端"""
    print("=== 流式LightRAG构建器示例 - 支持实时增量更新 ===")
    print("适用场景：需要实时处理文档流，支持增量更新，但不需要复杂的验证和合并功能\n")

    # 1. 创建流式构建器 - 实现StreamingGraphBuilder和IncrementalBuilder接口
    streaming_builder = StreamingLightRAGBuilder("./workdir/streaming_lightrag_storage")

    # 2. 准备初始文档
    initial_docs = [
        "人工智能是计算机科学的一个分支，致力于创建智能机器。",
        "机器学习是人工智能的核心技术之一。",
    ]

    try:
        # 3. 构建初始图谱
        print("构建初始流式图谱...")
        graph = await streaming_builder.build_graph(texts=initial_docs, graph_name="流式示例图谱")
        print(f"初始图谱: {len(graph.entities)} 实体, {len(graph.relations)} 关系")

        # 4. 模拟实时文档流 - IncrementalBuilder接口的特色功能
        document_batches = [
            ["深度学习是机器学习的一个重要子领域。"],
            ["自然语言处理技术正在快速发展。", "计算机视觉在图像识别中应用广泛。"],
            ["强化学习通过奖励机制训练智能体。"],
        ]

        for i, batch in enumerate(document_batches):
            print(f"\n处理第 {i+1} 批文档: {len(batch)} 个文档")
            updated_graph = await streaming_builder.add_documents(batch)
            print(f"更新后: {len(updated_graph.entities)} 实体, {len(updated_graph.relations)} 关系")

        print("✅ 流式构建器示例完成 - 支持实时增量更新，适合文档流处理\n")

    except Exception as e:
        print(f"流式构建器示例失败: {e}")
    finally:
        streaming_builder.cleanup()


async def batch_builder_example():
    """批量构建器示例 - 演示ISP原则：需要处理多数据源的客户端"""
    print("=== 批量LightRAG构建器示例 - 优化多数据源处理 ===")
    print("适用场景：需要同时处理多个数据源并合并，但不需要增量更新或验证功能\n")

    # 1. 创建批量构建器 - 实现BatchGraphBuilder和GraphMerger接口
    batch_builder = BatchLightRAGBuilder("./workdir/batch_lightrag_storage")

    # 2. 准备不同类型的数据源
    sources = [
        {
            "type": "text",
            "data": [
                "量子计算是利用量子力学现象进行计算的技术。",
                "量子比特是量子计算的基本单位。",
            ],
        },
        {
            "type": "text",
            "data": [
                "区块链是一种分布式账本技术。",
                "比特币是最著名的区块链应用。",
            ],
        },
        {
            "type": "mixed",
            "data": {
                "texts": [
                    "云计算提供了弹性和可扩展的计算资源。",
                    "边缘计算将计算能力推向网络边缘。",
                ]
            },
        },
    ]

    try:
        # 3. 批量处理多个数据源 - BatchGraphBuilder接口的特色功能
        print(f"批量处理 {len(sources)} 个数据源...")
        merged_graph = await batch_builder.build_from_multiple_sources(sources, "批量处理示例图谱")

        print(f"批量处理完成: {len(merged_graph.entities)} 实体, {len(merged_graph.relations)} 关系")
        print("✅ 批量构建器示例完成 - 高效处理多数据源，支持合并功能\n")

    except Exception as e:
        print(f"批量构建器示例失败: {e}")
    finally:
        batch_builder.cleanup()


async def comprehensive_builder_example():
    """全功能构建器示例 - 演示ISP反模式：需要所有功能的客户端"""
    print("=== 全功能LightRAG构建器示例 - 包含所有功能 ===")
    print("适用场景：需要所有功能的复杂应用，但大多数客户端不应使用这个类\n")
    print("⚠️  注意：这违反了ISP原则，只有真正需要所有功能时才使用！")

    # 1. 创建全功能构建器 - 实现所有接口（违反ISP）
    comprehensive_builder = LightRAGBuilder("./workdir/comprehensive_lightrag_storage")

    # 2. 准备测试文档
    documents = [
        "物联网连接了数十亿的智能设备。",
        "5G网络提供了超高速的无线连接。",
        "边缘AI将人工智能推向设备端。",
    ]

    try:
        # 3. 构建图谱
        print("使用全功能构建器构建图谱...")
        graph = await comprehensive_builder.build_graph(texts=documents, graph_name="全功能示例图谱")
        print(f"构建完成: {len(graph.entities)} 实体, {len(graph.relations)} 关系")

        # 4. 演示所有功能都可用（但客户端可能不需要）
        print("\n可用功能演示:")
        print("✓ 构建功能 (BasicGraphBuilder)")
        print("✓ 更新功能 (UpdatableGraphBuilder)")
        print("✓ 验证功能 (GraphValidator)")
        print("✓ 合并功能 (GraphMerger)")
        print("✓ 导出功能 (GraphExporter)")
        print("✓ 统计功能 (GraphStatistics)")

        # 5. 获取统计信息
        stats = comprehensive_builder.get_statistics()
        print(f"\n统计信息: {stats.get('entities_count', 0)} 实体, {stats.get('relations_count', 0)} 关系")

        print("\n⚠️  全功能构建器示例完成 - 功能齐全但违反ISP，谨慎使用\n")

    except Exception as e:
        print(f"全功能构建器示例失败: {e}")
    finally:
        comprehensive_builder.cleanup()


async def main():
    """主函数 - 演示所有ISP-compliant LightRAG构建器"""
    print("🚀 ISP-Compliant LightRAG Builders Examples")
    print("展示接口隔离原则在LightRAG构建器中的应用")
    print("=" * 60)
    print()

    try:
        await comprehensive_builder_example()
    except Exception as e:
        print(f"示例执行过程中出现错误: {e}")

    print("\n🎉 所有ISP-compliant LightRAG构建器示例执行完成!")
    print("   选择适合你需求的构建器，享受接口隔离原则带来的好处!")


if __name__ == "__main__":
    # 确保工作目录存在
    Path("./workdir").mkdir(exist_ok=True)

    # 运行所有示例
    asyncio.run(main())
