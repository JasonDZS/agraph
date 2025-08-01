"""
知识图谱构建器使用示例

本示例展示如何使用StandardGraphBuilder和MultiSourceGraphBuilder构建、验证和管理知识图谱。
"""

import asyncio
import logging

from agraph.builders.graph_builder import MultiSourceGraphBuilder, StandardGraphBuilder
from agraph.entities import Entity, EntityType
from agraph.graph import KnowledgeGraph
from agraph.relations import Relation, RelationType

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """基本使用示例"""
    print("=== 标准知识图谱构建器基本使用示例 ===")

    # 1. 创建标准构建器
    builder = StandardGraphBuilder()

    # 2. 准备示例文档
    texts = [
        """
        苹果公司是一家美国跨国科技公司，总部位于加利福尼亚州库比蒂诺。
        该公司设计、开发并销售消费电子产品、计算机软件和在线服务。
        苹果公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年4月1日创立。
        """,
        """
        iPhone是苹果公司开发的智能手机系列。第一代iPhone于2007年6月29日发布。
        iPhone运行iOS操作系统，具有触摸屏界面和多种传感器。
        iPhone改变了智能手机行业，成为了移动互联网时代的标志性产品。
        """,
        """
        史蒂夫·乔布斯是苹果公司的联合创始人和前首席执行官。
        他被认为是个人计算机革命的先驱，对苹果公司的成功起到了关键作用。
        乔布斯以其创新的产品设计理念和市场营销策略而闻名。
        """,
    ]

    try:
        # 3. 构建知识图谱
        print("正在构建知识图谱...")
        graph = builder.build_graph(texts=texts, graph_name="苹果公司知识图谱")

        print(f"构建完成! 实体数量: {len(graph.entities)}, 关系数量: {len(graph.relations)}")

        # 4. 显示实体信息
        print("\n=== 实体信息 ===")
        for i, entity in enumerate(list(graph.entities.values())[:8]):
            print(f"{i+1}. {entity.name} ({entity.entity_type.value})")
            print(f"   描述: {entity.description[:80]}..." if entity.description else "   无描述")
            print(f"   置信度: {entity.confidence:.2f}")
            print()

        # 5. 显示关系信息
        print("=== 关系信息 ===")
        for i, relation in enumerate(list(graph.relations.values())[:5]):
            if relation.head_entity and relation.tail_entity:
                print(f"{i+1}. {relation.head_entity.name} -> {relation.tail_entity.name}")
                print(f"   关系类型: {relation.relation_type.value}")
                print(f"   置信度: {relation.confidence:.2f}")
                print()

        return graph, builder

    except Exception as e:
        print(f"构建过程中出现错误: {e}")
        return None, builder


async def validation_example(graph: KnowledgeGraph, builder: StandardGraphBuilder):
    """图谱验证示例"""
    print("=== 知识图谱验证示例 ===")

    if not graph:
        print("图谱未构建，跳过验证示例")
        return

    try:
        # 验证图谱质量
        validation_result = builder.validate_graph(graph)

        print("图谱验证结果:")
        print(f"- 整体有效性: {'有效' if validation_result['valid'] else '无效'}")
        print(f"- 问题数量: {len(validation_result['issues'])}")

        # 显示统计信息
        stats = validation_result["statistics"]
        print(f"- 实体总数: {stats['total_entities']}")
        print(f"- 关系总数: {stats['total_relations']}")
        print(f"- 实体类型数: {len(stats['entity_types'])}")
        print(f"- 关系类型数: {len(stats['relation_types'])}")

        # 显示问题详情
        if validation_result["issues"]:
            print("\n发现的问题:")
            for i, issue in enumerate(validation_result["issues"][:3]):
                print(f"{i+1}. 类型: {issue['type']}, 严重程度: {issue.get('severity', 'medium')}")
                if "count" in issue:
                    print(f"   数量: {issue['count']}")

        # 显示建议
        if validation_result["recommendations"]:
            print("\n改进建议:")
            for i, rec in enumerate(validation_result["recommendations"][:3]):
                print(f"{i+1}. {rec}")

    except Exception as e:
        print(f"验证过程中出现错误: {e}")


async def update_example(graph: KnowledgeGraph, builder: StandardGraphBuilder):
    """图谱更新示例"""
    print("=== 知识图谱更新示例 ===")

    if not graph:
        print("图谱未构建，跳过更新示例")
        return

    try:
        # 创建新实体
        new_entities = [
            Entity(
                name="iPad", entity_type=EntityType.PRODUCT, description="苹果公司开发的平板电脑产品线", confidence=0.9
            ),
            Entity(
                name="macOS",
                entity_type=EntityType.SOFTWARE,
                description="苹果公司为Mac计算机开发的操作系统",
                confidence=0.85,
            ),
        ]

        # 创建新关系
        apple_entity = None
        for entity in graph.entities.values():
            if "苹果" in entity.name and entity.entity_type == EntityType.ORGANIZATION:
                apple_entity = entity
                break

        new_relations = []
        if apple_entity:
            for new_entity in new_entities:
                relation = Relation(
                    head_entity=apple_entity,
                    tail_entity=new_entity,
                    relation_type=RelationType.DEVELOPS,
                    confidence=0.8,
                    description=f"{apple_entity.name}开发了{new_entity.name}",
                )
                new_relations.append(relation)

        print("正在更新知识图谱...")
        original_entity_count = len(graph.entities)
        original_relation_count = len(graph.relations)

        updated_graph = builder.update_graph(graph, new_entities, new_relations)

        print("更新完成!")
        print(f"实体数量: {original_entity_count} -> {len(updated_graph.entities)}")
        print(f"关系数量: {original_relation_count} -> {len(updated_graph.relations)}")

        # 显示新添加的实体
        print("\n新添加的实体:")
        for entity in new_entities:
            if entity.id in updated_graph.entities:
                print(f"- {entity.name} ({entity.entity_type.value})")

        return updated_graph

    except Exception as e:
        print(f"更新过程中出现错误: {e}")
        return graph


async def multi_source_example():
    """多源图谱构建示例"""
    print("=== 多源知识图谱构建器示例 ===")

    # 创建多源构建器
    multi_builder = MultiSourceGraphBuilder()

    # 准备多个数据源
    sources = [
        {
            "type": "text",
            "name": "技术文档",
            "weight": 1.0,
            "data": [
                "Python是一种高级编程语言，以其简洁的语法和强大的功能而著称。",
                "Django是一个基于Python的Web框架，用于快速开发安全和可维护的网站。",
                "Flask是另一个Python Web框架，以其轻量级和灵活性而知名。",
            ],
        },
        {
            "type": "text",
            "name": "市场分析",
            "weight": 0.8,
            "data": [
                "Python在数据科学和机器学习领域占据主导地位。",
                "许多大型科技公司如Google、Facebook、Netflix都在使用Python。",
                "Python的就业市场需求持续增长，是最受欢迎的编程语言之一。",
            ],
        },
        {
            "type": "mixed",
            "name": "综合信息",
            "weight": 0.9,
            "data": {
                "texts": [
                    "Guido van Rossum是Python编程语言的创造者，被称为'仁慈的独裁者'。",
                    "Python软件基金会负责管理Python语言的发展。",
                ],
                "database_schema": {
                    "tables": [
                        {
                            "name": "programming_languages",
                            "columns": ["id", "name", "creator", "year_created"],
                            "sample_data": [{"name": "Python", "creator": "Guido van Rossum", "year_created": 1991}],
                        }
                    ]
                },
            },
        },
    ]

    try:
        print("正在从多个数据源构建知识图谱...")
        multi_graph = multi_builder.build_graph_from_multiple_sources(sources, "Python生态系统知识图谱")

        print("多源图谱构建完成!")
        print(f"实体数量: {len(multi_graph.entities)}")
        print(f"关系数量: {len(multi_graph.relations)}")

        # 显示不同置信度的实体
        print("\n=== 实体置信度分析 ===")
        high_conf = [e for e in multi_graph.entities.values() if e.confidence > 0.8]
        medium_conf = [e for e in multi_graph.entities.values() if 0.5 < e.confidence <= 0.8]
        low_conf = [e for e in multi_graph.entities.values() if e.confidence <= 0.5]

        print(f"高置信度实体 (>0.8): {len(high_conf)}")
        print(f"中等置信度实体 (0.5-0.8): {len(medium_conf)}")
        print(f"低置信度实体 (≤0.5): {len(low_conf)}")

        # 显示一些高置信度实体
        print("\n高置信度实体示例:")
        for entity in high_conf[:5]:
            print(f"- {entity.name}: {entity.confidence:.3f}")

        return multi_graph

    except Exception as e:
        print(f"多源构建过程中出现错误: {e}")
        return None


async def merge_graphs_example():
    """图谱合并示例"""
    print("=== 图谱合并示例 ===")

    builder = StandardGraphBuilder()

    # 创建两个不同主题的图谱
    tech_texts = [
        "人工智能是计算机科学的一个分支，包括机器学习和深度学习。",
        "TensorFlow是Google开发的机器学习框架。",
        "PyTorch是Facebook开发的深度学习框架。",
    ]

    business_texts = [
        "科技公司通过创新技术产品获得竞争优势。",
        "Google是全球领先的搜索引擎和云服务提供商。",
        "Facebook专注于社交网络和虚拟现实技术。",
    ]

    try:
        # 构建两个子图谱
        print("构建技术图谱...")
        tech_graph = builder.build_graph(texts=tech_texts, graph_name="技术图谱")

        print("构建商业图谱...")
        business_graph = builder.build_graph(texts=business_texts, graph_name="商业图谱")

        print(f"技术图谱: {len(tech_graph.entities)} 实体, {len(tech_graph.relations)} 关系")
        print(f"商业图谱: {len(business_graph.entities)} 实体, {len(business_graph.relations)} 关系")

        # 合并图谱
        print("\n正在合并图谱...")
        merged_graph = builder.merge_graphs([tech_graph, business_graph])

        print(f"合并后图谱: {len(merged_graph.entities)} 实体, {len(merged_graph.relations)} 关系")

        # 分析合并效果
        total_before = len(tech_graph.entities) + len(business_graph.entities)
        reduction = total_before - len(merged_graph.entities)
        print(f"实体去重效果: 原有{total_before}个实体，去重后{len(merged_graph.entities)}个，减少{reduction}个")

        return merged_graph

    except Exception as e:
        print(f"图谱合并过程中出现错误: {e}")
        return None


async def performance_example():
    """性能测试示例"""
    print("=== 性能测试示例 ===")

    import time

    builder = StandardGraphBuilder()

    # 生成大量测试文本
    large_texts = []
    for i in range(50):
        text = f"""
        公司{i}是一家专注于人工智能的科技企业。该公司成立于202{i % 10}年，
        主要产品包括智能助手{i}和数据分析平台{i}。公司{i}的创始人是张{i}和李{i}。
        公司{i}位于北京市海淀区，员工人数约{100 + i*10}人。
        """
        large_texts.append(text)

    try:
        print(f"准备处理{len(large_texts)}个文档...")

        start_time = time.time()
        large_graph = builder.build_graph(texts=large_texts, graph_name="大规模测试图谱")
        build_time = time.time() - start_time

        print("构建完成!")
        print(f"构建时间: {build_time:.2f} 秒")
        print(f"实体数量: {len(large_graph.entities)}")
        print(f"关系数量: {len(large_graph.relations)}")
        print(f"平均每秒处理: {len(large_texts)/build_time:.1f} 文档")

        # 验证性能
        start_time = time.time()
        validation_result = builder.validate_graph(large_graph)
        validation_time = time.time() - start_time

        print(f"验证时间: {validation_time:.2f} 秒")
        print(f"验证结果: {'通过' if validation_result['valid'] else '未通过'}")

        return large_graph

    except Exception as e:
        print(f"性能测试过程中出现错误: {e}")
        return None


async def main():
    """主函数"""
    print("知识图谱构建器完整测试示例")
    print("=" * 60)

    # 1. 基本使用示例
    graph, builder = await basic_usage_example()

    if graph and builder:
        # 2. 验证示例
        await validation_example(graph, builder)

        # 3. 更新示例
        # updated_graph = await update_example(graph, builder)

    # 4. 多源构建示例
    await multi_source_example()

    # 5. 图谱合并示例
    await merge_graphs_example()

    # 6. 性能测试示例
    await performance_example()

    print("\n" + "=" * 60)
    print("所有测试示例执行完成!")


if __name__ == "__main__":
    asyncio.run(main())
