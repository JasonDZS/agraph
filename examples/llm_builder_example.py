"""
LLM ISP构建器使用示例

展示如何根据不同需求选择合适的LLM图构建器，
体现Interface Segregation Principle的优势
"""

import asyncio
import logging

from agraph.builders import (
    BatchLLMGraphBuilder,
    FlexibleLLMGraphBuilder,
    LLMGraphBuilder,
    LLMSearchBuilder,
    MinimalLLMGraphBuilder,
    StreamingLLMGraphBuilder,
)
from agraph.config import settings
from agraph.embeddings import JsonVectorStorage
from agraph.entities import Entity
from agraph.types import EntityType

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_builder():
    """
    示例1: 基础构建器

    适用场景：
    - 只需要简单的文本到图谱转换
    - 不需要更新、合并、验证等功能
    - 追求最小依赖和简单性
    """
    print("\n" + "=" * 50)
    print("📝 示例1: LLM基础构建器")
    print("=" * 50)

    # 创建基础构建器
    builder = MinimalLLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,
        temperature=0.1,
    )

    # 示例文本
    texts = [
        "苹果公司是一家美国跨国技术公司，总部位于加利福尼亚州库比蒂诺。",
        "史蒂夫·乔布斯是苹果公司的联合创始人，他在2011年去世。",
        "iPhone是苹果公司开发的智能手机产品线。",
    ]

    try:
        # 构建图谱 - 现在是异步的
        graph = await builder.build_graph(texts=texts, graph_name="basic_example_graph")

        print("✅ 成功构建基础图谱:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")
        print(f"   - 图谱名称: {graph.name}")

        # 显示一些实体
        print("\n📋 实体示例:")
        for i, (entity_id, entity) in enumerate(list(graph.entities.items())[:3]):
            print(f"   {i+1}. {entity.name} ({entity.entity_type.value})")

    except Exception as e:
        print(f"❌ 基础构建器示例失败: {e}")


async def example_updatable_builder():
    """
    示例2: 可更新构建器

    适用场景：
    - 需要构建图谱后进行增量更新
    - 不需要合并、验证等高级功能
    - 追求构建+更新的组合功能
    """
    print("\n" + "=" * 50)
    print("🔄 示例2: LLM可更新构建器")
    print("=" * 50)

    # 创建可更新构建器
    builder = FlexibleLLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        vector_storage=JsonVectorStorage(file_path="workdir/isp_vector_store.json"),  # 假设使用JSON存储作为向量存储
    )

    # 初始文本
    initial_texts = ["微软公司是一家美国跨国技术公司。", "比尔·盖茨是微软公司的联合创始人。"]

    try:
        # 构建初始图谱
        graph = await builder.build_graph(texts=initial_texts, graph_name="updatable_example_graph")

        print("✅ 初始图谱构建完成:")
        print(f"   - 初始实体数: {len(graph.entities)}")
        print(f"   - 初始关系数: {len(graph.relations)}")

        # 准备新实体进行更新
        new_entity = Entity(
            id="entity_new_001",
            name="Windows操作系统",
            entity_type=EntityType.PRODUCT,
            description="微软公司开发的操作系统",
        )

        # 更新图谱
        updated_graph = await builder.update_graph(
            graph=graph,
            new_entities=[new_entity],
        )

        print("\n🔄 图谱更新完成:")
        print(f"   - 更新后实体数: {len(updated_graph.entities)}")
        print(f"   - 更新后关系数: {len(updated_graph.relations)}")

    except Exception as e:
        print(f"❌ 可更新构建器示例失败: {e}")


async def example_streaming_builder():
    """
    示例3: 流式构建器

    适用场景：
    - 实时处理流式到达的文档
    - 需要文档级别的增删操作
    - 追求增量处理能力
    """
    print("\n" + "=" * 50)
    print("🌊 示例3: LLM流式构建器")
    print("=" * 50)

    # 创建流式构建器
    builder = StreamingLLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,
    )

    # 初始文档
    initial_docs = ["谷歌公司是一家美国跨国技术公司。", "拉里·佩奇和谢尔盖·布林创立了谷歌。"]

    try:
        # 构建初始图谱
        graph = await builder.build_graph(texts=initial_docs, graph_name="streaming_example_graph")

        print("✅ 流式图谱初始化:")
        print(f"   - 初始实体数: {len(graph.entities)}")

        # 模拟新文档到达
        new_documents = ["YouTube是谷歌旗下的视频分享平台。", "Android是谷歌开发的移动操作系统。"]

        # 增量添加文档
        updated_graph = await builder.add_documents_async(
            documents=new_documents, document_ids=["doc_youtube", "doc_android"]
        )

        print("\n📄 新文档处理完成:")
        print(f"   - 更新后实体数: {len(updated_graph.entities)}")

        # 查看文档注册表
        registry = builder.get_document_registry()
        print("\n📚 文档注册表:")
        for doc_id, entity_ids in registry.items():
            print(f"   - {doc_id}: {len(entity_ids)} 个实体")

    except Exception as e:
        print(f"❌ 流式构建器示例失败: {e}")


async def example_batch_builder():
    """
    示例4: 批量构建器

    适用场景：
    - 需要处理大量文档
    - 需要合并多个数据源
    - 追求高性能批量处理
    """
    print("\n" + "=" * 50)
    print("📦 示例4: LLM批量构建器")
    print("=" * 50)

    # 创建批量构建器
    builder = BatchLLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        max_concurrent=8,  # 高并发批量处理
    )

    # 大量文档示例
    batch_texts = [
        "特斯拉是一家美国电动汽车制造商。",
        "埃隆·马斯克是特斯拉的CEO。",
        "Model S是特斯拉的豪华电动轿车。",
        "Autopilot是特斯拉的自动驾驶技术。",
        "Gigafactory是特斯拉的电池工厂。",
    ]

    try:
        # 批量构建图谱
        graph = await builder.build_graph(texts=batch_texts, graph_name="batch_example_graph")

        print("✅ 批量图谱构建完成:")
        print(f"   - 处理文档数: {len(batch_texts)}")
        print(f"   - 生成实体数: {len(graph.entities)}")
        print(f"   - 生成关系数: {len(graph.relations)}")

        # 演示多源合并
        sources = [
            {"type": "text", "data": ["亚马逊是全球最大的电子商务公司。"]},
            {"type": "text", "data": ["杰夫·贝佐斯创立了亚马逊公司。"]},
        ]

        merged_graph = await builder.build_from_multiple_sources(sources=sources, graph_name="multi_source_graph")

        print("\n🔗 多源合并完成:")
        print(f"   - 合并后实体数: {len(merged_graph.entities)}")

    except Exception as e:
        print(f"❌ 批量构建器示例失败: {e}")


async def example_full_featured_builder():
    """
    示例5: 全功能构建器

    适用场景：
    - 需要所有功能：构建、更新、合并、验证、导出
    - 企业级应用，需要完整功能
    - 可以承受额外的复杂性和依赖
    """
    print("\n" + "=" * 50)
    print("🏆 示例5: LLM全功能构建器")
    print("=" * 50)

    # 创建全功能构建器
    builder = LLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        max_concurrent=10,
        vector_storage=JsonVectorStorage(file_path="workdir/isp_vector_store.json"),  # 假设使用JSON存储作为向量存储
    )

    # 复杂文档
    complex_texts = [
        "阿里巴巴集团是中国的一家跨国技术公司。",
        "马云是阿里巴巴集团的创始人之一。",
        "淘宝是阿里巴巴旗下的在线购物平台。",
        "支付宝是阿里巴巴的数字支付平台。",
    ]

    try:
        # 构建图谱（自动包含嵌入和验证）
        graph = await builder.build_graph(texts=complex_texts, graph_name="full_featured_graph")

        print("✅ 全功能图谱构建完成:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")

        # 验证图谱
        validation_result = await builder.validate_graph(graph)
        print("\n🔍 图谱验证结果:")
        print(f"   - 验证通过: {validation_result.get('valid', False)}")
        if validation_result.get("issues"):
            print(f"   - 发现问题: {len(validation_result['issues'])} 个")

        # 导出图谱
        exported_data = await builder.export_to_format(graph, "json")
        print("\n📤 图谱导出:")
        print("   - 导出格式: JSON")
        print(f"   - 数据键: {list(exported_data.keys())}")
        with open("workdir/full_featured_graph.json", "w", encoding="utf8") as f:
            import json

            json.dump(exported_data, f, ensure_ascii=False, indent=2)
        # 获取详细统计
        detailed_stats = await builder.get_detailed_statistics(graph)
        print("\n📊 详细统计:")
        for key, value in detailed_stats.items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value}")

        # 打印使用摘要
        builder.print_usage_summary()

        # 清理资源
        builder.cleanup()

    except Exception as e:
        print(f"❌ 全功能构建器示例失败: {e}")


async def example_search_builder():
    """
    示例6: LLM搜索构建器

    适用场景：
    - 需要对已构建的图谱进行智能搜索
    - 基于问题进行检索和问答
    - 只需要搜索功能，不需要构建功能
    """
    print("\n" + "=" * 50)
    print("🔍 示例6: LLM搜索构建器")
    print("=" * 50)

    try:
        # 首先创建一个图谱用于搜索演示
        print("🏗️ 先构建一个示例图谱...")
        builder = FlexibleLLMGraphBuilder(
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            llm_model=settings.LLM_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            vector_storage=JsonVectorStorage(file_path="workdir/search_example_vectors.json"),
        )

        # 构建一个包含多种实体的知识图谱
        knowledge_texts = [
            "OpenAI是一家人工智能研究公司，成立于2015年。",
            "GPT-4是OpenAI开发的大型语言模型，具有强大的自然语言处理能力。",
            "ChatGPT是基于GPT模型的对话系统，能够进行智能对话。",
            "深度学习是机器学习的一个分支，使用神经网络进行学习。",
            "Transformer是一种神经网络架构，被广泛用于自然语言处理任务。",
            "BERT是Google开发的预训练语言模型，擅长理解上下文。",
            "机器学习是人工智能的核心技术，通过数据学习规律。",
            "神经网络是模拟人脑神经元工作方式的计算模型。",
        ]

        graph = await builder.build_graph(texts=knowledge_texts, graph_name="search_example_graph")

        print("✅ 图谱构建完成:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")

        # 创建搜索构建器
        search_builder = LLMSearchBuilder(
            graph=graph,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            llm_model=settings.LLM_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            vector_storage=JsonVectorStorage(file_path="workdir/search_vectors.json"),
        )

        print("\n🔍 开始搜索演示...")

        # 1. 实体搜索示例
        print("\n📋 实体搜索示例:")
        entity_query = "人工智能和机器学习"
        entities = await search_builder.search_entities(query=entity_query, top_k=3, similarity_threshold=0.6)

        print(f"   查询: '{entity_query}'")
        print(f"   找到 {len(entities)} 个相关实体:")
        for i, entity in enumerate(entities[:3], 1):
            print(f"   {i}. {entity['entity_name']} ({entity['entity_type']})")
            print(f"      相似度: {entity['similarity_score']:.3f}")
            print(f"      描述: {entity['description'][:100]}...")

        # 2. 关系搜索示例
        print("\n🔗 关系搜索示例:")
        relation_query = "开发和创建"
        relations = await search_builder.search_relations(query=relation_query, top_k=3, similarity_threshold=0.6)

        print(f"   查询: '{relation_query}'")
        print(f"   找到 {len(relations)} 个相关关系:")
        for i, relation in enumerate(relations[:3], 1):
            head = relation["head_entity"]["name"] if relation["head_entity"] else "Unknown"
            tail = relation["tail_entity"]["name"] if relation["tail_entity"] else "Unknown"
            print(f"   {i}. {head} --({relation['relation_type']})--> {tail}")
            print(f"      相似度: {relation['similarity_score']:.3f}")

        # 3. 综合搜索示例
        print("\n🎯 综合搜索示例:")
        search_results = await search_builder.search_graph(
            query="深度学习和神经网络的关系", search_type="hybrid", top_k=5
        )

        print("   查询: '深度学习和神经网络的关系'")
        print(f"   实体结果: {len(search_results['entities'])} 个")
        print(f"   关系结果: {len(search_results['relations'])} 个")

        # 4. 智能问答示例
        print("\n🤖 智能问答示例:")
        questions = [
            "什么是GPT-4？",
            "OpenAI开发了哪些产品？",
            "深度学习和机器学习有什么关系？",
            "Transformer架构有什么用途？",
        ]

        for i, question in enumerate(questions[:2], 1):  # 只演示前2个问题
            print(f"\n   问题 {i}: {question}")
            answer_result = await search_builder.answer_question(
                question=question, context_entities=3, context_relations=2, include_reasoning=True
            )

            print(f"   回答: {answer_result['answer']}")
            print(f"   置信度: {answer_result['confidence']:.3f}")
            print(f"   使用实体: {answer_result['context']['entities_used']} 个")
            print(f"   使用关系: {answer_result['context']['relations_used']} 个")

            if answer_result.get("reasoning"):
                print(f"   推理过程: {answer_result['reasoning'][:200]}...")

        # 5. 统计信息
        print("\n📊 搜索构建器统计:")
        stats = search_builder.get_statistics()
        print(f"   - 图谱名称: {stats['graph_name']}")
        print(f"   - 实体总数: {stats['entities_count']}")
        print(f"   - 关系总数: {stats['relations_count']}")
        print(f"   - LLM模型: {stats['llm_model']}")
        print(f"   - 搜索能力: {', '.join(stats['search_capabilities'])}")

        # 6. 导出功能演示
        print("\n📤 导出功能演示:")
        summary = await search_builder.export_to_format(graph, "summary")
        print("   - 图谱摘要已生成")
        print(f"   - 实体类型: {len(summary['entity_types'])} 种")
        print(f"   - 关系类型: {len(summary['relation_types'])} 种")

        # 清理资源
        search_builder.cleanup()
        builder.cleanup()

        print("\n✅ 搜索构建器示例完成!")

    except Exception as e:
        print(f"❌ 搜索构建器示例失败: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """
    运行所有示例
    """
    print("🎯 LLM ISP构建器示例集合")
    print("演示Interface Segregation Principle在LLM图构建中的应用")

    # 运行各个示例
    # await example_basic_builder()
    # await example_updatable_builder()
    # await example_streaming_builder()
    # await example_batch_builder()
    # await example_full_featured_builder()
    await example_search_builder()

    print("\n🎉 所有示例运行完成!")


if __name__ == "__main__":
    asyncio.run(main())
