"""
ChromaVectorStore 使用示例。

演示如何使用基于ChromaDB的向量存储来管理知识图谱数据。

安装依赖:
    pip install 'agraph[vectordb]'
    或者
    pip install chromadb>=0.5.0

运行示例:
    python examples/chroma_vectorstore_example.py
"""

import asyncio
from pathlib import Path
from typing import List

try:
    from agraph.base.clusters import Cluster
    from agraph.base.entities import Entity
    from agraph.base.relations import Relation
    from agraph.base.text import TextChunk
    from agraph.vectordb import ChromaVectorStore

    CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"ChromaDB not available: {e}")
    print("Install with: pip install 'agraph[vectordb]' or pip install chromadb>=0.5.0")
    CHROMADB_AVAILABLE = False


async def basic_usage_example():
    """基本使用示例"""
    print("=== ChromaVectorStore 基本使用示例 ===")

    # 创建向量存储实例（内存模式）
    vector_store = ChromaVectorStore(
        collection_name="example_kg",
        persist_directory=None,  # None表示内存模式，可以设置路径进行持久化
    )

    async with vector_store as store:
        # 创建示例实体
        entities = [
            Entity(name="Python", description="一种编程语言", entity_type="CONCEPT"),
            Entity(name="机器学习", description="人工智能的分支", entity_type="CONCEPT"),
            Entity(name="OpenAI", description="人工智能公司", entity_type="ORGANIZATION"),
        ]

        # 批量添加实体
        print("添加实体...")
        results = await store.batch_add_entities(entities)
        print(f"添加结果: {results}")

        # 搜索实体
        print("\n搜索 'Python' 相关实体:")
        search_results = await store.search_entities("Python", top_k=3)
        for entity, score in search_results:
            print(f"  - {entity.name}: {entity.description} (相似度: {score:.3f})")

        # 获取统计信息
        stats = await store.get_stats()
        print(f"\n统计信息: {stats}")


async def advanced_usage_example():
    """高级使用示例"""
    print("\n=== ChromaVectorStore 高级使用示例 ===")

    # 创建持久化存储
    persist_dir = Path("./chroma_data")
    vector_store = ChromaVectorStore(collection_name="advanced_kg", persist_directory=str(persist_dir))

    async with vector_store as store:
        # 创建实体
        python_entity = Entity(
            name="Python",
            description="高级编程语言",
            entity_type="CONCEPT",
            properties={"type": "programming_language", "paradigm": "multi-paradigm"},
        )

        ai_entity = Entity(
            name="人工智能",
            description="模拟人类智能的技术",
            entity_type="CONCEPT",
            properties={"field": "computer_science", "applications": ["nlp", "cv", "ml"]},
        )

        # 添加实体
        await store.add_entity(python_entity)
        await store.add_entity(ai_entity)

        # 创建关系
        relation = Relation(
            head_entity=python_entity,
            tail_entity=ai_entity,
            relation_type="USED_FOR",
            description="Python常用于人工智能开发",
        )

        await store.add_relation(relation)

        # 创建文本块
        text_chunk = TextChunk(
            content="Python是一种广泛用于人工智能和机器学习的编程语言。",
            title="Python与AI",
            chunk_type="paragraph",
            entities={python_entity.id, ai_entity.id},
            relations={relation.id},
        )

        await store.add_text_chunk(text_chunk)

        # 创建聚类
        cluster = Cluster(
            name="编程与AI聚类",
            description="编程语言和人工智能相关概念",
            cluster_type="SEMANTIC",
            entities={python_entity.id, ai_entity.id},
            relations={relation.id},
        )

        await store.add_cluster(cluster)

        # 混合搜索
        print("混合搜索 '人工智能':")
        hybrid_results = await store.hybrid_search(
            query="人工智能", search_types={"entity", "relation", "cluster", "text_chunk"}, top_k=5
        )

        for data_type, results in hybrid_results.items():
            print(f"\n{data_type}结果:")
            for obj, score in results:
                if hasattr(obj, "name"):
                    print(f"  - {obj.name} (相似度: {score:.3f})")
                elif hasattr(obj, "title"):
                    print(f"  - {obj.title} (相似度: {score:.3f})")
                else:
                    print(f"  - {obj.id} (相似度: {score:.3f})")

        # 使用过滤器搜索
        print(f"\n搜索实体类型为 'CONCEPT' 的实体:")
        filtered_results = await store.search_entities(query="编程", top_k=10, filter_dict={"entity_type": "CONCEPT"})

        for entity, score in filtered_results:
            print(f"  - {entity.name}: {entity.entity_type} (相似度: {score:.3f})")


async def persistence_example():
    """持久化示例"""
    print("\n=== 持久化存储示例 ===")

    persist_dir = Path("./chroma_persistent")

    # 第一次运行：创建数据
    print("创建持久化数据...")
    vector_store1 = ChromaVectorStore(collection_name="persistent_kg", persist_directory=str(persist_dir))

    async with vector_store1 as store:
        entity = Entity(name="持久化测试", description="测试持久化功能")
        await store.add_entity(entity)

        stats = await store.get_stats()
        print(f"第一次运行统计: {stats}")

    # 第二次运行：读取数据
    print("\n重新加载持久化数据...")
    vector_store2 = ChromaVectorStore(collection_name="persistent_kg", persist_directory=str(persist_dir))

    async with vector_store2 as store:
        stats = await store.get_stats()
        print(f"第二次运行统计: {stats}")

        # 搜索之前添加的数据
        results = await store.search_entities("持久化", top_k=5)
        print("搜索结果:")
        for entity, score in results:
            print(f"  - {entity.name}: {entity.description}")


async def custom_embedding_example():
    """自定义嵌入函数示例"""
    print("\n=== 自定义嵌入函数示例 ===")

    # 注意：这里使用模拟的嵌入函数
    # 实际使用时可以集成OpenAI、HuggingFace等嵌入模型
    class MockEmbeddingFunction:
        """模拟嵌入函数"""

        def __call__(self, texts: List[str]) -> List[List[float]]:
            """简单的文本嵌入模拟"""
            embeddings = []
            for text in texts:
                # 基于字符的简单嵌入（仅用于演示）
                embedding = [float(ord(c) % 10) for c in text[:10].ljust(10, " ")]
                embeddings.append(embedding)
            return embeddings

    # 使用自定义嵌入函数
    vector_store = ChromaVectorStore(collection_name="custom_embedding_kg", embedding_function=MockEmbeddingFunction())

    async with vector_store as store:
        entity = Entity(name="自定义嵌入", description="测试自定义嵌入函数")
        await store.add_entity(entity)

        results = await store.search_entities("自定义", top_k=3)
        print("使用自定义嵌入函数的搜索结果:")
        for entity, score in results:
            print(f"  - {entity.name} (相似度: {score:.3f})")


async def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    vector_store = ChromaVectorStore(collection_name="error_test")

    # 未初始化时的错误处理
    try:
        entity = Entity(name="测试实体")
        await vector_store.add_entity(entity)
    except Exception as e:
        print(f"未初始化错误: {e}")

    # 正确的使用方式
    async with vector_store as store:
        entity = Entity(name="正确的实体")
        result = await store.add_entity(entity)
        print(f"正确添加实体: {result}")


async def main():
    """主函数"""
    if not CHROMADB_AVAILABLE:
        print("ChromaDB 不可用，请安装后重试")
        return

    try:
        await basic_usage_example()
        await advanced_usage_example()
        await persistence_example()
        await custom_embedding_example()
        await error_handling_example()

        print("\n=== 示例完成 ===")
        print("提示: 可以查看生成的 chroma_data/ 和 chroma_persistent/ 目录")

    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
