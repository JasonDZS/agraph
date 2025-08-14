"""
OpenAI 嵌入函数使用示例。

演示如何在向量存储中使用 OpenAI-Compatible API 进行嵌入，
包括异步并发优化和缓存功能。

环境变量配置:
    OPENAI_API_KEY: OpenAI API 密钥
    OPENAI_API_BASE: API 基础地址 (可选，默认为 OpenAI 官方 API)
    EMBEDDING_MODEL: 嵌入模型名称 (可选，默认从配置读取)

运行示例:
    python examples/openai_embeddings_example.py
"""

import asyncio
import os
import time
from typing import List

try:
    from agraph.base.entities import Entity
    from agraph.base.text import TextChunk
    from agraph.vectordb import (
        CachedOpenAIEmbeddingFunction,
        MemoryVectorStore,
        OpenAIEmbeddingFunction,
        create_openai_embedding_function,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Dependencies not available: {e}")
    print("Install with: pip install openai>=1.99.9")
    DEPENDENCIES_AVAILABLE = False

try:
    from agraph.vectordb import ChromaVectorStore

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


async def basic_openai_embeddings_example():
    """基本 OpenAI 嵌入功能示例"""
    print("=== 基本 OpenAI 嵌入功能示例 ===")

    # 检查 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过：需要设置 OPENAI_API_KEY 环境变量")
        return

    # 创建嵌入函数
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model="text-embedding-3-small",  # 或其他兼容模型
        batch_size=50,  # 每批处理 50 个文本
        max_concurrency=5,  # 最大并发请求数
        timeout=30.0,
    )

    try:
        # 单个文本嵌入
        print("单个文本嵌入:")
        text = "Python 是一种高级编程语言"
        embedding = await embedding_fn.embed_single(text)
        print(f"文本: {text}")
        print(f"嵌入维度: {len(embedding)}")
        print(f"前5个值: {embedding[:5]}")

        # 批量文本嵌入
        print("\n批量文本嵌入:")
        texts = [
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络",
            "自然语言处理处理文本数据",
            "计算机视觉处理图像数据",
            "强化学习通过试错学习",
        ]

        start_time = time.time()
        embeddings = await embedding_fn.aembed_texts(texts)
        elapsed_time = time.time() - start_time

        print(f"处理 {len(texts)} 个文本")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"每个文本平均耗时: {elapsed_time/len(texts):.3f} 秒")

        # 显示统计信息
        stats = embedding_fn.get_stats()
        print(f"\n嵌入统计:")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  总文本数: {stats['total_texts']}")
        print(f"  总 Token 数: {stats['total_tokens']}")
        print(f"  平均响应时间: {stats['avg_response_time']:.3f} 秒")
        print(f"  错误数量: {stats['error_count']}")

    finally:
        await embedding_fn.close()


async def cached_embeddings_example():
    """缓存嵌入功能示例"""
    print("\n=== 缓存嵌入功能示例 ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过：需要设置 OPENAI_API_KEY 环境变量")
        return

    # 创建带缓存的嵌入函数
    embedding_fn = CachedOpenAIEmbeddingFunction(
        api_key=api_key, model="text-embedding-3-small", cache_size=100, batch_size=10
    )

    try:
        texts = [
            "Python 编程语言",
            "机器学习算法",
            "数据科学分析",
            "Python 编程语言",  # 重复文本
            "机器学习算法",  # 重复文本
        ]

        print("第一次嵌入（缓存未命中）:")
        start_time = time.time()
        embeddings1 = await embedding_fn.aembed_texts(texts)
        first_time = time.time() - start_time

        print("第二次嵌入（部分缓存命中）:")
        start_time = time.time()
        embeddings2 = await embedding_fn.aembed_texts(texts)
        second_time = time.time() - start_time

        print(f"第一次耗时: {first_time:.3f} 秒")
        print(f"第二次耗时: {second_time:.3f} 秒")
        print(f"加速比: {first_time/second_time:.1f}x")

        # 验证结果一致性
        print(f"结果一致性: {embeddings1 == embeddings2}")

        # 缓存统计
        cache_stats = embedding_fn.get_cache_stats()
        print(f"\n缓存统计:")
        print(f"  缓存大小: {cache_stats['cache_size']}")
        print(f"  缓存命中: {cache_stats['cache_hits']}")
        print(f"  缓存未命中: {cache_stats['cache_misses']}")
        print(f"  命中率: {cache_stats['hit_rate']:.2%}")

    finally:
        await embedding_fn.close()


async def memory_vector_store_with_openai_example():
    """内存向量存储 + OpenAI 嵌入示例"""
    print("\n=== 内存向量存储 + OpenAI 嵌入示例 ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过：需要设置 OPENAI_API_KEY 环境变量")
        return

    # 创建使用 OpenAI 嵌入的向量存储
    vector_store = MemoryVectorStore(
        collection_name="openai_demo",
        use_openai_embeddings=True,
        openai_embedding_config={
            "api_key": api_key,
            "model": "text-embedding-3-small",
            "batch_size": 20,
            "max_concurrency": 3,
            "use_cache": True,
            "cache_size": 50,
        },
    )

    async with vector_store as store:
        # 创建示例实体
        entities = [
            Entity(name="Python", description="高级编程语言，适用于数据科学和机器学习"),
            Entity(name="机器学习", description="人工智能的分支，让计算机从数据中学习"),
            Entity(name="深度学习", description="使用多层神经网络的机器学习方法"),
            Entity(name="自然语言处理", description="让计算机理解和生成人类语言的技术"),
            Entity(name="计算机视觉", description="让计算机理解和分析图像的技术"),
        ]

        print(f"添加 {len(entities)} 个实体...")
        start_time = time.time()

        # 批量添加实体（会自动生成嵌入）
        results = await store.batch_add_entities(entities)

        elapsed_time = time.time() - start_time
        print(f"添加完成，耗时: {elapsed_time:.2f} 秒")
        print(f"成功添加: {sum(results)}/{len(results)} 个实体")

        # 测试搜索
        print("\n搜索测试:")
        queries = ["编程语言", "人工智能", "神经网络", "语言理解"]

        for query in queries:
            print(f"\n查询: '{query}'")
            results = await store.search_entities(query, top_k=3)

            for i, (entity, score) in enumerate(results, 1):
                print(f"  {i}. {entity.name} (相似度: {score:.3f})")
                print(f"     描述: {entity.description}")

        # 显示嵌入统计
        embedding_stats = store.get_embedding_stats()
        if embedding_stats:
            print(f"\n嵌入统计:")
            for key, value in embedding_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")


async def chroma_vector_store_with_openai_example():
    """ChromaDB 向量存储 + OpenAI 嵌入示例"""
    if not CHROMADB_AVAILABLE:
        print("\n跳过 ChromaDB 示例：ChromaDB 未安装")
        return

    print("\n=== ChromaDB 向量存储 + OpenAI 嵌入示例 ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过：需要设置 OPENAI_API_KEY 环境变量")
        return

    # 创建使用 OpenAI 嵌入的 ChromaDB 存储
    vector_store = ChromaVectorStore(
        collection_name="openai_chroma_demo",
        use_openai_embeddings=True,
        openai_embedding_config={
            "api_key": api_key,
            "model": "text-embedding-3-small",
            "use_cache": True,
        },
    )

    async with vector_store as store:
        # 创建文本块示例
        text_chunks = [
            TextChunk(
                title="Python 编程",
                content="Python是一种解释型、高级、通用的编程语言。它的设计哲学强调代码的可读性和简洁的语法。",
            ),
            TextChunk(
                title="机器学习基础",
                content="机器学习是人工智能的一个子集，它使计算机系统能够从数据中自动学习和改进，而无需明确编程。",
            ),
            TextChunk(
                title="深度学习原理",
                content="深度学习是机器学习的一个子集，使用具有多个层次的神经网络来分析各种因素的数据。",
            ),
        ]

        print(f"添加 {len(text_chunks)} 个文本块...")
        results = await store.batch_add_text_chunks(text_chunks)
        print(f"成功添加: {sum(results)}/{len(results)} 个文本块")

        # 测试文本搜索
        print("\n文本搜索测试:")
        query = "什么是机器学习"
        results = await store.search_text_chunks(query, top_k=3)

        for i, (chunk, score) in enumerate(results, 1):
            print(f"{i}. {chunk.title} (相似度: {score:.3f})")
            print(f"   内容: {chunk.content[:50]}...")

        # 混合搜索
        print(f"\n混合搜索: '{query}'")
        hybrid_results = await store.hybrid_search(query=query, search_types={"text_chunk"}, top_k=2)

        for data_type, results in hybrid_results.items():
            print(f"\n{data_type} 结果:")
            for obj, score in results:
                if hasattr(obj, "title"):
                    print(f"  - {obj.title} (相似度: {score:.3f})")
                else:
                    print(f"  - {obj.name} (相似度: {score:.3f})")


async def performance_comparison_example():
    """性能对比示例"""
    print("\n=== 性能对比示例 ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过：需要设置 OPENAI_API_KEY 环境变量")
        return

    # 测试数据
    test_texts = [f"这是测试文本 {i}，用于性能对比测试" for i in range(20)]

    # 测试不同配置
    configs = [
        {"batch_size": 5, "max_concurrency": 1, "name": "小批次串行"},
        {"batch_size": 20, "max_concurrency": 1, "name": "大批次串行"},
        {"batch_size": 5, "max_concurrency": 4, "name": "小批次并发"},
        {"batch_size": 10, "max_concurrency": 2, "name": "中批次适度并发"},
    ]

    print(f"测试 {len(test_texts)} 个文本的嵌入性能:")
    print("-" * 50)

    for config in configs:
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model="text-embedding-3-small",
            batch_size=config["batch_size"],
            max_concurrency=config["max_concurrency"],
            timeout=60.0,
        )

        try:
            start_time = time.time()
            embeddings = await embedding_fn.aembed_texts(test_texts)
            elapsed_time = time.time() - start_time

            stats = embedding_fn.get_stats()

            print(f"{config['name']}:")
            print(f"  配置: batch_size={config['batch_size']}, concurrency={config['max_concurrency']}")
            print(f"  总耗时: {elapsed_time:.2f} 秒")
            print(f"  平均每文本: {elapsed_time/len(test_texts):.3f} 秒")
            print(f"  API 请求数: {stats['total_requests']}")
            print(f"  Token 使用: {stats['total_tokens']}")
            print(f"  平均响应时间: {stats['avg_response_time']:.3f} 秒")
            print()

        except Exception as e:
            print(f"{config['name']}: 测试失败 - {e}")

        finally:
            await embedding_fn.close()


async def error_handling_and_fallback_example():
    """错误处理和回退机制示例"""
    print("\n=== 错误处理和回退机制示例 ===")

    # 测试无效 API 密钥
    print("1. 测试无效 API 密钥:")
    try:
        embedding_fn = OpenAIEmbeddingFunction(
            api_key="invalid-key", model="text-embedding-3-small", max_retries=1, timeout=5.0
        )

        await embedding_fn.embed_single("测试文本")

    except Exception as e:
        print(f"   预期错误: {type(e).__name__}: {e}")
    finally:
        if "embedding_fn" in locals():
            await embedding_fn.close()

    # 测试 MemoryVectorStore 的回退机制
    print("\n2. 测试回退到简单嵌入:")
    vector_store = MemoryVectorStore(
        collection_name="fallback_test",
        use_openai_embeddings=True,
        openai_embedding_config={
            "api_key": "invalid-key",  # 无效密钥，会回退
            "model": "text-embedding-3-small",
        },
    )

    async with vector_store as store:
        entity = Entity(name="测试实体", description="用于测试回退机制")
        result = await store.add_entity(entity)
        print(f"   添加实体结果: {result}")

        # 测试搜索
        search_results = await store.search_entities("测试", top_k=1)
        print(f"   搜索结果数量: {len(search_results)}")
        if search_results:
            entity, score = search_results[0]
            print(f"   找到实体: {entity.name} (相似度: {score:.3f})")


async def main():
    """主函数"""
    if not DEPENDENCIES_AVAILABLE:
        print("依赖不可用，请安装后重试")
        return

    print("OpenAI 嵌入函数示例")
    print("=" * 50)

    try:
        await basic_openai_embeddings_example()
        await cached_embeddings_example()
        await memory_vector_store_with_openai_example()
        await chroma_vector_store_with_openai_example()
        await performance_comparison_example()
        await error_handling_and_fallback_example()

        print("\n=== 示例完成 ===")
        print("\n提示:")
        print("1. 设置 OPENAI_API_KEY 环境变量以启用 OpenAI 嵌入")
        print("2. 可以设置 OPENAI_API_BASE 使用兼容的 API 服务")
        print("3. 根据需要调整 batch_size 和 max_concurrency 优化性能")
        print("4. 使用缓存功能可以显著提高重复查询的性能")

    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
