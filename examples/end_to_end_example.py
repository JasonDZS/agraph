#!/usr/bin/env python3
"""
端到端示例：使用 KnowledgeGraphBuilder 构建知识图谱

此示例展示了如何使用 agraph 包处理 examples/documents/ 文件夹中的文档，
构建完整的知识图谱，并展示结果。
"""
import os
import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv(project_root / ".env")

from agraph.builder.builder import KnowledgeGraphBuilder
from agraph.base import KnowledgeGraph
from agraph.config import BuilderConfig, settings
from agraph.vectordb import ChromaVectorStore


def main():
    """主函数：演示端到端知识图谱构建过程"""

    print("🚀 启动端到端知识图谱构建示例")
    print("=" * 50)

    # 1. 设置文档目录
    documents_dir = project_root / "examples" / "documents"
    settings.workdir = project_root / "workdir" / "end_to_end" # 设置工作目录
    os.makedirs(settings.workdir, exist_ok=True)

    if not documents_dir.exists():
        print(f"❌ 文档目录不存在: {documents_dir}")
        return

    # 获取所有文档文件
    document_files = list(documents_dir.glob("*"))
    print(f"📄 发现 {len(document_files)} 个文档文件:")
    for doc in document_files:
        print(f"   - {doc.name}")
    print()

    # 2. 配置构建器
    print("⚙️ 配置知识图谱构建器...")

    print(f"   使用API Base: {settings.openai.api_base}")
    print(f"   使用模型: {settings.llm.model}")

    config = BuilderConfig(
        # 基本配置
        chunk_size=1000,
        chunk_overlap=200,

        # LLM配置 - 使用环境变量
        llm_provider="openai",  # 使用OpenAI兼容接口
        llm_model=settings.llm.model,

        # 置信度阈值
        entity_confidence_threshold=0.7,
        relation_confidence_threshold=0.6,

        # 聚类配置
        cluster_algorithm="community_detection",
        min_cluster_size=2,

        # 缓存配置
        cache_dir=str(settings.workdir / "cache")
    )

    # 3. 创建构建器实例
    builder = KnowledgeGraphBuilder(config=config)

    try:
        # 4. 先测试文档处理
        print("🔧 开始构建知识图谱...")
        print("   这个过程可能需要几分钟时间，请耐心等待...")
        print()

        knowledge_graph = builder.build_from_documents(
            documents=document_files,
            graph_name="公司知识图谱",
            graph_description="基于公司文档构建的综合知识图谱",
            use_cache=True
        )

        print("✅ 知识图谱构建完成！")
        print()

        # 5. 展示构建结果
        display_results(knowledge_graph, builder)

        # 6. 展示增量更新功能
        demonstrate_incremental_updates(builder, document_files)

        # 7. 保存知识图谱
        asyncio.run(save_knowledge_graph(knowledge_graph))

        print("💾 知识图谱已保存到工作目录")
        demonstrate_graph_operations(knowledge_graph)

    except Exception as e:
        print(f"❌ 构建过程中出现错误: {e}")
        print("请检查:")
        print("1. 是否正确配置了LLM API密钥")
        print("2. 网络连接是否正常")
        print("3. 文档文件是否可读")

        # 显示构建状态
        build_status = builder.get_build_status()
        print(f"\n当前构建状态: {build_status}")


def display_results(kg, builder):
    """展示知识图谱构建结果"""

    print("📊 知识图谱构建结果:")
    print("-" * 30)
    print(f"图谱名称: {kg.name}")
    print(f"图谱描述: {kg.description}")
    print()

    # 统计信息 - 直接访问字典属性
    entities = list(kg.entities.values())
    relations = list(kg.relations.values())
    clusters = list(kg.clusters.values())
    chunks = list(kg.text_chunks.values())

    print(f"📋 统计信息:")
    print(f"   - 实体数量: {len(entities)}")
    print(f"   - 关系数量: {len(relations)}")
    print(f"   - 聚类数量: {len(clusters)}")
    print(f"   - 文本块数量: {len(chunks)}")
    print()

    # 展示部分实体
    if entities:
        print("🏷️  提取的实体样例 (前10个):")
        for i, entity in enumerate(entities[:10]):
            print(f"   {i+1}. {entity.name} ({entity.entity_type}) - 置信度: {entity.confidence:.2f}")
        if len(entities) > 10:
            print(f"   ... 还有 {len(entities) - 10} 个实体")
        print()

    # 展示部分关系
    if relations:
        print("🔗 提取的关系样例 (前10个):")
        for i, relation in enumerate(relations[:10]):
            # 使用正确的属性名
            source_name = relation.head_entity.name if relation.head_entity else "未知"
            target_name = relation.tail_entity.name if relation.tail_entity else "未知"
            print(f"   {i+1}. {source_name} --[{relation.relation_type}]--> {target_name} "
                  f"(置信度: {relation.confidence:.2f})")
        if len(relations) > 10:
            print(f"   ... 还有 {len(relations) - 10} 个关系")
        print()

    # 展示聚类信息
    if clusters:
        print("🎯 聚类信息:")
        for i, cluster in enumerate(clusters[:5]):
            entities_in_cluster = len(cluster.entities) if hasattr(cluster, 'entities') else 0
            print(f"   聚类 {i+1}: {cluster.name} (包含 {entities_in_cluster} 个实体)")
        if len(clusters) > 5:
            print(f"   ... 还有 {len(clusters) - 5} 个聚类")
        print()

    # 显示缓存信息
    cache_info = builder.get_cache_info()
    backend_info = cache_info.get('backend', {})
    print(f"💾 缓存信息:")
    print(f"   - 缓存目录: {backend_info.get('cache_dir', 'N/A')}")
    print(f"   - 缓存文件数: {backend_info.get('total_files', 0)}")
    print(f"   - 文档处理缓存: {cache_info.get('document_processing', {}).get('total_documents', 0)} 个文档已缓存")


async def save_knowledge_graph(kg: KnowledgeGraph):
    """保存知识图谱到文件"""

    output_dir = settings.workdir / "output"
    output_dir.mkdir(exist_ok=True)
    vectordb = ChromaVectorStore(
        persist_directory= str(output_dir / "vectordb"),
        use_openai_embeddings=True
    )

    try:
        # 导出为JSON格式 (如果KnowledgeGraph支持的话)
        if hasattr(kg, 'export_to_json'):
            json_path = output_dir / "knowledge_graph.json"
            kg.export_to_json(json_path)
            print(f"💾 知识图谱已保存为JSON: {json_path}")

        if hasattr(kg, 'export_to_graphml'):
            graphml_path = output_dir / "knowledge_graph.graphml"
            kg.export_to_graphml(graphml_path)
            print(f"💾 知识图谱已保存为GraphML: {graphml_path}")

        # 导出实体和关系信息
        entities_path = output_dir / "entities.txt"
        with open(entities_path, 'w', encoding='utf-8') as f:
            f.write("提取的实体:\n")
            f.write("=" * 50 + "\n")
            for entity in kg.entities.values():
                f.write(f"{entity.name} ({entity.entity_type}) - 置信度: {entity.confidence:.2f}\n")

        relations_path = output_dir / "relations.txt"
        with open(relations_path, 'w', encoding='utf-8') as f:
            f.write("提取的关系:\n")
            f.write("=" * 50 + "\n")
            for relation in kg.relations.values():
                # 使用正确的属性名
                source_name = relation.head_entity.name if relation.head_entity else "未知"
                target_name = relation.tail_entity.name if relation.tail_entity else "未知"
                f.write(f"{source_name} --[{relation.relation_type}]--> {target_name} "
                       f"(置信度: {relation.confidence:.2f})\n")

        print(f"📝 实体信息已保存到: {entities_path}")
        print(f"📝 关系信息已保存到: {relations_path}")
        print()

    except Exception as e:
        print(f"⚠️  保存文件时出现错误: {e}")

    try:
        # 保存向量数据库
        await vectordb.initialize()
        await vectordb.batch_add_entities(kg.entities.values())
        await vectordb.batch_add_relations(kg.relations.values())
        await vectordb.batch_add_clusters(kg.clusters.values())
        await vectordb.batch_add_text_chunks(kg.text_chunks.values())
        await vectordb.close()

        print(f"🔗 向量数据库已保存到: {vectordb.persist_directory}")
    except Exception as e:
        print(f"⚠️ 保存向量数据库时出现错误: {e}")


def demonstrate_incremental_updates(builder, document_files):
    """演示增量更新功能"""

    print("🔄 增量更新功能演示:")
    print("-" * 30)

    # 显示文档处理状态
    print("📊 文档处理状态总结:")
    doc_status = builder.get_document_processing_status()
    print(f"   - 总文档数: {doc_status.get('total_documents', 0)}")
    print(f"   - 已完成: {doc_status.get('completed', 0)}")
    print(f"   - 失败: {doc_status.get('failed', 0)}")
    print(f"   - 总处理时间: {doc_status.get('total_processing_time', 0):.4f} 秒")
    print()

    # 显示各个文档的状态
    print("📋 各文档处理状态:")
    for i, doc_path in enumerate(document_files[:5]):  # 只显示前5个
        status = builder.get_document_processing_status(doc_path)
        if status:
            print(f"   {i+1}. {doc_path.name}:")
            print(f"      状态: {status.get('processing_status', 'unknown')}")
            print(f"      处理时间: {status.get('processing_time', 0):.4f}s")
            print(f"      文件哈希: {status.get('file_hash', 'N/A')[:16]}...")

    if len(document_files) > 5:
        print(f"   ... 还有 {len(document_files) - 5} 个文档")
    print()

    # 获取缓存信息
    cache_info = builder.get_cache_info()
    print("💾 详细缓存信息:")
    backend_info = cache_info.get('backend', {})
    print(f"   - 缓存目录: {backend_info.get('cache_dir', 'N/A')}")
    print(f"   - 总缓存文件: {backend_info.get('total_files', 0)}")
    print(f"   - 缓存总大小: {backend_info.get('total_size', 0)} 字节")

    doc_processing_info = cache_info.get('document_processing', {})
    if doc_processing_info:
        print(f"   - 文档缓存统计: {doc_processing_info}")
    print()

    print("💡 提示:")
    print("   - 再次运行此脚本时，已处理的文档将使用缓存，大大提高速度")
    print("   - 如果文档被修改，系统会自动检测并重新处理")
    print("   - 使用 builder.force_reprocess_document(path) 可强制重新处理特定文档")
    print("   - 使用 builder.clear_document_cache() 可清除所有文档缓存")
    print()


def demonstrate_graph_operations(kg: KnowledgeGraph):
    """演示知识图谱的基本操作"""

    print("🔍 知识图谱操作演示:")
    print("-" * 30)

    # 搜索实体
    if hasattr(kg, 'search_entities'):
        print("搜索包含'公司'的实体:")
        company_entities = kg.search_entities("公司")
        for entity in company_entities[:5]:
            print(f"   - {entity.name}")
        print()

    vectordb = ChromaVectorStore(
        persist_directory=str(settings.workdir / "output" / "vectordb"),
        use_openai_embeddings=True
    )
    try:
        asyncio.run(vectordb.initialize())
        print("🔗 向量数据库已初始化")
        results = asyncio.run(vectordb.search_entities("公司", top_k=5))
        print("搜索向量数据库中的实体:")
        for entity, score in results:
            print(f"   - {entity.name} ({entity.entity_type}) - 置信度: {score:.2f}")
        results = asyncio.run(vectordb.search_relations("公司", top_k=5))
        print("搜索向量数据库中的关系:")
        for relation, score in results:
            source_name = relation.head_entity.name if relation.head_entity else "未知"
            target_name = relation.tail_entity.name if relation.tail_entity else "未知"
            print(f"   - {source_name} --[{relation.relation_type}]--> {target_name} "
                  f"(置信度: {score:.2f})")
        results = asyncio.run(vectordb.search_text_chunks("公司", top_k=5))
        print("搜索向量数据库中的文本块:")
        for chunk, score in results:
            print(f"   - {chunk.content[:50]}... (置信度: {score:.2f})")

        asyncio.run(vectordb.close())
    except Exception as e:
        print(f"⚠️ 向量数据库初始化失败: {e}")

if __name__ == "__main__":
    main()
