#!/usr/bin/env python3
"""
AGraph 示例：统一的知识图谱构建、向量存储和对话系统

此示例展示了如何使用AGraph类的完整功能：
1. 从文档构建知识图谱
2. 保存到向量存储
3. 进行知识库检索和对话
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv(project_root / ".env")

from agraph import AGraph


async def main():
    """主函数：演示AGraph的完整功能"""

    print("🚀 启动AGraph统一知识图谱系统示例")
    print("=" * 50)

    # 1. 初始化AGraph系统
    print("⚙️ 初始化AGraph系统...")

    # 设置工作目录
    workdir = project_root / "workdir" / "agraph_demo"
    os.makedirs(workdir, exist_ok=True)

    # 创建AGraph实例
    agraph = AGraph(
        collection_name="demo_knowledge_graph",
        persist_directory=str(workdir / "vectordb"),
        vector_store_type="chroma",
        use_openai_embeddings=True
    )

    async with agraph:
        print("✅ AGraph系统初始化成功")
        print()

        # 2. 从文档构建知识图谱
        await demo_knowledge_graph_building(agraph)

        # 3. 演示检索功能
        await demo_search_functionality(agraph)

        # 4. 演示对话功能
        await demo_chat_functionality(agraph)

        # 5. 显示系统统计
        await demo_system_stats(agraph)


async def demo_knowledge_graph_building(agraph: AGraph):
    """演示知识图谱构建功能"""
    print("📚 知识图谱构建演示")
    print("-" * 30)

    # 准备示例文档目录
    documents_dir = project_root / "examples" / "documents"

    if documents_dir.exists():
        # 从文档构建
        document_files = list(documents_dir.glob("*"))
        print(f"📄 发现 {len(document_files)} 个文档文件")

        if document_files:
            knowledge_graph = agraph.build_from_documents(
                documents=document_files,
                graph_name="示例知识图谱",
                graph_description="基于示例文档构建的知识图谱",
                use_cache=True,
                save_to_vector_store=True
            )

            print(f"✅ 知识图谱构建完成:")
            print(f"   - 实体数量: {len(knowledge_graph.entities)}")
            print(f"   - 关系数量: {len(knowledge_graph.relations)}")
            print(f"   - 文本块数量: {len(knowledge_graph.text_chunks)}")
        else:
            print("📝 未找到文档文件，使用文本构建示例...")
            await demo_text_building(agraph)
    else:
        print("📝 文档目录不存在，使用文本构建示例...")
        await demo_text_building(agraph)

    print()


async def demo_text_building(agraph: AGraph):
    """从文本构建知识图谱示例"""
    sample_texts = [
        "苹果公司是一家美国的跨国科技公司，总部位于加利福尼亚州库比蒂诺。",
        "史蒂夫·乔布斯是苹果公司的联合创始人，他在2007年推出了iPhone。",
        "iPhone是苹果公司生产的智能手机产品系列，改变了移动通信行业。",
        "微软公司是苹果公司的主要竞争对手之一，两家公司都专注于消费电子产品。"
    ]

    print("使用示例文本构建知识图谱...")
    knowledge_graph = agraph.build_from_texts(
        texts=sample_texts,
        graph_name="示例文本知识图谱",
        graph_description="基于示例文本构建的知识图谱"
    )

    print(f"✅ 文本知识图谱构建完成:")
    print(f"   - 实体数量: {len(knowledge_graph.entities)}")
    print(f"   - 关系数量: {len(knowledge_graph.relations)}")
    print(f"   - 文本块数量: {len(knowledge_graph.text_chunks)}")


async def demo_search_functionality(agraph: AGraph):
    """演示检索功能"""
    print("🔍 检索功能演示")
    print("-" * 30)

    if not agraph.has_knowledge_graph:
        print("⚠️ 没有可用的知识图谱，跳过检索演示")
        return

    # 演示实体搜索
    print("1. 搜索实体 '公司':")
    try:
        entity_results = await agraph.search_entities("公司", top_k=3)
        for i, (entity, score) in enumerate(entity_results):
            print(f"   {i+1}. {entity.name} ({entity.entity_type}) - 相似度: {score:.3f}")
    except Exception as e:
        print(f"   实体搜索失败: {e}")

    # 演示关系搜索
    print("\n2. 搜索关系 '创始人':")
    try:
        relation_results = await agraph.search_relations("创始人", top_k=3)
        for i, (relation, score) in enumerate(relation_results):
            head_name = relation.head_entity.name if relation.head_entity else "未知"
            tail_name = relation.tail_entity.name if relation.tail_entity else "未知"
            print(f"   {i+1}. {head_name} --[{relation.relation_type}]--> {tail_name} - 相似度: {score:.3f}")
    except Exception as e:
        print(f"   关系搜索失败: {e}")

    # 演示文本块搜索
    print("\n3. 搜索文本块 'iPhone':")
    try:
        chunk_results = await agraph.search_text_chunks("iPhone", top_k=3)
        for i, (chunk, score) in enumerate(chunk_results):
            preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            print(f"   {i+1}. {preview} - 相似度: {score:.3f}")
    except Exception as e:
        print(f"   文本块搜索失败: {e}")

    print()


async def demo_chat_functionality(agraph: AGraph):
    """演示对话功能"""
    print("💬 对话功能演示")
    print("-" * 30)

    if not agraph.has_knowledge_graph:
        print("⚠️ 没有可用的知识图谱，跳过对话演示")
        return

    # 示例对话
    questions = [
        "苹果公司是什么？",
        "谁是苹果公司的创始人？",
        "iPhone什么时候发布的？"
    ]

    conversation_history = []

    for i, question in enumerate(questions):
        print(f"{i+1}. 用户问题: {question}")

        try:
            response = await agraph.chat(
                question=question,
                conversation_history=conversation_history,
                entity_top_k=3,
                text_chunk_top_k=3,
                response_type="简洁回答"
            )

            answer = response["answer"]
            print(f"   助手回答: {answer}")

            # 更新对话历史
            conversation_history.append({
                "user": question,
                "assistant": answer
            })

            # 显示检索到的上下文信息
            context = response["context"]
            if context.get("entities"):
                print(f"   检索实体数: {len(context['entities'])}")
            if context.get("text_chunks"):
                print(f"   检索文档数: {len(context['text_chunks'])}")

        except Exception as e:
            print(f"   对话失败: {e}")

        print()


async def demo_system_stats(agraph: AGraph):
    """演示系统统计信息"""
    print("📊 系统统计信息")
    print("-" * 30)

    try:
        stats = await agraph.get_stats()

        # 显示向量存储统计
        if "vector_store" in stats:
            vs_stats = stats["vector_store"]
            print("向量存储统计:")
            for key, value in vs_stats.items():
                print(f"   - {key}: {value}")

        # 显示知识图谱统计
        if "knowledge_graph" in stats:
            kg_stats = stats["knowledge_graph"]
            print("\n知识图谱统计:")
            for key, value in kg_stats.items():
                print(f"   - {key}: {value}")

        # 显示构建器统计
        if "builder" in stats:
            builder_stats = stats["builder"]
            print("\n构建器统计:")
            if "build_status" in builder_stats:
                build_status = builder_stats["build_status"]
                if hasattr(build_status, 'to_dict'):
                    build_dict = build_status.to_dict()
                    for key, value in build_dict.items():
                        print(f"   - {key}: {value}")

    except Exception as e:
        print(f"获取统计信息失败: {e}")

    print()
    print("✅ AGraph示例演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
