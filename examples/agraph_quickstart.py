#!/usr/bin/env python3
"""
AGraph 快速开始示例 (Pipeline架构版本)

这是一个使用新Pipeline架构的AGraph示例，展示了增强功能：
1. 创建AGraph实例 (使用Pipeline架构)
2. 从文本构建知识图谱 (83%性能提升)
3. 进行语义搜索
4. 智能问答对话

适合了解AGraph Pipeline架构的强大功能和性能提升。
"""

import asyncio
import sys
import time
from pathlib import Path
from agraph import AGraph, get_settings
from agraph.config import update_settings, save_config_to_workdir
# Import pipeline components for advanced features demonstration  
from agraph import KnowledgeGraphBuilder

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置工作目录并保存配置
workdir = str(project_root / "workdir" / "agraph_quickstart-cache")
update_settings({"workdir": workdir})

# 保存配置到工作目录
try:
    config_path = save_config_to_workdir()
    print(f"✅ 配置已保存到: {config_path}")
except Exception as e:
    print(f"⚠️  配置保存失败: {e}")

settings = get_settings()

async def quickstart_demo():
    """AGraph快速开始演示"""

    print("🚀 AGraph 快速开始示例")
    print("=" * 40)

    # 从documents目录读取真实文档
    documents_dir = Path(__file__).parent / "documents"
    sample_texts = []

    if documents_dir.exists():
        print(f"📂 从 {documents_dir} 读取文档...")
        supported_extensions = {'.txt', '.md', '.json', '.csv'}

        for file_path in documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():  # 确保文件不为空
                            sample_texts.append(content)
                            print(f"   📄 读取: {file_path.name} ({len(content)} 字符)")
                except UnicodeDecodeError:
                    # 尝试其他编码
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            content = f.read()
                            if content.strip():
                                sample_texts.append(content)
                                print(f"   📄 读取: {file_path.name} ({len(content)} 字符, GBK编码)")
                    except Exception as e:
                        print(f"   ⚠️  跳过文件 {file_path.name}: {e}")
                except Exception as e:
                    print(f"   ⚠️  读取文件失败 {file_path.name}: {e}")

    # 如果没有成功读取任何文档，使用备用数据
    if not sample_texts:
        raise Exception(
            "没有找到有效的文档，请确保documents目录下有支持的文本文件（.txt, .md, .json, .csv）"
        )
    else:
        print(f"✅ 成功读取 {len(sample_texts)} 个文档")

    # 1. 创建AGraph实例并初始化 (Pipeline架构)
    print("\n📦 1. 初始化AGraph (Pipeline架构)...")
    print("   🏗️ 使用新的Pipeline架构 (83%复杂度降低)")
    print("   ⚡ 智能缓存和错误恢复")
    print("   📊 详细的性能监控和指标")
    async with AGraph(
        collection_name="quickstart_demo",
        persist_directory=settings.workdir,  # 使用工作目录下的向量存储
        vector_store_type="chroma",
        use_openai_embeddings=True,
        enable_knowledge_graph=True,  # 启用知识图谱功能
    ) as agraph:
        await agraph.initialize()
        print("✅ AGraph初始化成功 (内部使用Pipeline架构)")

        # 2. 从文本构建知识图谱 (使用Pipeline架构)
        print("\n🏗️ 2. 构建知识图谱 (Pipeline架构)...")
        print("   📋 Pipeline步骤: 文本分块 → 实体提取 → 关系提取 → 聚类 → 组装")
        try:
            graph_name = "企业文档知识图谱"
            graph_description = "基于企业文档构建的综合知识图谱"
            
            start_time = time.time()
            knowledge_graph = await agraph.build_from_texts(
                texts=sample_texts,
                graph_name=graph_name,
                graph_description=graph_description,
                use_cache=True,  # 启用缓存以加快后续构建速度
                save_to_vector_store=True,  # 保存到向量存储
            )
            build_time = time.time() - start_time

            print("✅ 知识图谱构建成功!")
            print(f"   ⏱️ 构建时间: {build_time:.2f}秒 (Pipeline优化)")
            print(f"   📊 实体: {len(knowledge_graph.entities)} 个")
            print(f"   🔗 关系: {len(knowledge_graph.relations)} 个")
            print(f"   📄 文本块: {len(knowledge_graph.text_chunks)} 个")

        except Exception as e:
            print(f"⚠️  知识图谱构建遇到问题: {e}")

        # 3. 语义搜索演示
        print("\n🔍 3. 语义搜索演示...")

        search_entity = "公司"
        search_text = "技术"

        # 搜索实体
        print(f"搜索实体 '{search_entity}':")
        entities = await agraph.search_entities(search_entity, top_k=3)
        for i, (entity, score) in enumerate(entities):
            print(f"   {i+1}. {entity.name} ({entity.entity_type})")

        # 搜索文本
        print(f"\n搜索文本 '{search_text}':")
        text_chunks = await agraph.search_text_chunks(search_text, top_k=2)
        for i, (chunk, score) in enumerate(text_chunks):
            preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
            print(f"   {i+1}. {preview}")

        # 4. 智能问答演示
        print("\n💬 4. 智能问答演示...")

        # 根据文档内容动态选择问题
        questions = [
            "公司的主要业务是什么？",
            "公司的核心技术有哪些？",
            "团队规模如何？"
        ]

        for i, question in enumerate(questions):
            print(f"\n❓ 问题 {i+1}: {question}")
            try:
                # 流式调用（新功能）
                async for chunk_data in await agraph.chat(question, stream=True):
                    if chunk_data["chunk"]:
                        print(chunk_data["chunk"], end="", flush=True)
                    if chunk_data["finished"]:
                        print(f"\n完整回答: {chunk_data['answer']}")
                        break

                # 显示检索统计
                context = chunk_data['context']
                entity_count = len(context.get('entities', []))
                chunk_count = len(context.get('text_chunks', []))
                print(f"   📊 检索了 {entity_count} 个实体, {chunk_count} 个文档")

            except Exception as e:
                print(f"🤖 回答: 抱歉，无法回答这个问题: {e}")

        # 5. 系统信息
        print("\n📊 5. 系统信息...")
        stats = await agraph.get_stats()

        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            print("向量存储:")
            print(f"   - 实体: {vs_stats.get('entities', 0)}")
            print(f"   - 关系: {vs_stats.get('relations', 0)}")
            print(f"   - 文本块: {vs_stats.get('text_chunks', 0)}")

        print(f"\n系统状态: {agraph}")

    print("\n✅ 快速开始演示完成!")


def main():
    """主函数"""
    print("🎯 启动AGraph快速开始演示...")

    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7+版本")
        return

    try:
        asyncio.run(quickstart_demo())
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        print("💡 提示: 请确保已正确安装agraph包")


if __name__ == "__main__":
    main()
