"""
简化的文件夹处理示例

这是一个简单的示例，展示如何快速从文件夹读取文档并构建知识图谱的核心步骤。
"""

import asyncio
from pathlib import Path

from agraph import create_lightrag_graph_builder
from agraph.processer import can_process, process_document


async def simple_folder_to_knowledge_graph(folder_path: str, output_dir: str = "./workdir/simple_example"):
    """从文件夹简单快速地构建知识图谱

    Args:
        folder_path: 文档文件夹路径
        output_dir: 输出目录
    """

    print(f"📁 扫描文件夹: {folder_path}")

    # 1. 扫描并处理所有支持的文件
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 找到所有支持的文件
    supported_files = [f for f in folder.rglob("*") if f.is_file() and can_process(f)]
    print(f"🔍 发现 {len(supported_files)} 个可处理的文件")

    # 处理每个文件
    for file_path in supported_files:
        try:
            print(f"📄 处理文件: {file_path.name}")
            content = process_document(file_path)

            # 添加文件来源信息
            doc_with_source = f"[文件: {file_path.name}]\n\n{content}"
            documents.append(doc_with_source)

        except Exception as e:
            print(f"⚠️  处理 {file_path.name} 时出错: {e}")

    if not documents:
        print("❌ 没有成功处理任何文档")
        return

    print(f"✅ 成功处理 {len(documents)} 个文档")

    # 2. 使用LightRAG构建知识图谱
    print("🧠 开始构建知识图谱...")
    builder = create_lightrag_graph_builder(output_dir)

    try:
        # 构建图谱
        graph = await builder.abuild_graph(texts=documents, graph_name="文件夹知识图谱")

        print("🎉 知识图谱构建完成！")
        print(f"   📊 实体数量: {len(graph.entities)}")
        print(f"   🔗 关系数量: {len(graph.relations)}")

        # 3. 简单的搜索演示
        print("\n🔍 搜索演示:")
        test_queries = ["主要内容是什么？", "涉及哪些技术？", "有哪些重要信息？"]

        for query in test_queries:
            try:
                result = await builder.asearch_graph(query, "hybrid")
                answer = result.get("result", "无结果")[:150] + "..."
                print(f"   Q: {query}")
                print(f"   A: {answer}\n")
            except Exception as e:
                print(f"   搜索'{query}'失败: {e}")

        return builder, graph

    except Exception as e:
        print(f"❌ 构建知识图谱失败: {e}")
        return None, None


async def main():
    """主函数"""
    # 使用examples/documents文件夹作为输入
    documents_folder = "./examples/documents"

    print("=" * 50)
    print("🚀 简单文件夹到知识图谱示例")
    print("=" * 50)

    builder, graph = await simple_folder_to_knowledge_graph(documents_folder)

    if builder:
        print("\n📈 查看更多统计信息:")
        stats = builder.get_graph_statistics()
        for key, value in stats.items():
            if key != "error":
                print(f"   {key}: {value}")

        # 清理资源
        try:
            builder.cleanup()
        except Exception as e:
            print(e)

    print("\n✨ 示例完成！")


if __name__ == "__main__":
    asyncio.run(main())
