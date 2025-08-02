"""
文件夹文档处理示例

本示例展示如何读取指定文件夹下的所有文件，使用文档处理器解析各种格式的文件，
然后使用LightRAGGraphBuilder构建知识图谱。

支持的文件格式：
- 文本文件：.txt, .md, .markdown
- JSON文件：.json
- HTML文件：.html, .htm
- CSV文件：.csv
- Word文档：.docx, .doc
- PDF文件：.pdf
- 图片文件：.jpg, .jpeg, .png, .gif (使用多模态AI模型)
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Tuple

from agraph import LightRAGGraphBuilder, create_lightrag_graph_builder
from agraph.processer import (
    DocumentProcessorManager,
    can_process,
    get_supported_extensions,
    process_document,
)


class FolderDocumentProcessor:
    """文件夹文档处理器"""

    def __init__(self, documents_folder: str = "./examples/documents"):
        """初始化文件夹文档处理器

        Args:
            documents_folder: 文档文件夹路径
        """
        self.documents_folder = Path(documents_folder)
        self.processor = DocumentProcessorManager()
        self.processed_documents: List[Dict] = []

    def scan_folder(self) -> List[Path]:
        """扫描文件夹获取所有支持的文件

        Returns:
            支持处理的文件路径列表
        """
        if not self.documents_folder.exists():
            print(f"文件夹不存在: {self.documents_folder}")
            return []

        supported_files = []
        supported_exts = get_supported_extensions()

        print(f"扫描文件夹: {self.documents_folder}")
        print(f"支持的文件格式: {', '.join(supported_exts)}")
        print("-" * 50)

        for file_path in self.documents_folder.rglob("*"):
            if file_path.is_file() and can_process(file_path):
                supported_files.append(file_path)
                print(f"发现支持的文件: {file_path.name} ({file_path.suffix})")

        print(f"\n总共发现 {len(supported_files)} 个可处理的文件")
        return supported_files

    def process_file(self, file_path: Path) -> Dict:
        """处理单个文件

        Args:
            file_path: 文件路径

        Returns:
            包含文件信息和处理结果的字典
        """
        try:
            print(f"正在处理: {file_path.name}")

            # 根据文件类型设置不同的处理参数
            kwargs = {}
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                # 图片文件使用描述性提示
                kwargs["prompt"] = "请详细描述这张图片的内容，包括其中的文字、对象、场景等信息。"
            elif file_path.suffix.lower() == ".csv":
                # CSV文件包含表头
                kwargs["include_headers"] = True
                kwargs["max_rows"] = 1000
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                # Excel文件处理所有工作表
                kwargs["sheet_name"] = "all"
                kwargs["max_rows"] = 1000
            elif file_path.suffix.lower() == ".html":
                # HTML文件保留结构并提取链接
                kwargs["preserve_structure"] = True
                kwargs["extract_links"] = True
            elif file_path.suffix.lower() == ".json":
                # JSON文件美化显示
                kwargs["pretty_print"] = True

            # 处理文档内容
            content = process_document(file_path, **kwargs)

            # 提取元数据
            metadata = self.processor.extract_metadata(file_path)

            # 构建文档信息
            doc_info = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
                "content": content,
                "metadata": metadata,
                "content_length": len(content),
                "processing_status": "success",
            }

            print(f"  ✓ 处理成功，提取内容长度: {len(content)} 字符")
            return doc_info

        except Exception as e:
            error_info = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "error": str(e),
                "processing_status": "failed",
            }
            print(f"  ✗ 处理失败: {str(e)}")
            return error_info

    def process_all_files(self) -> Tuple[List[str], List[Dict]]:
        """处理文件夹中的所有文件

        Returns:
            (成功处理的文档内容列表, 所有文档的详细信息列表)
        """
        files = self.scan_folder()
        if not files:
            return [], []

        print(f"\n开始处理 {len(files)} 个文件...")
        print("=" * 50)

        successful_contents = []
        all_doc_info = []

        for file_path in files:
            doc_info = self.process_file(file_path)
            all_doc_info.append(doc_info)

            if doc_info["processing_status"] == "success":
                # 为知识图谱构建添加文件来源信息
                content_with_source = (
                    f"文件来源: {doc_info['file_name']}\n文件类型: {doc_info['file_type']}\n\n{doc_info['content']}"
                )
                successful_contents.append(content_with_source)

        self.processed_documents = all_doc_info

        success_count = len(successful_contents)
        failed_count = len(files) - success_count

        print("\n" + "=" * 50)
        print(f"文件处理完成！成功: {success_count}, 失败: {failed_count}")

        return successful_contents, all_doc_info

    def print_processing_summary(self):
        """打印处理摘要信息"""
        if not self.processed_documents:
            print("暂无处理结果")
            return

        print("\n" + "=" * 60)
        print("文档处理摘要")
        print("=" * 60)

        # 按文件类型统计
        type_stats = {}
        success_count = 0
        total_content_length = 0

        for doc in self.processed_documents:
            file_type = doc["file_type"]
            type_stats[file_type] = type_stats.get(file_type, 0) + 1

            if doc["processing_status"] == "success":
                success_count += 1
                total_content_length += doc.get("content_length", 0)

        print(f"总文件数: {len(self.processed_documents)}")
        print(f"成功处理: {success_count}")
        print(f"处理失败: {len(self.processed_documents) - success_count}")
        print(f"总内容长度: {total_content_length:,} 字符")

        print("\n文件类型分布:")
        for file_type, count in sorted(type_stats.items()):
            print(f"  {file_type}: {count} 个文件")

        # 显示处理失败的文件
        failed_files = [doc for doc in self.processed_documents if doc["processing_status"] == "failed"]
        if failed_files:
            print("\n处理失败的文件:")
            for doc in failed_files:
                print(f"  {doc['file_name']}: {doc.get('error', '未知错误')}")


async def build_knowledge_graph_from_folder(
    documents_folder: str = "./examples/documents",
    graph_name: str = "文件夹知识图谱",
    working_dir: str = "./workdir/folder_processing",
):
    """从文件夹构建知识图谱的主要函数

    Args:
        documents_folder: 文档文件夹路径
        graph_name: 知识图谱名称
        working_dir: LightRAG工作目录
    """

    print("=" * 80)
    print("从文件夹构建知识图谱示例")
    print("=" * 80)

    # 1. 初始化文档处理器
    processor = FolderDocumentProcessor(documents_folder)

    # 2. 处理文件夹中的所有文件
    document_contents, doc_details = processor.process_all_files()

    if not document_contents:
        print("没有成功处理任何文档，无法构建知识图谱")
        return None

    # 3. 显示处理摘要
    processor.print_processing_summary()

    # 4. 创建LightRAG构建器
    print(f"\n初始化LightRAG构建器，工作目录: {working_dir}")
    builder = create_lightrag_graph_builder(working_dir)

    try:
        # 5. 构建知识图谱
        print(f"\n开始构建知识图谱: {graph_name}")
        print(f"文档数量: {len(document_contents)}")
        print("这可能需要几分钟时间，请耐心等待...")
        print("-" * 50)

        graph = await builder.abuild_graph(texts=document_contents, graph_name=graph_name)

        print("\n✓ 知识图谱构建完成！")
        print(f"  实体数量: {len(graph.entities)}")
        print(f"  关系数量: {len(graph.relations)}")

        # 6. 显示部分实体信息
        print("\n实体示例 (显示前10个):")
        print("-" * 40)
        for i, entity in enumerate(list(graph.entities.values())[:10]):
            print(f"{i+1:2d}. {entity.name} ({entity.entity_type.value})")
            if entity.description:
                desc = entity.description[:60] + "..." if len(entity.description) > 60 else entity.description
                print(f"     {desc}")

        # 7. 显示部分关系信息
        print("\n关系示例 (显示前8个):")
        print("-" * 40)
        for i, relation in enumerate(list(graph.relations.values())[:8]):
            if relation.head_entity and relation.tail_entity:
                print(f"{i+1:2d}. {relation.head_entity.name} -> {relation.tail_entity.name}")
                print(f"     关系: {relation.relation_type.value} (置信度: {relation.confidence:.2f})")

        # 8. 获取统计信息
        print("\n图谱统计信息:")
        print("-" * 30)
        stats = builder.get_graph_statistics()
        for key, value in stats.items():
            if key != "error":
                print(f"  {key}: {value}")

        return builder, graph

    except Exception as e:
        print(f"\n构建知识图谱时出现错误: {e}")
        return None, None


async def demo_graph_search(builder: LightRAGGraphBuilder):
    """演示图谱搜索功能"""
    if not builder:
        print("构建器未初始化，跳过搜索演示")
        return

    print("\n" + "=" * 60)
    print("知识图谱搜索演示")
    print("=" * 60)

    # 预定义的查询问题
    demo_queries = [
        ("公司的主要产品有哪些？", "hybrid"),
        ("团队的技术背景如何？", "local"),
        ("公司使用了哪些技术栈？", "global"),
        ("有哪些研究成果和论文？", "hybrid"),
        ("公司的核心业务是什么？", "naive"),
    ]

    for i, (query, search_mode) in enumerate(demo_queries, 1):
        try:
            print(f"\n查询 {i}: {query}")
            print(f"搜索模式: {search_mode}")
            print("-" * 40)

            result = await builder.asearch_graph(query, search_mode)
            answer = result.get("result", "未找到相关信息")

            # 限制答案长度用于演示
            if len(answer) > 300:
                answer = answer[:300] + "..."

            print(f"回答: {answer}")

        except Exception as e:
            print(f"搜索失败: {e}")

        print()


async def main():
    """主函数"""
    # 设置文档文件夹路径
    documents_folder = "./examples/documents"

    # 检查文档文件夹是否存在
    if not Path(documents_folder).exists():
        print(f"文档文件夹不存在: {documents_folder}")
        print("请确保examples/documents文件夹存在并包含要处理的文档文件")
        return

    try:
        # 1. 从文件夹构建知识图谱
        builder, graph = await build_knowledge_graph_from_folder(
            documents_folder=documents_folder,
            graph_name="企业文档知识图谱",
            working_dir="./workdir/folder_lightrag_storage",
        )

        if builder and graph:
            # 2. 演示搜索功能
            await demo_graph_search(builder)

            # 3. 可选：导出图谱
            print("\n导出知识图谱到GraphML文件...")
            export_path = "./exported_folder_knowledge_graph.graphml"
            success = builder.export_to_graphml(graph, export_path)

            if success:
                print(f"✓ 图谱已导出到: {export_path}")
                file_size = os.path.getsize(export_path)
                print(f"  文件大小: {file_size:,} bytes")
            else:
                print("✗ 导出失败")

        else:
            print("知识图谱构建失败")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
    finally:
        # 清理资源
        if "builder" in locals() and builder:
            try:
                builder.cleanup()
            except Exception:
                pass

    print("\n程序执行完成！")


if __name__ == "__main__":
    # 显示支持的文件格式信息
    print("支持的文档格式:")
    supported_formats = get_supported_extensions()
    for fmt in sorted(supported_formats):
        print(f"  {fmt}")
    print()

    # 运行主程序
    asyncio.run(main())
