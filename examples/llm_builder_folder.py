"""
文件夹知识库构建器示例

展示如何读取文件夹下的各种文件类型并构建知识库，
支持PDF、Word、文本、HTML、Excel、JSON、图片等多种格式
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agraph.builders import (
    BatchLLMGraphBuilder,
    FlexibleLLMGraphBuilder,
    LLMGraphBuilder,
    MinimalLLMGraphBuilder,
)
from agraph.config import Settings
from agraph.embeddings import JsonVectorStorage
from agraph.processer import (
    DocumentProcessorManager,
    can_process,
    get_supported_extensions,
    process_document,
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FolderKnowledgeBaseBuilder:
    """文件夹知识库构建器"""

    def __init__(self, builder_type: str = "flexible", output_dir: str = "workdir"):
        """
        初始化文件夹知识库构建器

        Args:
            builder_type: 构建器类型 ("minimal", "flexible", "full", "batch")
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 文档处理器
        self.doc_processor = DocumentProcessorManager()

        # 创建对应的LLM构建器
        self.builder = self._create_builder(builder_type)

        # 处理统计
        self.processing_stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "supported_files": 0,
            "unsupported_files": 0,
            "file_types": {},
            "errors": [],
        }

    def _create_builder(self, builder_type: str):
        """创建LLM构建器"""
        if builder_type == "minimal":
            return MinimalLLMGraphBuilder(
                openai_api_key=Settings.OPENAI_API_KEY,
                openai_api_base=Settings.OPENAI_API_BASE,
                llm_model=Settings.LLM_MODEL,
                temperature=0.1,
            )
        elif builder_type == "flexible":
            return FlexibleLLMGraphBuilder(
                openai_api_key=Settings.OPENAI_API_KEY,
                openai_api_base=Settings.OPENAI_API_BASE,
                llm_model=Settings.LLM_MODEL,
                embedding_model=Settings.EMBEDDING_MODEL,
                vector_storage=JsonVectorStorage(file_path=str(self.output_dir / "vector_store.json")),
            )
        elif builder_type == "batch":
            return BatchLLMGraphBuilder(
                openai_api_key=Settings.OPENAI_API_KEY,
                openai_api_base=Settings.OPENAI_API_BASE,
                llm_model=Settings.LLM_MODEL,
                embedding_model=Settings.EMBEDDING_MODEL,
                max_concurrent=8,
            )
        elif builder_type == "full":
            return LLMGraphBuilder(
                openai_api_key=Settings.OPENAI_API_KEY,
                openai_api_base=Settings.OPENAI_API_BASE,
                llm_model=Settings.LLM_MODEL,
                embedding_model=Settings.EMBEDDING_MODEL,
                max_concurrent=10,
                vector_storage=JsonVectorStorage(file_path=str(self.output_dir / "vector_store.json")),
            )
        else:
            raise ValueError(f"不支持的构建器类型: {builder_type}")

    def scan_folder(
        self, folder_path: str, recursive: bool = True, file_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """
        扫描文件夹获取所有可处理的文件

        Args:
            folder_path: 文件夹路径
            recursive: 是否递归扫描子文件夹
            file_patterns: 文件模式过滤（如 ["*.pdf", "*.txt"]）

        Returns:
            文件路径列表
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")

        if not folder.is_dir():
            raise ValueError(f"路径不是文件夹: {folder_path}")

        files = []

        # 获取支持的扩展名
        supported_extensions = get_supported_extensions()
        logger.info(f"支持的文件类型: {supported_extensions}")

        # 扫描文件
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in folder.glob(pattern):
            if file_path.is_file():
                self.processing_stats["total_files"] += 1

                # 检查文件扩展名
                ext = file_path.suffix.lower()
                self.processing_stats["file_types"][ext] = self.processing_stats["file_types"].get(ext, 0) + 1

                # 检查是否可以处理
                if can_process(file_path):
                    self.processing_stats["supported_files"] += 1

                    # 应用文件模式过滤
                    if file_patterns:
                        if any(file_path.match(pattern) for pattern in file_patterns):
                            files.append(file_path)
                    else:
                        files.append(file_path)
                else:
                    self.processing_stats["unsupported_files"] += 1
                    logger.debug(f"不支持的文件类型: {file_path}")

        logger.info(
            f"扫描完成: 总文件 {self.processing_stats['total_files']} 个，"
            f"支持处理 {self.processing_stats['supported_files']} 个，"
            f"将处理 {len(files)} 个"
        )

        return files

    def process_files(self, file_paths: List[Path]) -> Tuple[List[str], Dict[str, str]]:
        """
        处理文件列表，提取文本内容

        Args:
            file_paths: 文件路径列表

        Returns:
            (成功提取的文本列表, 文件路径到文本的映射)
        """
        texts = []
        file_text_mapping = {}

        logger.info(f"开始处理 {len(file_paths)} 个文件...")

        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"处理文件 {i}/{len(file_paths)}: {file_path.name}")

                # 处理文件
                text_content = process_document(file_path)

                if text_content and text_content.strip():
                    # 添加文件信息到文本内容
                    enhanced_content = f"文件: {file_path.name}\n路径: {file_path}\n\n{text_content}"
                    texts.append(enhanced_content)
                    file_text_mapping[str(file_path)] = enhanced_content

                    self.processing_stats["processed_files"] += 1
                    logger.debug(f"成功处理: {file_path.name}, 内容长度: {len(text_content)}")
                else:
                    logger.warning(f"文件内容为空: {file_path}")

            except Exception as e:
                self.processing_stats["failed_files"] += 1
                self.processing_stats["errors"].append(f"{file_path}: {str(e)}")
                logger.error(f"处理文件失败 {file_path}: {e}")

        logger.info(
            f"文件处理完成: 成功 {self.processing_stats['processed_files']} 个，"
            f"失败 {self.processing_stats['failed_files']} 个"
        )

        return texts, file_text_mapping

    async def build_knowledge_base(
        self,
        folder_path: str,
        graph_name: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> Dict:
        """
        从文件夹构建知识库

        Args:
            folder_path: 文件夹路径
            graph_name: 图谱名称
            recursive: 是否递归扫描
            file_patterns: 文件模式过滤
            chunk_size: 文本分块大小（可选）

        Returns:
            包含图谱和统计信息的字典
        """
        logger.info(f"开始构建知识库: {graph_name}")
        logger.info(f"源文件夹: {folder_path}")

        try:
            # 1. 扫描文件夹
            file_paths = self.scan_folder(folder_path, recursive, file_patterns)

            if not file_paths:
                logger.warning("没有找到可处理的文件")
                return {"graph": None, "stats": self.processing_stats, "message": "没有找到可处理的文件"}

            # 2. 处理文件
            texts, file_mapping = self.process_files(file_paths)

            if not texts:
                logger.warning("没有成功提取到文本内容")
                return {
                    "graph": None,
                    "stats": self.processing_stats,
                    "file_mapping": file_mapping,
                    "message": "没有成功提取到文本内容",
                }

            # 3. 文本分块（如果需要）
            if chunk_size:
                texts = self._chunk_texts(texts, chunk_size)

            # 4. 构建图谱
            logger.info(f"使用 {len(texts)} 个文本块构建图谱...")
            graph = await self.builder.build_graph(texts=texts, graph_name=graph_name)

            # 5. 保存结果
            await self._save_results(graph, file_mapping, graph_name)

            logger.info(f"知识库构建完成: {graph_name}")

            return {
                "graph": graph,
                "stats": self.processing_stats,
                "file_mapping": file_mapping,
                "message": "知识库构建成功",
            }

        except Exception as e:
            logger.error(f"构建知识库失败: {e}")
            self.processing_stats["errors"].append(f"构建失败: {str(e)}")
            return {"graph": None, "stats": self.processing_stats, "message": f"构建失败: {str(e)}"}

    def _chunk_texts(self, texts: List[str], chunk_size: int) -> List[str]:
        """将长文本分块"""
        chunked_texts = []

        for text in texts:
            if len(text) <= chunk_size:
                chunked_texts.append(text)
            else:
                # 简单的按字符分块（可以改进为按段落或句子分块）
                for i in range(0, len(text), chunk_size):
                    chunk = text[i : i + chunk_size]
                    if chunk.strip():
                        chunked_texts.append(chunk)

        logger.info(f"文本分块完成: {len(texts)} -> {len(chunked_texts)}")
        return chunked_texts

    async def _save_results(self, graph, file_mapping: Dict[str, str], graph_name: str):
        """保存构建结果"""
        try:
            # 保存图谱信息
            graph_info = {
                "name": graph.name,
                "entities_count": len(graph.entities),
                "relations_count": len(graph.relations),
                "processing_stats": self.processing_stats,
                "file_mapping": {path: len(content) for path, content in file_mapping.items()},
            }

            import json

            info_file = self.output_dir / f"{graph_name}_info.json"
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(graph_info, f, ensure_ascii=False, indent=2)

            # 如果是全功能构建器，导出图谱
            if hasattr(self.builder, "export_to_format"):
                exported_data = await self.builder.export_to_format(graph, "json")
                export_file = self.output_dir / f"{graph_name}_graph.json"
                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(exported_data, f, ensure_ascii=False, indent=2)
                logger.info(f"图谱已导出到: {export_file}")

            logger.info(f"构建信息已保存到: {info_file}")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def print_statistics(self):
        """打印处理统计信息"""
        stats = self.processing_stats

        print("\n" + "=" * 60)
        print("📊 文件夹处理统计信息")
        print("=" * 60)

        print(f"总文件数: {stats['total_files']}")
        print(f"支持的文件数: {stats['supported_files']}")
        print(f"不支持的文件数: {stats['unsupported_files']}")
        print(f"成功处理: {stats['processed_files']}")
        print(f"处理失败: {stats['failed_files']}")

        print("\n📁 文件类型分布:")
        for ext, count in stats["file_types"].items():
            print(f"  {ext or '(无扩展名)'}: {count} 个")

        if stats["errors"]:
            print("\n❌ 错误信息:")
            for error in stats["errors"][:5]:  # 只显示前5个错误
                print(f"  - {error}")
            if len(stats["errors"]) > 5:
                print(f"  ... 还有 {len(stats['errors']) - 5} 个错误")


async def example_simple_folder():
    """示例1: 简单文件夹处理"""
    print("\n" + "=" * 50)
    print("📁 示例1: 简单文件夹知识库构建")
    print("=" * 50)

    # 创建简单构建器
    builder = FolderKnowledgeBaseBuilder(builder_type="minimal", output_dir="workdir/simple_kb")

    # 构建知识库
    result = await builder.build_knowledge_base(
        folder_path="examples/documents", graph_name="simple_folder_kb", recursive=True  # 假设有一个文档文件夹
    )

    if result["graph"]:
        graph = result["graph"]
        print("✅ 简单知识库构建成功:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")
    else:
        print(f"❌ 构建失败: {result['message']}")

    # 打印统计
    builder.print_statistics()


async def example_advanced_folder():
    """示例2: 高级文件夹处理"""
    print("\n" + "=" * 50)
    print("🚀 示例2: 高级文件夹知识库构建")
    print("=" * 50)

    # 创建高级构建器
    builder = FolderKnowledgeBaseBuilder(builder_type="full", output_dir="workdir/advanced_kb")

    # 指定文件类型过滤
    file_patterns = ["*.pdf", "*.txt", "*.md", "*.docx"]

    # 构建知识库
    result = await builder.build_knowledge_base(
        folder_path="examples/documents",
        graph_name="advanced_folder_kb",
        recursive=True,
        file_patterns=file_patterns,
        chunk_size=2000,  # 2000字符分块
    )

    if result["graph"]:
        graph = result["graph"]
        print("✅ 高级知识库构建成功:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")

        # 显示一些实体
        print("\n📋 实体示例:")
        for i, (entity_id, entity) in enumerate(list(graph.entities.items())[:5]):
            print(f"   {i+1}. {entity.name} ({entity.entity_type.value})")

        # 如果是全功能构建器，进行验证
        if hasattr(builder.builder, "validate_graph"):
            validation_result = await builder.builder.validate_graph(graph)
            print("\n🔍 图谱验证:")
            print(f"   - 验证通过: {validation_result.get('valid', False)}")
    else:
        print(f"❌ 构建失败: {result['message']}")

    # 打印统计
    builder.print_statistics()


async def example_batch_processing():
    """示例3: 批量处理大文件夹"""
    print("\n" + "=" * 50)
    print("📦 示例3: 批量处理大文件夹")
    print("=" * 50)

    # 创建批量构建器
    builder = FolderKnowledgeBaseBuilder(builder_type="batch", output_dir="workdir/batch_kb")

    # 构建知识库
    result = await builder.build_knowledge_base(
        folder_path="examples/documents",
        graph_name="batch_folder_kb",
        recursive=True,
        chunk_size=1500,  # 较小的分块以提高并行度
    )

    if result["graph"]:
        graph = result["graph"]
        print("✅ 批量知识库构建成功:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")
    else:
        print(f"❌ 构建失败: {result['message']}")

    # 打印统计
    builder.print_statistics()


async def example_create_sample_documents():
    """创建示例文档文件夹"""
    docs_dir = Path("examples/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 创建一些示例文件
    sample_files = {
        "公司介绍.txt": """
腾讯控股有限公司是一家中国跨国技术集团公司。
公司成立于1998年，总部位于深圳。
主要业务包括社交网络、网络游戏、移动支付、云计算等。
创始人包括马化腾、张志东、许晨晔、陈一丹、曾李青。
        """,
        "产品信息.md": """
# 腾讯主要产品

## 即时通讯
- 微信：移动即时通讯应用
- QQ：桌面和移动即时通讯软件

## 游戏
- 王者荣耀：移动MOBA游戏
- 和平精英：大逃杀类手游

## 金融科技
- 微信支付：移动支付平台
- QQ钱包：数字钱包服务
        """,
        "人物简介.txt": """
马化腾，腾讯公司主要创办人之一，现任腾讯公司董事会主席兼首席执行官。
1971年出生于广东省汕头市。
1993年毕业于深圳大学计算机系。
1998年与同学张志东等人创立腾讯公司。
        """,
        "技术信息.txt": """
腾讯云是腾讯倾力打造的面向云时代的智能产业共同体。
提供云计算、大数据、人工智能等技术服务。
服务范围涵盖游戏、视频、金融、零售、教育等多个行业。
在全球27个地理区域内运营着70个可用区。
        """,
    }

    for filename, content in sample_files.items():
        file_path = docs_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

    print(f"✅ 示例文档已创建在: {docs_dir}")
    print(f"   创建了 {len(sample_files)} 个示例文件")


async def main():
    """运行所有示例"""
    print("🎯 文件夹知识库构建示例")
    print("演示如何从文件夹构建知识库")

    # 首先创建示例文档
    await example_create_sample_documents()

    # 运行各个示例
    await example_simple_folder()
    await example_advanced_folder()
    await example_batch_processing()

    print("\n🎉 所有示例运行完成!")

    # 显示支持的文件类型
    print("\n📋 支持的文件类型:")
    extensions = get_supported_extensions()
    for ext in sorted(extensions):
        print(f"   - {ext}")


if __name__ == "__main__":
    asyncio.run(main())
