#!/usr/bin/env python3
"""
AGraph 文档处理示例

此示例展示如何使用AGraph从真实文档构建知识图谱，包括：
1. 从documents目录读取各种格式的文档
2. 构建综合知识图谱
3. 基于文档内容进行智能问答
4. 演示企业级应用场景
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agraph import AGraph, BuilderConfig


class DocumentsDemo:
    """基于真实文档的AGraph演示"""

    def __init__(self):
        self.documents_dir = Path(__file__).parent / "documents"
        self.workdir = project_root / "workdir" / "documents_demo"
        self.agraph = None

    async def run_demo(self):
        """运行文档处理演示"""
        print("📚 AGraph文档处理示例")
        print("=" * 50)

        # 1. 检查文档目录
        if not self.check_documents():
            return

        # 2. 初始化系统
        await self.initialize_system()

        # 3. 处理文档并构建知识图谱
        await self.process_documents()

        # 4. 演示企业知识问答
        await self.demo_enterprise_qa()

        # 5. 展示搜索功能
        await self.demo_search_capabilities()

        # 6. 系统信息展示
        await self.show_system_info()

        print("\n✅ 文档处理演示完成!")

    def check_documents(self) -> bool:
        """检查文档目录和文件"""
        print("\n📂 1. 检查文档目录...")

        if not self.documents_dir.exists():
            print(f"❌ 文档目录不存在: {self.documents_dir}")
            return False

        # 获取所有文档文件
        doc_files = []
        supported_extensions = {'.txt', '.md', '.json', '.csv'}  # 暂时只支持这些格式

        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                doc_files.append(file_path)

        if not doc_files:
            print("❌ 未找到支持的文档文件")
            return False

        print(f"✅ 找到 {len(doc_files)} 个文档文件:")
        for doc in doc_files:
            file_size = doc.stat().st_size
            print(f"   📄 {doc.name} ({file_size} bytes)")

        self.doc_files = doc_files
        return True

    async def initialize_system(self):
        """初始化AGraph系统"""
        print("\n⚙️ 2. 初始化AGraph系统...")

        # 创建工作目录
        self.workdir.mkdir(parents=True, exist_ok=True)

        # 配置AGraph
        config = BuilderConfig(
            chunk_size=1000,
            chunk_overlap=200,
            entity_confidence_threshold=0.7,
            relation_confidence_threshold=0.6,
            cache_dir=str(self.workdir / "cache")
        )

        self.agraph = AGraph(
            collection_name="enterprise_documents",
            persist_directory=str(self.workdir / "vectordb"),
            vector_store_type="memory",  # 使用内存存储进行演示
            config=config,
            use_openai_embeddings=False
        )

        await self.agraph.initialize()
        print("✅ AGraph系统初始化完成")

    async def process_documents(self):
        """处理文档并构建知识图谱"""
        print("\n🏗️ 3. 处理文档并构建知识图谱...")

        try:
            # 读取文档内容
            document_contents = []
            for doc_file in self.doc_files:
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        document_contents.append(content)
                        print(f"   📖 读取: {doc_file.name} ({len(content)} 字符)")
                except UnicodeDecodeError:
                    # 尝试其他编码
                    try:
                        with open(doc_file, 'r', encoding='gbk') as f:
                            content = f.read()
                            document_contents.append(content)
                            print(f"   📖 读取: {doc_file.name} ({len(content)} 字符, GBK编码)")
                    except Exception as e:
                        print(f"   ⚠️  跳过文件 {doc_file.name}: {e}")
                except Exception as e:
                    print(f"   ⚠️  读取文件失败 {doc_file.name}: {e}")

            if not document_contents:
                print("❌ 没有成功读取任何文档内容")
                return

            print(f"\n🔨 开始构建知识图谱 (共{len(document_contents)}个文档)...")

            # 从文档内容构建知识图谱
            try:
                knowledge_graph = self.agraph.build_from_texts(
                    texts=document_contents,
                    graph_name="企业文档知识图谱",
                    graph_description="基于企业内部文档构建的综合知识图谱",
                    save_to_vector_store=True
                )

                print("✅ 知识图谱构建成功!")
                print(f"   🏷️  实体数量: {len(knowledge_graph.entities)}")
                print(f"   🔗 关系数量: {len(knowledge_graph.relations)}")
                print(f"   📄 文本块数量: {len(knowledge_graph.text_chunks)}")

                # 显示提取的关键实体
                if knowledge_graph.entities:
                    print("\n🔍 提取的关键实体:")
                    entities_by_type = {}
                    for entity_id, entity in knowledge_graph.entities.items():
                        entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                        if entity_type not in entities_by_type:
                            entities_by_type[entity_type] = []
                        entities_by_type[entity_type].append(entity.name)

                    for entity_type, names in entities_by_type.items():
                        print(f"   📋 {entity_type}: {', '.join(names[:5])}" +
                              (f" (及其他{len(names)-5}个)" if len(names) > 5 else ""))

            except Exception as e:
                print(f"⚠️  知识图谱构建遇到问题: {e}")
                print("📝 创建基础数据结构继续演示...")
                await self.create_fallback_data(document_contents)

        except Exception as e:
            print(f"❌ 文档处理失败: {e}")

    async def create_fallback_data(self, document_contents):
        """创建备用数据结构"""
        from agraph.base.entities import Entity
        from agraph.base.text import TextChunk
        from agraph.base.types import EntityType

        # 从文档内容中提取关键词作为实体
        entities = [
            Entity(name="人工智能", entity_type=EntityType.CONCEPT,
                  description="AI技术概念", confidence=0.9),
            Entity(name="公司", entity_type=EntityType.ORGANIZATION,
                  description="企业组织", confidence=0.85),
            Entity(name="北京市", entity_type=EntityType.LOCATION,
                  description="公司所在地", confidence=0.8),
            Entity(name="数据分析", entity_type=EntityType.CONCEPT,
                  description="数据处理技术", confidence=0.85),
        ]

        # 创建文本块
        text_chunks = []
        for i, content in enumerate(document_contents[:3]):  # 只取前3个文档
            chunk = TextChunk(
                content=content[:500] + "..." if len(content) > 500 else content,
                title=f"文档{i+1}",
                source=f"doc_{i+1}",
                start_index=0,
                end_index=min(500, len(content))
            )
            text_chunks.append(chunk)

        # 添加到向量存储
        if self.agraph.vector_store:
            await self.agraph.vector_store.batch_add_entities(entities)
            await self.agraph.vector_store.batch_add_text_chunks(text_chunks)

        print("✅ 备用数据结构创建完成")

    async def demo_enterprise_qa(self):
        """演示企业知识问答"""
        print("\n💼 4. 企业知识问答演示...")

        # 基于文档内容的实际问题
        questions = [
            "公司的主要业务是什么？",
            "公司成立于什么时候？",
            "公司总部在哪里？",
            "团队有多少人？",
            "公司的核心技术是什么？",
            "公司的愿景是什么？"
        ]

        print("基于企业文档内容回答问题:\n")

        for i, question in enumerate(questions):
            print(f"❓ 问题 {i+1}: {question}")

            try:
                response = await self.agraph.chat(
                    question=question,
                    entity_top_k=5,
                    relation_top_k=3,
                    text_chunk_top_k=3,
                    response_type="详细回答"
                )

                answer = response['answer']
                print(f"🤖 回答: {answer}")

                # 显示引用信息
                context = response['context']
                if context.get('text_chunks'):
                    sources = []
                    for item in context['text_chunks'][:2]:
                        chunk = item['text_chunk']
                        sources.append(f"{chunk.title or chunk.source}")
                    if sources:
                        print(f"📚 参考文档: {', '.join(sources)}")

                print()

            except Exception as e:
                print(f"🤖 回答: 抱歉，无法回答此问题: {e}\n")

    async def demo_search_capabilities(self):
        """演示搜索能力"""
        print("\n🔍 5. 搜索功能演示...")

        search_terms = ["人工智能", "数据分析", "技术团队", "北京"]

        for term in search_terms:
            print(f"\n🔎 搜索关键词: '{term}'")

            try:
                # 搜索实体
                entities = await self.agraph.search_entities(term, top_k=3)
                if entities:
                    print("   📋 相关实体:")
                    for entity, score in entities:
                        print(f"      - {entity.name} ({entity.entity_type}) [相似度: {score:.3f}]")

                # 搜索文档
                text_chunks = await self.agraph.search_text_chunks(term, top_k=2)
                if text_chunks:
                    print("   📄 相关文档片段:")
                    for chunk, score in text_chunks:
                        preview = chunk.content[:80].replace('\n', ' ') + "..." if len(chunk.content) > 80 else chunk.content.replace('\n', ' ')
                        print(f"      - {preview} [相似度: {score:.3f}]")

                if not entities and not text_chunks:
                    print("   🔍 未找到相关结果")

            except Exception as e:
                print(f"   ❌ 搜索失败: {e}")

    async def show_system_info(self):
        """显示系统信息"""
        print("\n📊 6. 系统信息...")

        try:
            stats = await self.agraph.get_stats()

            # 向量存储统计
            if 'vector_store' in stats:
                vs_stats = stats['vector_store']
                print("🗂️  向量存储统计:")
                print(f"   - 实体: {vs_stats.get('entities', 0)}")
                print(f"   - 关系: {vs_stats.get('relations', 0)}")
                print(f"   - 文本块: {vs_stats.get('text_chunks', 0)}")

            # 知识图谱统计
            if 'knowledge_graph' in stats:
                kg_stats = stats['knowledge_graph']
                print("\n🕸️  知识图谱统计:")
                for key, value in kg_stats.items():
                    print(f"   - {key}: {value}")

            # 系统状态
            print(f"\n⚙️  系统状态:")
            print(f"   - 集合名称: {self.agraph.collection_name}")
            print(f"   - 存储类型: {self.agraph.vector_store_type}")
            print(f"   - 已初始化: {self.agraph.is_initialized}")
            print(f"   - 有知识图谱: {self.agraph.has_knowledge_graph}")

        except Exception as e:
            print(f"❌ 获取系统信息失败: {e}")

    async def cleanup(self):
        """清理资源"""
        if self.agraph:
            await self.agraph.close()


async def main():
    """主函数"""
    demo = DocumentsDemo()

    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("📚 启动AGraph文档处理演示...")
    print("📌 此演示将处理examples/documents/目录中的文档文件")
    print("⏱️  处理时间取决于文档数量和大小\n")

    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7+版本")
        sys.exit(1)

    asyncio.run(main())
