#!/usr/bin/env python3
"""
AGraph vs 传统方法对比示例

此示例对比展示：
1. 传统的文本检索方法
2. AGraph的知识图谱增强检索
3. 两种方法在问答任务上的差异
4. AGraph的优势和应用场景

帮助用户理解AGraph的价值和适用场景。
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agraph import AGraph


class TraditionalSearch:
    """传统文本检索方法"""

    def __init__(self, documents: List[str]):
        self.documents = documents
        self.processed_docs = [self._preprocess(doc) for doc in documents]

    def _preprocess(self, text: str) -> str:
        """简单的文本预处理"""
        # 转为小写，移除标点符号
        return re.sub(r'[^\w\s]', '', text.lower())

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """基于关键词匹配的简单搜索"""
        query_processed = self._preprocess(query)
        query_words = set(query_processed.split())

        results = []
        for i, doc in enumerate(self.processed_docs):
            doc_words = set(doc.split())
            # 计算简单的词汇重叠分数
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0
            results.append((self.documents[i], score))

        # 按分数排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def answer_question(self, question: str) -> str:
        """基于检索结果的简单问答"""
        results = self.search(question, top_k=1)
        if results and results[0][1] > 0:
            # 返回最相关的文档片段
            doc = results[0][0]
            sentences = doc.split('。')
            # 返回第一个句子作为答案
            return sentences[0] + "。" if sentences else "无法找到相关答案。"
        return "无法找到相关答案。"


class ComparisonDemo:
    """AGraph vs 传统方法对比演示"""

    def __init__(self):
        self.sample_documents = [
            "苹果公司是一家美国跨国科技公司，总部位于加利福尼亚州库比蒂诺。公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年4月1日创立。",

            "iPhone是苹果公司设计和销售的智能手机系列。第一代iPhone于2007年1月9日由史蒂夫·乔布斯发布，彻底改变了移动通信行业。",

            "史蒂夫·乔布斯是苹果公司的联合创始人和前首席执行官。他以其创新的产品设计和营销策略而闻名，被认为是个人电脑革命的先驱人物。",

            "iPad是苹果公司开发的平板电脑系列，于2010年首次发布。iPad开创了现代平板电脑市场，广泛应用于教育、商务和娱乐。",

            "macOS是苹果公司为Mac计算机开发的专有操作系统。它以其直观的用户界面和强大的功能而受到用户喜爱。",

            "App Store是苹果公司为iOS设备开发的数字发行平台，于2008年推出。开发者可以通过App Store向用户分发应用程序。",

            "苹果公司在库比蒂诺的总部被称为Apple Park，于2017年开放。这个环形建筑体现了苹果公司对设计和可持续发展的承诺。"
        ]

        self.test_questions = [
            "苹果公司是什么时候成立的？",
            "谁创立了苹果公司？",
            "iPhone什么时候发布的？",
            "苹果公司的总部在哪里？",
            "iPad有什么特点？",
            "史蒂夫·乔布斯的主要贡献是什么？",
            "App Store是用来做什么的？"
        ]

    async def run_comparison(self):
        """运行对比演示"""
        print("⚖️  AGraph vs 传统方法对比演示")
        print("=" * 60)

        # 1. 初始化两种方法
        print("\n📋 1. 初始化系统...")
        await self.initialize_systems()

        # 2. 数据准备对比
        print("\n📊 2. 数据处理对比...")
        self.compare_data_processing()

        # 3. 检索能力对比
        print("\n🔍 3. 检索能力对比...")
        await self.compare_search_capabilities()

        # 4. 问答效果对比
        print("\n💬 4. 问答效果对比...")
        await self.compare_qa_performance()

        # 5. 总结和建议
        print("\n📈 5. 总结和建议...")
        self.show_summary()

        print("\n✅ 对比演示完成！")

    async def initialize_systems(self):
        """初始化两种系统"""
        # 传统搜索
        self.traditional_search = TraditionalSearch(self.sample_documents)
        print("✅ 传统文本检索系统初始化完成")

        # AGraph系统
        self.agraph = AGraph(
            collection_name="comparison_demo",
            vector_store_type="memory",
            use_openai_embeddings=False
        )

        await self.agraph.initialize()
        print("✅ AGraph知识图谱系统初始化完成")

        # 构建知识图谱
        try:
            await self.build_knowledge_graph()
        except Exception as e:
            print(f"⚠️  知识图谱构建遇到问题: {e}")
            await self.create_fallback_kg_data()

    async def build_knowledge_graph(self):
        """构建知识图谱"""
        print("🔨 构建知识图谱...")

        knowledge_graph = self.agraph.build_from_texts(
            texts=self.sample_documents,
            graph_name="苹果公司知识图谱",
            save_to_vector_store=True
        )

        print(f"✅ 知识图谱构建完成 - 实体:{len(knowledge_graph.entities)}, 关系:{len(knowledge_graph.relations)}")

    async def create_fallback_kg_data(self):
        """创建备用知识图谱数据"""
        from agraph.base.entities import Entity
        from agraph.base.relations import Relation
        from agraph.base.text import TextChunk
        from agraph.base.types import EntityType, RelationType

        # 创建实体
        entities = [
            Entity(name="苹果公司", entity_type=EntityType.ORGANIZATION, confidence=0.95),
            Entity(name="史蒂夫·乔布斯", entity_type=EntityType.PERSON, confidence=0.95),
            Entity(name="iPhone", entity_type=EntityType.PRODUCT, confidence=0.90),
            Entity(name="iPad", entity_type=EntityType.PRODUCT, confidence=0.90),
            Entity(name="库比蒂诺", entity_type=EntityType.LOCATION, confidence=0.85),
        ]

        # 创建关系
        relations = [
            Relation(head_entity=entities[1], tail_entity=entities[0],
                    relation_type=RelationType.FOUNDED_BY, confidence=0.90),
            Relation(head_entity=entities[0], tail_entity=entities[2],
                    relation_type=RelationType.DEVELOPS, confidence=0.85),
            Relation(head_entity=entities[0], tail_entity=entities[4],
                    relation_type=RelationType.LOCATED_IN, confidence=0.80),
        ]

        # 创建文本块
        text_chunks = [
            TextChunk(content=doc[:200] + "..." if len(doc) > 200 else doc,
                     title=f"文档{i+1}", source=f"doc_{i+1}",
                     start_index=0, end_index=min(200, len(doc)))
            for i, doc in enumerate(self.sample_documents[:5])
        ]

        # 添加到向量存储
        if self.agraph.vector_store:
            await self.agraph.vector_store.batch_add_entities(entities)
            await self.agraph.vector_store.batch_add_relations(relations)
            await self.agraph.vector_store.batch_add_text_chunks(text_chunks)

        print("✅ 备用知识图谱数据创建完成")

    def compare_data_processing(self):
        """对比数据处理方式"""
        print("对比两种方法的数据处理特点:\n")

        print("📝 传统文本检索:")
        print("   - 数据结构: 扁平文本列表")
        print("   - 处理方式: 简单分词和预处理")
        print("   - 存储形式: 原始文本")
        print("   - 语义理解: 基于关键词匹配，无语义理解")

        print("\n🧠 AGraph知识图谱:")
        print("   - 数据结构: 实体-关系图结构")
        print("   - 处理方式: 实体识别、关系抽取、聚类分析")
        print("   - 存储形式: 结构化知识 + 向量嵌入")
        print("   - 语义理解: 深度语义理解和推理能力")

        print(f"\n📊 数据统计:")
        print(f"   - 文档数量: {len(self.sample_documents)}")
        print(f"   - 总字符数: {sum(len(doc) for doc in self.sample_documents)}")

    async def compare_search_capabilities(self):
        """对比检索能力"""
        print("对比两种方法的检索能力:\n")

        search_queries = ["创始人", "智能手机", "总部位置"]

        for query in search_queries:
            print(f"🔎 搜索查询: '{query}'")

            # 传统搜索结果
            print("📝 传统方法结果:")
            traditional_results = self.traditional_search.search(query, top_k=2)
            for i, (doc, score) in enumerate(traditional_results):
                preview = doc[:60] + "..." if len(doc) > 60 else doc
                print(f"   {i+1}. [{score:.3f}] {preview}")

            # AGraph搜索结果
            print("🧠 AGraph方法结果:")
            try:
                # 搜索实体
                entity_results = await self.agraph.search_entities(query, top_k=2)
                if entity_results:
                    print("   实体:")
                    for entity, score in entity_results:
                        print(f"      - {entity.name} ({entity.entity_type}) [{score:.3f}]")

                # 搜索文本
                text_results = await self.agraph.search_text_chunks(query, top_k=2)
                if text_results:
                    print("   文档:")
                    for chunk, score in text_results:
                        preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
                        print(f"      - [{score:.3f}] {preview}")

            except Exception as e:
                print(f"   搜索失败: {e}")

            print()

    async def compare_qa_performance(self):
        """对比问答性能"""
        print("对比两种方法的问答效果:\n")

        traditional_correct = 0
        agraph_correct = 0

        for i, question in enumerate(self.test_questions):
            print(f"❓ 问题 {i+1}: {question}")

            # 传统方法回答
            traditional_answer = self.traditional_search.answer_question(question)
            print(f"📝 传统方法: {traditional_answer}")

            # AGraph方法回答
            try:
                response = await self.agraph.chat(
                    question=question,
                    entity_top_k=3,
                    text_chunk_top_k=2,
                    response_type="简洁回答"
                )
                agraph_answer = response['answer']
                print(f"🧠 AGraph方法: {agraph_answer}")

                # 显示AGraph的额外信息
                context = response['context']
                entity_count = len(context.get('entities', []))
                if entity_count > 0:
                    print(f"   💡 检索了{entity_count}个相关实体")

            except Exception as e:
                print(f"🧠 AGraph方法: 回答失败 - {e}")

            print()

    def show_summary(self):
        """显示对比总结"""
        print("📋 对比总结:")
        print("\n传统文本检索的特点:")
        print("✅ 优势:")
        print("   - 实现简单，资源消耗低")
        print("   - 适合简单关键词匹配")
        print("   - 无需复杂的预处理")

        print("❌ 局限:")
        print("   - 缺乏语义理解能力")
        print("   - 无法处理复杂推理")
        print("   - 检索精度有限")
        print("   - 难以发现隐含关系")

        print("\nAGraph知识图谱的特点:")
        print("✅ 优势:")
        print("   - 强大的语义理解能力")
        print("   - 支持复杂推理和关联查询")
        print("   - 结构化知识表示")
        print("   - 可扩展的实体和关系模型")
        print("   - 支持多轮对话和上下文理解")

        print("❌ 局限:")
        print("   - 构建成本相对较高")
        print("   - 需要更多的计算资源")
        print("   - 对数据质量要求更高")

        print("\n🎯 应用建议:")
        print("📝 传统方法适合:")
        print("   - 简单的关键词搜索")
        print("   - 资源受限的环境")
        print("   - 快速原型开发")

        print("🧠 AGraph适合:")
        print("   - 企业知识管理")
        print("   - 智能问答系统")
        print("   - 复杂信息检索")
        print("   - 需要语义理解的应用")

    async def cleanup(self):
        """清理资源"""
        if self.agraph:
            await self.agraph.close()


async def main():
    """主函数"""
    demo = ComparisonDemo()

    try:
        await demo.run_comparison()
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("⚖️  启动AGraph vs 传统方法对比演示...")
    print("📌 此演示将对比传统文本检索与AGraph知识图谱方法的差异")
    print("⏱️  演示大约需要3-5分钟\n")

    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7+版本")
        sys.exit(1)

    asyncio.run(main())
