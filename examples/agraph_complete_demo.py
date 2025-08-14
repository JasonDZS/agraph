#!/usr/bin/env python3
"""
AGraph 完整示例：知识图谱构建、向量存储和智能对话

此示例展示了AGraph的完整功能：
1. 从文档构建知识图谱
2. 保存到向量存储
3. 进行智能检索
4. 知识库对话
5. 系统管理和监控
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from agraph import AGraph, BuilderConfig


class AGraphDemo:
    """AGraph功能演示类"""

    def __init__(self):
        self.agraph: AGraph = None
        self.demo_data_dir = Path(__file__).parent / "demo_data"
        self.workdir = project_root / "workdir" / "agraph_complete_demo"

        # 确保工作目录存在
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.demo_data_dir, exist_ok=True)

    async def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 AGraph 完整功能演示")
        print("=" * 60)

        # 1. 初始化系统
        await self.initialize_agraph()

        # 2. 准备演示数据
        self.prepare_demo_data()

        # 3. 演示知识图谱构建
        await self.demo_knowledge_graph_building()

        # 4. 演示向量检索功能
        await self.demo_vector_search()

        # 5. 演示智能对话
        await self.demo_intelligent_chat()

        # 6. 演示系统管理
        await self.demo_system_management()

        # 7. 清理资源
        await self.cleanup()

        print("\n✅ AGraph完整演示完成！")

    async def initialize_agraph(self):
        """初始化AGraph系统"""
        print("\n📦 1. 初始化AGraph系统")
        print("-" * 30)

        # 创建自定义配置
        config = BuilderConfig(
            chunk_size=800,
            chunk_overlap=150,
            entity_confidence_threshold=0.75,
            relation_confidence_threshold=0.65,
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            cluster_algorithm="community_detection",
            cache_dir=str(self.workdir / "cache")
        )

        # 创建AGraph实例
        self.agraph = AGraph(
            collection_name="demo_knowledge_base",
            persist_directory=str(self.workdir / "vectordb"),
            vector_store_type="memory",  # 使用内存存储进行演示
            config=config,
            use_openai_embeddings=False  # 避免API依赖
        )

        # 初始化系统
        await self.agraph.initialize()

        print("✅ AGraph系统初始化成功")
        print(f"   - 集合名称: {self.agraph.collection_name}")
        print(f"   - 存储类型: {self.agraph.vector_store_type}")
        print(f"   - 工作目录: {self.workdir}")

    def prepare_demo_data(self):
        """准备演示数据"""
        print("\n📝 2. 准备演示数据")
        print("-" * 30)

        # 创建演示文本数据
        demo_texts = [
            # 科技公司信息
            "苹果公司（Apple Inc.）是一家美国跨国科技公司，总部位于加利福尼亚州库比蒂诺。公司设计、开发和销售消费电子产品、计算机软件和在线服务。",

            "史蒂夫·乔布斯（Steve Jobs）和史蒂夫·沃兹尼亚克（Steve Wozniak）于1976年4月1日创立了苹果公司。乔布斯担任首席执行官，推动了公司的创新发展。",

            "iPhone是苹果公司开发的智能手机系列，首款iPhone于2007年1月9日发布。iPhone革命性地改变了移动通信行业，成为智能手机的标杆产品。",

            "iPad是苹果公司的平板电脑产品线，于2010年首次发布。iPad开创了现代平板电脑市场，广泛应用于教育、商务和娱乐领域。",

            # 竞争对手信息
            "微软公司（Microsoft Corporation）是苹果公司的主要竞争对手之一，专注于软件、服务和解决方案。微软由比尔·盖茨和保罗·艾伦于1975年创立。",

            "谷歌公司（Google LLC）开发了Android操作系统，与苹果的iOS形成竞争关系。Android是全球使用最广泛的移动操作系统。",

            # 产品技术信息
            "iOS是苹果公司为iPhone和iPad开发的移动操作系统。iOS以其流畅的用户体验和强大的安全性著称。",

            "苹果公司的A系列芯片是专门为iPhone和iPad设计的处理器。A系列芯片在性能和能耗方面表现出色。",

            # 商业信息
            "苹果公司是全球市值最高的科技公司之一，其产品在全球范围内享有很高的品牌忠诚度。",

            "库比蒂诺市是苹果公司总部所在地，也是硅谷的重要组成部分。硅谷聚集了众多科技公司和创新企业。"
        ]

        # 保存到文件（模拟从文档读取）
        for i, text in enumerate(demo_texts):
            file_path = self.demo_data_dir / f"demo_doc_{i+1}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

        self.demo_texts = demo_texts
        self.demo_files = list(self.demo_data_dir.glob("demo_doc_*.txt"))

        print(f"✅ 准备了 {len(demo_texts)} 个演示文本")
        print(f"✅ 创建了 {len(self.demo_files)} 个演示文档")

    async def demo_knowledge_graph_building(self):
        """演示知识图谱构建"""
        print("\n🏗️ 3. 知识图谱构建演示")
        print("-" * 30)

        try:
            print("3.1 从文本构建知识图谱...")

            # 从文本构建知识图谱
            kg = self.agraph.build_from_texts(
                texts=self.demo_texts,
                graph_name="科技行业知识图谱",
                graph_description="基于科技公司和产品信息构建的知识图谱",
                save_to_vector_store=False  # 稍后手动保存
            )

            print("✅ 知识图谱构建成功！")
            print(f"   - 实体数量: {len(kg.entities)}")
            print(f"   - 关系数量: {len(kg.relations)}")
            print(f"   - 聚类数量: {len(kg.clusters)}")
            print(f"   - 文本块数量: {len(kg.text_chunks)}")

            # 显示部分实体
            if kg.entities:
                print("\n📋 提取的实体样例:")
                for i, (entity_id, entity) in enumerate(list(kg.entities.items())[:8]):
                    print(f"   {i+1}. {entity.name} ({entity.entity_type}) - 置信度: {entity.confidence:.2f}")

            # 显示部分关系
            if kg.relations:
                print("\n🔗 提取的关系样例:")
                for i, (rel_id, relation) in enumerate(list(kg.relations.items())[:5]):
                    head_name = relation.head_entity.name if relation.head_entity else "未知"
                    tail_name = relation.tail_entity.name if relation.tail_entity else "未知"
                    print(f"   {i+1}. {head_name} --[{relation.relation_type}]--> {tail_name}")

            # 保存到向量存储
            print("\n💾 保存知识图谱到向量存储...")
            await self.agraph.save_knowledge_graph()
            print("✅ 保存完成")

        except Exception as e:
            print(f"❌ 知识图谱构建失败: {e}")
            # 使用文本构建作为备选方案
            print("📝 使用简化文本构建备选方案...")
            await self.create_fallback_data()

    async def create_fallback_data(self):
        """创建备选演示数据"""
        from agraph.base.entities import Entity
        from agraph.base.relations import Relation
        from agraph.base.text import TextChunk
        from agraph.base.types import EntityType, RelationType

        # 创建演示实体
        entities = [
            Entity(name="苹果公司", entity_type=EntityType.ORGANIZATION,
                  description="美国科技公司", confidence=0.95),
            Entity(name="史蒂夫·乔布斯", entity_type=EntityType.PERSON,
                  description="苹果公司联合创始人", confidence=0.90),
            Entity(name="iPhone", entity_type=EntityType.PRODUCT,
                  description="智能手机产品", confidence=0.85),
            Entity(name="微软公司", entity_type=EntityType.ORGANIZATION,
                  description="软件公司", confidence=0.90),
            Entity(name="库比蒂诺", entity_type=EntityType.LOCATION,
                  description="苹果公司总部所在地", confidence=0.80)
        ]

        # 创建演示关系
        relations = [
            Relation(head_entity=entities[1], tail_entity=entities[0],
                    relation_type=RelationType.FOUNDED_BY, confidence=0.90),
            Relation(head_entity=entities[0], tail_entity=entities[2],
                    relation_type=RelationType.DEVELOPS, confidence=0.85),
            Relation(head_entity=entities[0], tail_entity=entities[4],
                    relation_type=RelationType.LOCATED_IN, confidence=0.80)
        ]

        # 创建演示文本块
        text_chunks = [
            TextChunk(content=text[:200] + "..." if len(text) > 200 else text,
                     title=f"演示文档{i+1}", source=f"demo_doc_{i+1}",
                     start_index=0, end_index=len(text))
            for i, text in enumerate(self.demo_texts[:5])
        ]

        # 添加到向量存储
        if self.agraph.vector_store:
            await self.agraph.vector_store.batch_add_entities(entities)
            await self.agraph.vector_store.batch_add_relations(relations)
            await self.agraph.vector_store.batch_add_text_chunks(text_chunks)

        print("✅ 备选演示数据创建完成")

    async def demo_vector_search(self):
        """演示向量检索功能"""
        print("\n🔍 4. 向量检索功能演示")
        print("-" * 30)

        # 定义搜索查询
        search_queries = [
            ("苹果公司", "实体"),
            ("创始人", "关系"),
            ("智能手机", "文本块")
        ]

        for query, search_type in search_queries:
            print(f"\n4.{search_queries.index((query, search_type)) + 1} 搜索'{query}' ({search_type})")

            try:
                if search_type == "实体":
                    results = await self.agraph.search_entities(query, top_k=3)
                    print(f"   找到 {len(results)} 个相关实体:")
                    for entity, score in results:
                        print(f"   - {entity.name} ({entity.entity_type}) [相似度: {score:.3f}]")

                elif search_type == "关系":
                    results = await self.agraph.search_relations(query, top_k=3)
                    print(f"   找到 {len(results)} 个相关关系:")
                    for relation, score in results:
                        head_name = relation.head_entity.name if relation.head_entity else "未知"
                        tail_name = relation.tail_entity.name if relation.tail_entity else "未知"
                        print(f"   - {head_name} --[{relation.relation_type}]--> {tail_name} [相似度: {score:.3f}]")

                else:  # 文本块
                    results = await self.agraph.search_text_chunks(query, top_k=3)
                    print(f"   找到 {len(results)} 个相关文本块:")
                    for chunk, score in results:
                        preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
                        print(f"   - {preview} [相似度: {score:.3f}]")

            except Exception as e:
                print(f"   ❌ 搜索失败: {e}")

    async def demo_intelligent_chat(self):
        """演示智能对话功能"""
        print("\n💬 5. 智能对话功能演示")
        print("-" * 30)

        # 预定义的问题和期望类型
        questions = [
            ("苹果公司是什么？", "简洁回答"),
            ("苹果公司的创始人是谁？", "详细回答"),
            ("iPhone什么时候发布的？", "简洁回答"),
            ("苹果公司和微软公司有什么关系？", "详细回答")
        ]

        conversation_history = []

        for i, (question, response_type) in enumerate(questions):
            print(f"\n5.{i+1} 问答演示")
            print(f"👤 用户: {question}")

            try:
                response = await self.agraph.chat(
                    question=question,
                    conversation_history=conversation_history,
                    entity_top_k=3,
                    relation_top_k=2,
                    text_chunk_top_k=3,
                    response_type=response_type
                )

                answer = response["answer"]
                print(f"🤖 助手: {answer}")

                # 显示检索上下文统计
                context = response["context"]
                entities_count = len(context.get("entities", []))
                relations_count = len(context.get("relations", []))
                chunks_count = len(context.get("text_chunks", []))

                print(f"📊 检索统计: {entities_count}个实体, {relations_count}个关系, {chunks_count}个文档")

                # 更新对话历史
                conversation_history.append({
                    "user": question,
                    "assistant": answer
                })

                # 显示部分检索结果
                if entities_count > 0:
                    print("🏷️  相关实体:", end=" ")
                    entity_names = [item["entity"].name for item in context["entities"][:2]]
                    print(", ".join(entity_names))

            except Exception as e:
                print(f"🤖 助手: 抱歉，处理您的问题时出现了错误: {e}")

        print(f"\n✅ 完成了 {len(questions)} 轮对话演示")

    async def demo_system_management(self):
        """演示系统管理功能"""
        print("\n⚙️ 6. 系统管理功能演示")
        print("-" * 30)

        try:
            # 获取系统统计
            stats = await self.agraph.get_stats()

            print("6.1 系统统计信息:")

            # 向量存储统计
            if "vector_store" in stats:
                vs_stats = stats["vector_store"]
                print("📊 向量存储:")
                for key, value in vs_stats.items():
                    print(f"   - {key}: {value}")

            # 知识图谱统计
            if "knowledge_graph" in stats:
                kg_stats = stats["knowledge_graph"]
                print("🕸️  知识图谱:")
                for key, value in kg_stats.items():
                    print(f"   - {key}: {value}")

            # 构建器统计
            if "builder" in stats:
                builder_stats = stats["builder"]
                print("🏗️  构建器:")
                if "build_status" in builder_stats:
                    build_status = builder_stats["build_status"]
                    print(f"   - 构建进度: {build_status.get('progress', 0):.1f}%")
                    print(f"   - 完成步骤: {build_status.get('completed_steps', 0)}/{build_status.get('total_steps', 6)}")

                if "cache_info" in builder_stats:
                    cache_info = builder_stats["cache_info"]
                    if "backend" in cache_info:
                        backend_info = cache_info["backend"]
                        print(f"   - 缓存文件: {backend_info.get('total_files', 0)}")
                        print(f"   - 缓存大小: {backend_info.get('total_size', 0)} bytes")

            # 演示属性检查
            print("\n6.2 系统状态:")
            print(f"✅ 已初始化: {self.agraph.is_initialized}")
            print(f"✅ 有知识图谱: {self.agraph.has_knowledge_graph}")
            print(f"📂 集合名称: {self.agraph.collection_name}")
            print(f"💾 存储类型: {self.agraph.vector_store_type}")

        except Exception as e:
            print(f"❌ 获取系统信息失败: {e}")

    async def cleanup(self):
        """清理资源"""
        print("\n🧹 7. 资源清理")
        print("-" * 30)

        try:
            # 清除数据（可选）
            print("清理演示数据...")
            # await self.agraph.clear_all()

            # 关闭系统
            print("关闭AGraph系统...")
            await self.agraph.close()

            # 清理演示文件
            print("清理演示文件...")
            if self.demo_data_dir.exists():
                import shutil
                shutil.rmtree(self.demo_data_dir)

            print("✅ 资源清理完成")

        except Exception as e:
            print(f"⚠️  清理过程中出现警告: {e}")


async def main():
    """主函数"""
    demo = AGraphDemo()

    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源清理
        if demo.agraph and demo.agraph.is_initialized:
            await demo.agraph.close()


if __name__ == "__main__":
    print("🎯 启动AGraph完整功能演示...")
    print("📌 提示: 这个演示将展示AGraph的所有核心功能")
    print("⏱️  演示大约需要2-3分钟时间")
    print()

    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7+版本")
        sys.exit(1)

    # 运行演示
    asyncio.run(main())
