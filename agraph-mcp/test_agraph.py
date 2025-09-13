#!/usr/bin/env python3
"""
AGraph 快速开始示例 (统一配置版本)

这是一个使用新统一配置系统的AGraph示例，展示了简化的配置方式：
1. 统一的Settings配置管理
2. 简化的AGraph初始化 (只需传入settings)
3. 从文本构建知识图谱
4. 进行语义搜索
5. 智能问答对话

适合了解AGraph统一配置系统的强大功能和简化使用方式。
"""

import asyncio
import sys
import time
from pathlib import Path
from agraph import AGraph
from agraph.config import get_settings, update_settings

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置工作目录并更新配置
workdir = "../workdir/agraph_quickstart-cache"
settings = get_settings()
settings = update_settings({"workdir": workdir, "llm_config": {"model": "Qwen/Qwen3-32B"}})

print(f"✅ 使用工作目录: {settings.workdir}")
print(f"📋 配置概览:")
print(f"   - LLM模型: {settings.llm.model}")
print(f"   - 嵌入模型: {settings.embedding.model}")
print(f"   - 最大并发: {settings.max_current}")
print(f"   - 缓存设置: TTL={settings.cache_config['cache_ttl']}s")

async def quickstart_demo():
    """AGraph快速开始演示"""

    # 1. 创建AGraph实例并初始化 (自动配置保存)
    print("\n📦 1. 初始化AGraph (自动配置保存)...")
    print("   🔧 使用统一Settings配置系统")
    print("   ⚡ 简化初始化过程 (自动保存配置到workdir)")
    print("   📋 自动从配置数据中获取所有参数")
    async with AGraph(collection_name="quickstart_demo") as agraph:
        await agraph.initialize()
        print("✅ AGraph初始化成功 (使用统一配置系统)")
        print(f"   🗺️ 持久化目录: {agraph.persist_directory}")
        print(f"   💾 向量存储类型: {agraph.vector_store_type}")
        print(f"   🧠 启用知识图谱: {agraph.enable_knowledge_graph}")
        print(f"   🌐 嵌入提供者: {settings.embedding.provider}")

        # 2. 语义搜索演示
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

                # 显示检索统计和配置信息
                context = chunk_data['context']
                entity_count = len(context.get('entities', []))
                chunk_count = len(context.get('text_chunks', []))
                print(f"   📊 检索了 {entity_count} 个实体, {chunk_count} 个文档")
                print(f"   ⚙️ 使用的LLM配置: {settings.llm.model} (temp={settings.llm.temperature})")

            except Exception as e:
                print(f"🤖 回答: 抱歉，无法回答这个问题: {e}")

        # 5. 系统信息和配置概览
        print("\n📊 5. 系统信息和统一配置...")
        stats = await agraph.get_stats()

        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            print("向量存储:")
            print(f"   - 实体: {vs_stats.get('entities', 0)}")
            print(f"   - 关系: {vs_stats.get('relations', 0)}")
            print(f"   - 文本块: {vs_stats.get('text_chunks', 0)}")

        # 显示统一配置概览
        print("\n统一配置概览:")
        all_configs = settings.get_all_configs()
        print(f"   - 核心配置: {all_configs['core']}")
        print(f"   - 缓存配置: {all_configs['unified_views']['cache']}")
        print(f"   - 处理配置: {all_configs['unified_views']['processing']}")

        print(f"\n系统状态: {agraph}")

    print("\n✅ 快速开始演示完成 (统一配置版本)!")
    print("🔧 体验了AGraph统一配置系统的强大功能:")
    print("   - 简化的初始化过程 (只需传入settings)")
    print("   - 统一的配置管理和访问")
    print("   - 自动化的参数派生和优化")


def main():
    """主函数"""
    print("🎯 启动AGraph快速开始演示 (统一配置版本)...")

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
