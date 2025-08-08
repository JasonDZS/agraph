import asyncio
import os

from agraph import ChatKnowledgeRetriever
from agraph.builders import LLMGraphBuilder
from agraph.config import settings
from agraph.logger import logger
from agraph.storage import JsonVectorStorage

settings.workdir = "./workdir/llm_builder_example"  # 设置工作目录
os.makedirs(settings.workdir, exist_ok=True)  # 确保工作目录存在


async def build_knowledge_graph():
    # 创建图构建器
    builder = LLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,  # 指定LLM模型
        embedding_model=settings.EMBEDDING_MODEL,  # 指定嵌入模型
        vector_storage=JsonVectorStorage(),  # 使用JSON向量存储
        temperature=0.1,
    )

    # 从文本构建知识图谱
    texts = [
        "苹果公司是由史蒂夫·乔布斯创立的科技公司。",
        "iPhone是苹果公司的旗舰智能手机产品。",
        "史蒂夫·乔布斯在2011年之前担任苹果公司CEO。",
    ]

    graph = await builder.build_graph(texts=texts, graph_name="科技公司")

    logger.info(f"构建了包含 {len(graph.entities)} 个实体和 {len(graph.relations)} 个关系的知识图谱")

    # 打印实体信息用于调试
    logger.info("\n=== 构建的实体 ===")
    for entity_id, entity in graph.entities.items():
        logger.info(f"- {entity.name} ({entity.entity_type.value}): {entity.description}")

    # 打印关系信息用于调试
    logger.info(f"\n=== 构建的关系 ({len(graph.relations)}) ===")
    for relation_id, relation in graph.relations.items():
        if relation.head_entity and relation.tail_entity:
            logger.info(
                f"- {relation.head_entity.name} --{relation.relation_type.value}--> {relation.tail_entity.name}: {relation.description}"
            )
    return graph, builder


# 运行示例
if __name__ == "__main__":
    graph, builder = asyncio.run(build_knowledge_graph())
    # 使用构建的知识图谱和构建器
    retriever = ChatKnowledgeRetriever()
    result = retriever.chat("谁创立了苹果公司？")

    logger.info("\n=== 检索结果调试 ===")
    logger.info(f"检索到的实体数量: {len(result.get('entities', []))}")
    logger.info(f"检索到的关系数量: {len(result.get('relations', []))}")
    if result.get("entities"):
        logger.info("检索到的实体:")
        for entity in result["entities"]:
            if hasattr(entity, "entity"):
                logger.info(f"  - {entity.entity.name}: score={entity.score}")

    logger.info("\n=== 最终结果 ===")
    print(result)
