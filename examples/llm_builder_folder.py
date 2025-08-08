import asyncio
import logging
import os

from agraph.builders import LLMGraphBuilder
from agraph.chunker import TokenChunker
from agraph.config import settings
from agraph.processer.factory import DocumentProcessorFactory
from agraph.retrieval import ChatKnowledgeRetriever
from agraph.storage import JsonVectorStorage

# 配置日志系统以显示详细信息
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

settings.workdir = "./workdir/llm_builder_folder"  # 设置工作目录
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

    # 创建Token分割器
    chunker = TokenChunker(
        chunk_size=800,  # 每个chunk最大800个token
        chunk_overlap=100,  # chunk之间重叠100个token
        model=settings.LLM_MODEL,  # 使用配置中的模型来计算token
    )

    document_paths = [
        "./examples/documents/company_info.txt",
        "./examples/documents/products.json",
        "./examples/documents/team.html",
        "./examples/documents/research_papers.csv",
        "./examples/documents/technology_stack.md",
    ]

    all_chunks = []
    processor_factory = DocumentProcessorFactory()

    print("开始处理和分割文档...")
    for doc_path in document_paths:
        print(f"处理文档: {doc_path}")
        processor = processor_factory.get_processor(doc_path)
        content = processor.process(doc_path)

        # 使用chunker分割文档内容
        document_text = f"文档: {doc_path}\n{content}"
        chunks = chunker.split_text(document_text)

        print(f"  - 文档被分割为 {len(chunks)} 个chunks")
        for i, chunk in enumerate(chunks):
            token_count = chunker.count_tokens(chunk)
            print(f"    chunk {i+1}: {token_count} tokens, {len(chunk)} 字符")

        all_chunks.extend(chunks)

    print(f"\n总计处理了 {len(all_chunks)} 个文本块")

    # 使用分割后的chunks构建知识图谱
    graph = await builder.build_graph(texts=all_chunks, graph_name="科技公司")

    print(f"构建了包含 {len(graph.entities)} 个实体和 {len(graph.relations)} 个关系的知识图谱")
    return graph, builder


# 运行示例
if __name__ == "__main__":
    graph, builder = asyncio.run(build_knowledge_graph())
    # 使用构建的知识图谱和构建器
    retriever = ChatKnowledgeRetriever()
    result = retriever.chat("公司总部位置")
    print(result)
