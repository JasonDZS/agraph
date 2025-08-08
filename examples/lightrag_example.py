"""
LightRAG Knowledge Graph Construction Example

This example demonstrates how to build, search, update, and export knowledge graphs
using the LightRAG framework. It showcases the complete workflow from document
processing to graph construction and querying.

Example Usage:
    python examples/lightrag_example.py

Requirements:
    - lightrag package installed (pip install lightrag)
    - OpenAI API key configured in environment
    - Sample documents in examples/documents/ directory
"""

import asyncio
import logging
import os
from typing import Tuple

from agraph.builders.lightrag_builder import LightRAGBuilder
from agraph.config import settings
from agraph.graph import KnowledgeGraph
from agraph.processer.factory import DocumentProcessorFactory

# Configure logging to show detailed information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set up working directory
settings.workdir = "./workdir/lightrag_example"
os.makedirs(settings.workdir, exist_ok=True)


async def build_knowledge_graph() -> Tuple[KnowledgeGraph, LightRAGBuilder]:
    """
    Build a knowledge graph using LightRAG from document sources.

    This function demonstrates the basic workflow of:
    1. Creating a LightRAG builder instance
    2. Processing multiple document formats
    3. Building the knowledge graph from processed texts
    4. Displaying entity and relation statistics

    Returns:
        Tuple[KnowledgeGraph, LightRAGBuilder]: A tuple containing the constructed knowledge graph and builder instance
    """
    # Create LightRAG graph builder
    builder = LightRAGBuilder(working_dir=os.path.join(settings.workdir, "lightrag_storage"))

    # Define document paths to process
    document_paths = [
        "./examples/documents/company_info.txt",
        "./examples/documents/products.json",
        "./examples/documents/team.html",
        "./examples/documents/technology_stack.md",
    ]

    texts = []
    processor_factory = DocumentProcessorFactory()

    # Process documents using appropriate processors
    for doc_path in document_paths:
        if os.path.exists(doc_path):
            processor = processor_factory.get_processor(doc_path)
            content = processor.process(doc_path)
            texts.append(f"Document: {doc_path}\n{content}")
        else:
            print(f"Warning: Document {doc_path} not found, skipping")

    # Use example texts if no documents are found
    if not texts:
        print("No documents found, using example texts")
        texts = [
            "Company: TechCorp is an artificial intelligence technology company founded in 2020.",
            "Products: The company's main products include intelligent customer service systems, "
            "data analysis platforms, and machine learning tools.",
            "Team: Founder John Smith is an AI expert, CTO Jane Doe is responsible for technical architecture.",
            "Technology: The company uses Python, machine learning, and deep learning technologies.",
        ]

    # Build knowledge graph from processed texts
    graph = await builder.build_graph(texts=texts, graph_name="LightRAG_TechCorp")

    print(f"Built knowledge graph with {len(graph.entities)} entities and {len(graph.relations)} relations")

    # Display sample entity information
    if graph.entities:
        print("\nEntity Examples:")
        for i, (entity_id, entity) in enumerate(graph.entities.items()):
            if i >= 5:  # Show only first 5 entities
                break
            description = entity.description[:100] + "..." if len(entity.description) > 100 else entity.description
            print(f"  - {entity.name} ({entity.entity_type}): {description}")

    # Display sample relation information
    if graph.relations:
        print("\nRelation Examples:")
        for i, (relation_id, relation) in enumerate(graph.relations.items()):
            if i >= 5:  # Show only first 5 relations
                break
            if relation.head_entity and relation.tail_entity:
                print(f"  - {relation.head_entity.name} -> {relation.tail_entity.name}")

    return graph, builder


async def search_knowledge_graph(builder):
    """
    Demonstrate knowledge graph search capabilities.

    This function showcases different search modes available in LightRAG:
    - Naive: Simple keyword-based search
    - Local: Local neighborhood search
    - Global: Global graph search
    - Hybrid: Combined local and global search

    Args:
        builder: The LightRAG builder instance with initialized graph
    """
    print("\n=== Knowledge Graph Search Demo ===")

    # Sample search queries
    search_queries = [
        "What are the company's main products?",
        "Who is the founder?",
        "What technologies are used?",
        "When was the company founded?",
    ]

    # Test each query with different search types
    for query in search_queries:
        try:
            print(f"\nQuery: {query}")
            # Use different search modes
            for search_type in ["naive", "local", "global", "hybrid"]:
                result = await builder.lightrag_core.asearch_graph(query, search_type)
                result_text = result.get("result", "No results")
                # Truncate long results for display
                display_result = result_text[:100] + "..." if len(result_text) > 100 else result_text
                print(f"  {search_type} search: {display_result}")
        except Exception as e:
            print(f"Search failed for '{query}': {e}")


async def main():
    """
    Main function that orchestrates the complete LightRAG workflow.

    Demonstrates:
    1. Knowledge graph construction from documents
    2. Graph search capabilities
    3. Graph updates with new information
    4. Graph export and statistics

    Includes proper error handling and resource cleanup.
    """
    builder = None

    try:
        print("=== LightRAG Knowledge Graph Construction Example ===")

        # Build initial knowledge graph
        graph, builder = await build_knowledge_graph()

        # Demonstrate search functionality
        await search_knowledge_graph(builder)

    except Exception as e:
        print(f"Example execution failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up resources
        if builder:
            builder.cleanup()
            print("Resources cleaned up")


if __name__ == "__main__":
    """
    Entry point for the LightRAG example.

    Run this script directly to see the complete workflow in action:
        python examples/lightrag_example.py
    """
    asyncio.run(main())
