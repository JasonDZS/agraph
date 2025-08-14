# AGraph Tutorial

This tutorial will guide you through learning the core features of the AGraph knowledge graph
toolkit with practical examples.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Core Features](#core-features)
4. [Advanced Usage](#advanced-usage)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Environment Setup

### System Requirements

- Python 3.10+
- OpenAI API key (for LLM and embedding services)

### Install Dependencies

```bash
# Development installation
make install-dev

# Or basic installation
pip install -e .
```

### Environment Configuration

Set environment variables or create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, default value
OPENAI_MODEL=gpt-3.5-turbo  # Optional, default value
```

## Quick Start

```python
import asyncio
from pathlib import Path
from agraph import AGraph, get_settings

# 1. Basic Setup
# Configure working directory
settings = get_settings()
settings.workdir = str(Path("workdir/my_project"))

async def main():
    # 2. Initialize AGraph Instance
    agraph = AGraph(
        collection_name="my_knowledge_graph",
        persist_directory=settings.workdir,
        vector_store_type="chroma",  # Choose vector store type
        use_openai_embeddings=True  # Use OpenAI embeddings
    )
    await agraph.initialize()
    print("✅ AGraph initialized successfully")

    # 3. Prepare Document Data
    sample_texts = [
        "AGraph is a toolkit for building knowledge graphs.",
        "It supports semantic search and intelligent Q&A.",
        "You can build knowledge graphs from text documents."
    ]

    # 4. Build Knowledge Graph
    knowledge_graph = await agraph.build_from_texts(
        texts=sample_texts,
        graph_name="Sample Knowledge Graph",
        graph_description="Knowledge graph built from sample texts",
        use_cache=True,  # Enable caching for acceleration
        save_to_vector_store=True  # Save to vector store
    )

    print(f"📊 Build completed: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relations)} relations")

    entities = await agraph.search_entities("company", top_k=5)
    for entity, score in entities:
        print(f"Entity: {entity.name} ({entity.entity_type}) - Similarity: {score:.3f}")

    text_chunks = await agraph.search_text_chunks("artificial intelligence technology", top_k=3)
    for chunk, score in text_chunks:
        preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        print(f"Text: {preview} - Similarity: {score:.3f}")
    # Properly close the AGraph instance
    await agraph.close()

asyncio.run(main())
```

## Core Features

### Semantic Search

#### Search Entities

```python
# Search entities by name
entities = await agraph.search_entities("company", top_k=5)
for entity, score in entities:
    print(f"Entity: {entity.name} ({entity.entity_type}) - Similarity: {score:.3f}")
```

#### Search Text Chunks

```python
# Search text chunks by content
text_chunks = await agraph.search_text_chunks("artificial intelligence technology", top_k=3)
for chunk, score in text_chunks:
    preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
    print(f"Text: {preview} - Similarity: {score:.3f}")
```

### Intelligent Q&A

#### Basic Q&A

```python
# Simple question answering
question = "What is the company's main business?"
response = await agraph.chat(question)
print(f"Answer: {response}")
```

#### Streaming Q&A

```python
# Streaming response, real-time display of generation process
async for chunk_data in await agraph.chat(question, stream=True):
    if chunk_data["chunk"]:
        print(chunk_data["chunk"], end="", flush=True)
    if chunk_data["finished"]:
        print(f"\n✅ Complete answer: {chunk_data['answer']}")

        # Display retrieval context information
        context = chunk_data['context']
        entities_used = len(context.get('entities', []))
        chunks_used = len(context.get('text_chunks', []))
        print(f"📊 Used {entities_used} entities, {chunks_used} document chunks")
        break
```

### Graph Analysis

#### View Statistics

```python
# Get system statistics
stats = await agraph.get_stats()

if 'vector_store' in stats:
    vs_stats = stats['vector_store']
    print("Vector store statistics:")
    print(f"  - Entities: {vs_stats.get('entities', 0)}")
    print(f"  - Relations: {vs_stats.get('relations', 0)}")
    print(f"  - Text chunks: {vs_stats.get('text_chunks', 0)}")
```

#### Entity Relationship Exploration

```python
# Get relationships of specific entities
entity_name = "company"
entities = await agraph.search_entities(entity_name, top_k=1)
if entities:
    entity = entities[0][0]
    print(f"Entity: {entity.name}")
    print(f"Type: {entity.entity_type}")
    print(f"Properties: {entity.properties}")
    print(f"Aliases: {entity.aliases}")
```

## Advanced Usage

### Caching Mechanism

AGraph provides intelligent caching to improve performance:

```python
# Enable cache build
knowledge_graph = await agraph.build_from_texts(
    texts=sample_texts,
    graph_name="cached_graph",
    use_cache=True,  # Results will be cached after first build
    cache_ttl=3600   # Cache for 1 hour
)

# Subsequent builds with the same texts will use cache directly
```

### Persistent Storage

```python
# Specify persistence directory
async with AGraph(
    collection_name="persistent_graph",
    persist_directory="/path/to/storage",
    vector_store_type="chroma"
) as agraph:
    # Data will be automatically saved to the specified directory
    # It will be automatically loaded on next startup
    pass
```

### Knowledge Graph Construction Control

AGraph supports controlling whether to enable knowledge graph construction. When disabled,
the system only performs text chunking and vector storage, skipping entity, relation, and
cluster extraction:

#### Enable Knowledge Graph (Default Behavior)

```python
# Complete knowledge graph construction, including entity, relation, cluster extraction
agraph = AGraph(
    collection_name="full_knowledge_graph",
    enable_knowledge_graph=True  # Default value, can be omitted
)

# Build complete knowledge graph
kg = await agraph.build_from_texts(texts)
print(f"Entities: {len(kg.entities)}")      # > 0
print(f"Relations: {len(kg.relations)}")     # > 0
print(f"Clusters: {len(kg.clusters)}")      # > 0
print(f"Text chunks: {len(kg.text_chunks)}")  # > 0
```

#### Disable Knowledge Graph (Text-Only Mode)

```python
# Only perform text processing and vector storage, suitable for pure document retrieval scenarios
agraph = AGraph(
    collection_name="text_only_mode",
    enable_knowledge_graph=False  # Key setting
)

# Only creates text chunks, skips entity/relation/cluster extraction
kg = await agraph.build_from_texts(texts)
print(f"Entities: {len(kg.entities)}")      # 0
print(f"Relations: {len(kg.relations)}")     # 0
print(f"Clusters: {len(kg.clusters)}")      # 0
print(f"Text chunks: {len(kg.text_chunks)}")  # > 0, preserves text chunking functionality

# Still supports text chunk-based search and Q&A
results = await agraph.search_text_chunks("query content")
response = await agraph.chat("user question")
```

#### Use Case Comparison

| Feature | Knowledge Graph Enabled | Knowledge Graph Disabled |
|---------|-------------------------|---------------------------|
| Text Chunking | ✅ | ✅ |
| Entity Extraction | ✅ | ❌ |
| Relation Extraction | ✅ | ❌ |
| Cluster Analysis | ✅ | ❌ |
| Semantic Search | ✅ | ✅ (text chunks only)|
| Intelligent Q&A | ✅ | ✅ (based on text chunks)|
| Build Speed | Slower | Fast |
| Storage Overhead | Larger | Smaller |

#### Performance and Resource Comparison

```python
# Test performance differences between two modes
import time

# Knowledge graph mode
start = time.time()
agraph_kg = AGraph(enable_knowledge_graph=True)
kg_full = await agraph_kg.build_from_texts(large_texts)
kg_time = time.time() - start
print(f"Knowledge graph mode time: {kg_time:.2f}s")

# Text-only mode
start = time.time()
agraph_text = AGraph(enable_knowledge_graph=False)
kg_text = await agraph_text.build_from_texts(large_texts)
text_time = time.time() - start
print(f"Text-only mode time: {text_time:.2f}s")
print(f"Speed improvement: {kg_time/text_time:.1f}x")
```

### Custom Configuration

```python
from agraph import get_settings

settings = get_settings()
# Customize LLM model
settings.openai_model = "gpt-4"
# Customize embedding model
settings.embedding_model = "text-embedding-ada-002"
# Customize text chunking size
settings.chunk_size = 1000
settings.chunk_overlap = 200
```

## Best Practices

### 1. Choose the Right Construction Mode

#### When to Enable Knowledge Graph

```python
# Scenarios suitable for enabling knowledge graph construction:
scenarios_for_kg = [
    "Need to analyze complex relationships between entities",
    "Want to perform knowledge graph visualization",
    "Need precise search based on entities and relationships",
    "Want to perform knowledge reasoning and path finding",
    "Documents contain rich structured information",
    "Need to perform cluster analysis and topic discovery"
]

# Example: Analyzing company organizational structure documents
agraph = AGraph(
    collection_name="company_structure",
    enable_knowledge_graph=True  # Need to extract personnel, department, position relationships
)
```

#### When to Disable Knowledge Graph

```python
# Scenarios suitable for disabling knowledge graph construction:
scenarios_for_text_only = [
    "Pure document retrieval and similarity search",
    "Large-scale document fast indexing",
    "Resource-constrained environment deployment",
    "Primarily keyword-based search",
    "Relatively simple document structure",
    "Need for rapid prototype validation"
]

# Example: Building FAQ knowledge base
agraph = AGraph(
    collection_name="faq_database",
    enable_knowledge_graph=False  # Only need Q&A matching, no entity relationships
)
```

#### Dynamic Mode Selection

```python
def choose_mode_by_content(texts):
    """Dynamically choose construction mode based on document content"""

    # Simple heuristic rules
    total_length = sum(len(text) for text in texts)
    avg_length = total_length / len(texts) if texts else 0

    # Detect if contains structured information
    structured_indicators = ["company", "department", "manager", "project", "product", "client"]
    structured_score = sum(
        1 for text in texts
        for indicator in structured_indicators
        if indicator in text
    ) / len(texts)

    if avg_length > 1000 and structured_score > 2:
        return True  # Enable knowledge graph
    else:
        return False  # Text-only mode

# Usage example
enable_kg = choose_mode_by_content(document_texts)
agraph = AGraph(
    collection_name="adaptive_mode",
    enable_knowledge_graph=enable_kg
)
```

### 2. Document Preprocessing

```python
def preprocess_texts(texts):
    """Text preprocessing best practices"""
    processed = []
    for text in texts:
        # Clean whitespace
        text = text.strip()
        # Filter out too short texts
        if len(text) < 50:
            continue
        # Normalize encoding
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        processed.append(text)
    return processed

sample_texts = preprocess_texts(raw_texts)
```

### 3. Error Handling

```python
async def robust_build():
    try:
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="robust_graph",
            use_cache=True
        )
        return knowledge_graph
    except Exception as e:
        print(f"Build failed: {e}")
        # Fallback handling or retry logic
        return None
```

### 4. Performance Optimization

```python
# Batch processing of large datasets
async def process_large_dataset(texts, batch_size=10):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        graph = await agraph.build_from_texts(
            texts=batch,
            graph_name=f"batch_{i//batch_size}",
            use_cache=True
        )
        results.append(graph)
    return results
```

### 5. Quality Monitoring

```python
async def monitor_quality():
    stats = await agraph.get_stats()

    # Check if entity count is reasonable
    entity_count = stats['vector_store'].get('entities', 0)
    text_chunks = stats['vector_store'].get('text_chunks', 0)

    if entity_count == 0:
        print("⚠️ Warning: No entities extracted")
    elif entity_count / text_chunks < 0.1:
        print("⚠️ Warning: Entity density too low, may need to adjust extraction parameters")
```

## Troubleshooting

### Common Issues

#### 1. Initialization Failure

```python
# Check API key
import os
if not os.getenv('OPENAI_API_KEY'):
    print("❌ Please set OPENAI_API_KEY environment variable")

# Check network connection
try:
    await agraph.initialize()
except Exception as e:
    print(f"Initialization failed: {e}")
    # Possibly network issues or API quota problems
```

#### 2. Slow Build Speed

```python
# Enable caching
knowledge_graph = await agraph.build_from_texts(
    texts=sample_texts,
    use_cache=True  # Important!
)

# Reduce text volume
if len(sample_texts) > 100:
    sample_texts = sample_texts[:100]  # Test with small dataset first
```

#### 3. Out of Memory

```python
# Process in batches
batch_size = 5  # Adjust based on system memory
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    # Process batch
```

#### 4. Knowledge Graph Mode Issues

```python
# Issue: No entities extracted in knowledge graph mode
async def debug_kg_mode():
    agraph = AGraph(enable_knowledge_graph=True)
    kg = await agraph.build_from_texts(texts)

    if len(kg.entities) == 0:
        print("⚠️ No entities extracted, possible reasons:")
        print("1. LLM API configuration issues")
        print("2. Text content lacks structured information")
        print("3. Entity confidence threshold too high")

        # Solution: Lower threshold or switch to text mode
        agraph_text = AGraph(enable_knowledge_graph=False)
        kg_text = await agraph_text.build_from_texts(texts)
        print(f"Text mode created {len(kg_text.text_chunks)} text chunks")

# Issue: Empty vector database
async def debug_empty_vectordb():
    agraph = AGraph(enable_knowledge_graph=False)
    kg = await agraph.build_from_texts(texts)

    # Check build results
    stats = await agraph.get_stats()
    print(f"Vector store statistics: {stats}")

    if stats.get('vector_store', {}).get('text_chunks', 0) == 0:
        print("❌ Vector database is empty, check:")
        print("1. Is texts empty or content too short")
        print("2. Is save_to_vector_store set to True")
        print("3. Is text chunking working properly")

# Issue: Data inconsistency after mode switching
async def handle_mode_switching():
    # Clear old data
    agraph = AGraph(collection_name="test", enable_knowledge_graph=True)
    await agraph.clear_all()

    # Rebuild
    kg = await agraph.build_from_texts(texts)
    print(f"Rebuild completed after mode switching")
```

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check vector store status
stats = await agraph.get_stats()
print(f"Debug info: {stats}")

# Test simple queries
simple_entities = await agraph.search_entities("test", top_k=1)
print(f"Test query results: {simple_entities}")
```

## Complete Examples

### Basic Example

Here's a complete working example:

```python
#!/usr/bin/env python3
import asyncio
from pathlib import Path
from agraph import AGraph, get_settings

async def complete_example():
    # Configuration
    settings = get_settings()
    settings.workdir = str(Path("workdir/tutorial"))

    # Sample documents
    sample_texts = [
        "TechCorp is an AI company founded in 2018. The company focuses on natural language processing and computer vision technologies.",
        "The company headquarters is located in Beijing Zhongguancun, with a total of 120 employees, 80% of whom are R&D personnel.",
        "TechCorp's main products include intelligent customer service systems, document analysis platforms, and image recognition APIs.",
        "The company completed Series B funding in 2023, raising $50 million led by Sequoia Capital."
    ]

    # Initialize and build
    async with AGraph(
        collection_name="techcorp_knowledge",
        persist_directory=settings.workdir,
        vector_store_type="chroma",
        use_openai_embeddings=True
    ) as agraph:
        await agraph.initialize()

        # Build knowledge graph
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="TechCorp Knowledge Graph",
            graph_description="Knowledge graph about TechCorp company",
            use_cache=True,
            save_to_vector_store=True
        )

        print(f"✅ Build completed: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relations)} relations")

        # Semantic search
        entities = await agraph.search_entities("company", top_k=3)
        print("\n🔍 Entity search results:")
        for entity, score in entities:
            print(f"  - {entity.name} ({entity.entity_type})")

        # Intelligent Q&A
        questions = [
            "When was TechCorp founded?",
            "How many employees does the company have?",
            "What are the main products?"
        ]

        print("\n💬 Q&A demonstration:")
        for question in questions:
            print(f"\n❓ {question}")
            response = await agraph.chat(question)
            print(f"🤖 {response}")

        # System statistics
        stats = await agraph.get_stats()
        print(f"\n📊 System statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(complete_example())
```

### Knowledge Graph Mode Comparison Example

This example demonstrates the difference between enabling and disabling knowledge graphs:

```python
#!/usr/bin/env python3
import asyncio
import time
from pathlib import Path
from agraph import AGraph, get_settings

async def compare_modes_example():
    """Compare differences between knowledge graph mode and text-only mode"""

    # Prepare test documents
    documents = [
        "Apple Inc. is an American multinational technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976.",
        "Microsoft Corporation is an American multinational technology company founded by Bill Gates and Paul Allen on April 4, 1975.",
        "Google LLC is an American multinational technology company founded by Larry Page and Sergey Brin on September 4, 1998.",
        "Tesla, Inc. is an American electric vehicle and energy company founded by Martin Eberhard and Marc Tarpenning in 2003."
    ]

    print("🔄 Starting comparison of different build modes")
    print("=" * 60)

    # Mode 1: Full knowledge graph construction
    print("\n📊 Mode 1: Knowledge Graph Enabled")
    start_time = time.time()

    async with AGraph(
        collection_name="kg_mode_demo",
        enable_knowledge_graph=True,
        persist_directory="./demo_kg"
    ) as agraph_kg:
        await agraph_kg.initialize()

        kg_full = await agraph_kg.build_from_texts(
            texts=documents,
            graph_name="Tech Companies Knowledge Graph",
            save_to_vector_store=True
        )

        kg_build_time = time.time() - start_time

        print(f"⏱️  Build time: {kg_build_time:.2f}s")
        print(f"📈 Build results:")
        print(f"   • Entities: {len(kg_full.entities)}")
        print(f"   • Relations: {len(kg_full.relations)}")
        print(f"   • Clusters: {len(kg_full.clusters)}")
        print(f"   • Text chunks: {len(kg_full.text_chunks)}")

        # Test entity search
        entities = await agraph_kg.search_entities("company", top_k=3)
        print(f"\n🔍 Entity search 'company' found {len(entities)} results:")
        for entity, score in entities[:2]:
            print(f"   • {entity.name} ({entity.entity_type}) - Similarity: {score:.3f}")

        # Test intelligent Q&A
        question = "When was Apple founded?"
        response = await agraph_kg.chat(question)
        print(f"\n💬 Q&A test: {question}")
        print(f"🤖 Answer: {response['answer'][:100]}...")

        stats_kg = await agraph_kg.get_stats()

    # Mode 2: Text-only mode
    print("\n" + "=" * 60)
    print("📄 Mode 2: Knowledge Graph Disabled (Text-Only Mode)")
    start_time = time.time()

    async with AGraph(
        collection_name="text_mode_demo",
        enable_knowledge_graph=False,
        persist_directory="./demo_text"
    ) as agraph_text:
        await agraph_text.initialize()

        kg_text = await agraph_text.build_from_texts(
            texts=documents,
            graph_name="Tech Companies Document Library",
            save_to_vector_store=True
        )

        text_build_time = time.time() - start_time

        print(f"⏱️  Build time: {text_build_time:.2f}s")
        print(f"📈 Build results:")
        print(f"   • Entities: {len(kg_text.entities)}")
        print(f"   • Relations: {len(kg_text.relations)}")
        print(f"   • Clusters: {len(kg_text.clusters)}")
        print(f"   • Text chunks: {len(kg_text.text_chunks)}")

        # Test text chunk search
        chunks = await agraph_text.search_text_chunks("Apple", top_k=3)
        print(f"\n🔍 Text search 'Apple' found {len(chunks)} results:")
        for chunk, score in chunks[:2]:
            preview = chunk.content[:50] + "..." if len(chunk.content) > 50 else chunk.content
            print(f"   • {preview} - Similarity: {score:.3f}")

        # Test intelligent Q&A (based on text chunks)
        question = "When was Apple founded?"
        response = await agraph_text.chat(question)
        print(f"\n💬 Q&A test: {question}")
        print(f"🤖 Answer: {response['answer'][:100]}...")

        stats_text = await agraph_text.get_stats()

    # Performance comparison summary
    print("\n" + "=" * 60)
    print("📊 Performance Comparison Summary")
    print(f"Build speed improvement: {kg_build_time/text_build_time:.1f}x (text mode is faster)")
    print(f"Knowledge graph mode: {kg_build_time:.2f}s")
    print(f"Text-only mode: {text_build_time:.2f}s")

    print(f"\n💾 Storage comparison:")
    if 'vector_store' in stats_kg and 'vector_store' in stats_text:
        kg_total = sum(stats_kg['vector_store'].values())
        text_total = sum(stats_text['vector_store'].values())
        print(f"Knowledge graph mode storage items: {kg_total}")
        print(f"Text-only mode storage items: {text_total}")

    print(f"\n✨ Feature comparison:")
    print("Knowledge graph mode: Supports entity search, relationship analysis, cluster discovery")
    print("Text-only mode: Supports document retrieval, similarity search, fast Q&A")

    print(f"\n🎯 Recommended use cases:")
    print("• Knowledge graph mode → Complex applications requiring deep entity relationship analysis")
    print("• Text-only mode → Lightweight applications for fast document retrieval and Q&A")

if __name__ == "__main__":
    asyncio.run(compare_modes_example())
```

Run this example:

```bash
python complete_example.py
```

## Summary

AGraph provides a simple yet powerful interface for building and querying knowledge graphs.
Through this tutorial, you have learned:

- ✅ Environment setup and initialization
- ✅ Building knowledge graphs from text
- ✅ Semantic search and intelligent Q&A
- ✅ **Knowledge Graph Construction Control** - New feature highlight!
- ✅ Choosing the right construction mode (full KG vs text-only)
- ✅ Performance optimization and error handling
- ✅ Best practices and debugging techniques

### 🎯 Key New Feature: Optional Knowledge Graph Construction

This tutorial highlights important new features in AGraph v0.2+:

- **Flexible Mode Selection**: Control build behavior through the `enable_knowledge_graph` parameter
- **Text-Only Mode**: Fast document indexing, skipping complex entity relationship extraction
- **Full KG Mode**: Deep analysis, extracting entities, relations, clusters
- **Seamless Switching**: Same API, different processing logic
- **Performance Optimization**: Choose the most suitable mode based on requirements

### 📈 Usage Recommendations

| Application Scenario | Recommended Mode | Advantages |
|---------------------|------------------|------------|
| Document Retrieval System | Text-only mode | Fast, lightweight |
| FAQ Knowledge Base | Text-only mode | Simple and efficient |
| Enterprise Knowledge Management | Full KG mode | Deep analysis |
| Research Literature Analysis | Full KG mode | Relationship mining |
| Rapid Prototype Validation | Text-only mode | Fast development |

Now you can flexibly use AGraph in your own projects based on specific requirements!

## Additional Resources

- [API Reference Documentation](../source/modules.rst)
- [Vector Database Tutorial](vectordb_tutorial.md)
- [Import/Export Features](import_export_tutorial.md)
- [Custom Vector Database Guide](custom_vectordb_guide.md)
