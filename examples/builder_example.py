"""
Example usage of KnowledgeGraphBuilder.
"""

from agraph import KnowledgeGraphBuilder, BuilderConfig


def basic_example():
    """Basic usage example."""
    print("=== Basic KnowledgeGraph Builder Example ===")

    # Create builder with default configuration
    builder = KnowledgeGraphBuilder()

    # Build knowledge graph from text
    texts = [
        "Apple Inc. is a technology company founded by Steve Jobs. The company is headquartered in Cupertino.",
        "Steve Jobs was the CEO of Apple Inc. He revolutionized the technology industry with innovative products.",
        "Cupertino is a city in California where many technology companies are located."
    ]

    kg = builder.build_from_text(
        texts=texts,
        graph_name="Technology Companies",
        graph_description="Knowledge graph about technology companies and key figures"
    )

    # Print statistics
    stats = kg.get_graph_statistics()
    print(f"Built knowledge graph with:")
    print(f"  - {stats['total_entities']} entities")
    print(f"  - {stats['total_relations']} relations")
    print(f"  - {stats['total_clusters']} clusters")
    print(f"  - {stats['total_text_chunks']} text chunks")


def cached_example():
    """Example with caching enabled."""
    print("\n=== Cached Builder Example ===")

    # Create builder with custom cache directory
    config = BuilderConfig(
        cache_dir="./example_cache",
        chunk_size=500,
        chunk_overlap=100,
        entity_confidence_threshold=0.8
    )

    builder = KnowledgeGraphBuilder(config=config)

    texts = [
        "Microsoft Corporation is a multinational technology company founded by Bill Gates and Paul Allen.",
        "Bill Gates served as CEO of Microsoft for many years. He is now focused on philanthropy.",
        "Paul Allen was co-founder of Microsoft alongside Bill Gates. He was also an investor and philanthropist."
    ]

    # First run - will process and cache results
    print("First run (processing and caching)...")
    _kg1 = builder.build_from_text(texts, graph_name="Microsoft Knowledge Graph")

    # Second run - will use cached results
    print("Second run (using cache)...")
    _kg2 = builder.build_from_text(texts, graph_name="Microsoft Knowledge Graph")

    # Show cache info
    cache_info = builder.get_cache_info()
    print(f"Cache info: {cache_info['backend']['total_files']} cached files")


def interactive_example():
    """Example with user interaction simulation."""
    print("\n=== Interactive Builder Example ===")

    builder = KnowledgeGraphBuilder(cache_dir="./interactive_cache")

    # Step 1: Process text and chunk
    texts = ["Google LLC is a technology company founded by Larry Page and Sergey Brin in 1998."]
    chunks = builder.chunk_texts(texts)
    print(f"Initial chunks: {len(chunks)}")

    # Simulate user editing chunks (in real app, this would be through UI)
    # For demo, just modify the first chunk
    if chunks:
        chunks[0].content = chunks[0].content + " Google is based in Mountain View, California."
        builder.update_chunks(chunks)
        print("Updated chunks with additional information")

    # Step 2: Extract entities
    entities = builder.extract_entities_from_chunks(chunks)
    print(f"Extracted entities: {len(entities)}")

    # Continue building
    relations = builder.extract_relations_from_chunks(chunks, entities)
    clusters = builder.form_clusters(entities, relations)

    print(f"Final results: {len(entities)} entities, {len(relations)} relations, {len(clusters)} clusters")


def resumable_example():
    """Example showing resumable builds."""
    print("\n=== Resumable Build Example ===")

    builder = KnowledgeGraphBuilder(cache_dir="./resumable_cache")

    texts = [
        "Amazon.com, Inc. is an American multinational technology company founded by Jeff Bezos.",
        "Jeff Bezos founded Amazon in 1994 and served as CEO until 2021.",
        "Amazon started as an online bookstore but expanded into cloud computing and other services."
    ]

    try:
        # Simulate interrupted build
        print("Starting build...")

        # Process first few steps
        _processed_texts = builder.process_documents([])  # Skip for text input
        chunks = builder.chunk_texts(texts)
        _entities = builder.extract_entities_from_chunks(chunks)

        print("Build interrupted after entity extraction...")

        # Simulate resuming from relation extraction step
        print("Resuming build from relation extraction...")
        kg = builder.build_from_text(
            texts,
            graph_name="Amazon Knowledge Graph",
            from_step="relation_extraction"
        )

        stats = kg.get_graph_statistics()
        print(f"Resumed and completed: {stats['total_entities']} entities, {stats['total_relations']} relations")

    except Exception as e:
        print(f"Build error: {e}")

        # Show build status
        status = builder.get_build_status()
        print(f"Build status: {status}")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    cached_example()
    interactive_example()
    resumable_example()

    print("\n=== All Examples Completed ===")
