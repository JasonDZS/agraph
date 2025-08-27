"""
Comprehensive demo of the new unified manager architecture.

This example demonstrates the key features of the refactored AGraph architecture:
- Result-based error handling
- Manager layer decoupling with DAO
- Batch operations with transactions
- Factory pattern for manager creation
- Performance metrics and monitoring
"""

import time
from typing import Dict, Any

# Import the new unified architecture components
from agraph.base.manager_factory import create_managers, get_manager_factory
from agraph.base.dao import MemoryDataAccessLayer
from agraph.base.result import Result, ResultUtils
from agraph.base.batch import BatchOperation, BatchOperationType, create_entity_batch_operation, create_relation_batch_operation
from agraph.base.entities import Entity
from agraph.base.relations import Relation
from agraph.base.clusters import Cluster
from agraph.base.text import TextChunk
from agraph.base.types import EntityType, RelationType, ClusterType


def demonstrate_result_based_apis():
    """Demonstrate the new Result-based API pattern."""
    print("=== Result-Based APIs Demo ===")

    # Create managers using the factory
    managers = create_managers("default")
    entity_manager = managers["entity_manager"]

    # Create a test entity
    entity = Entity(
        id="demo_person_1",
        name="Alice Johnson",
        entity_type=EntityType.PERSON,
        confidence=0.95,
        aliases=["Alice", "A. Johnson"]
    )

    # Demonstrate successful operation
    print("1. Adding entity...")
    add_result = entity_manager.add(entity)

    if add_result.is_ok():
        print(f"✅ Success: Added entity {add_result.data.name}")
        print(f"   Execution time: {add_result.metadata['execution_time']:.4f}s")
    else:
        print(f"❌ Error: {add_result.error_message}")

    # Demonstrate error handling
    print("\n2. Attempting to add duplicate...")
    duplicate_result = entity_manager.add(entity)

    if duplicate_result.is_error():
        print(f"❌ Expected error: {duplicate_result.error_message}")
        print(f"   Error code: {duplicate_result.error_code}")

    # Demonstrate functional programming with Results
    print("\n3. Using functional Result operations...")

    # Chain operations using map and filter
    validation_result = (entity_manager.get(entity.id)
                        .map(lambda e: e.name if e else "Unknown")
                        .filter(lambda name: len(name) > 5))

    if validation_result.is_ok():
        print(f"✅ Filtered result: {validation_result.data}")

    print()


def demonstrate_manager_decoupling():
    """Demonstrate the decoupled manager architecture."""
    print("=== Manager Decoupling Demo ===")

    # Create a custom DAO
    custom_dao = MemoryDataAccessLayer()

    # Create managers with different factories
    default_managers = create_managers("default", dao=custom_dao)
    optimized_managers = create_managers("optimized", dao=custom_dao)

    print("1. Using default factory...")
    entity_manager = default_managers["entity_manager"]

    entity = Entity(
        id="demo_org_1",
        name="Tech Corp",
        entity_type=EntityType.ORGANIZATION,
        confidence=0.9
    )

    add_result = entity_manager.add(entity)
    if add_result.is_ok():
        print(f"✅ Added via default manager: {add_result.data.name}")

    # Switch to optimized manager - should see same data (shared DAO)
    print("\n2. Switching to optimized factory...")
    optimized_entity_manager = optimized_managers["entity_manager"]

    get_result = optimized_entity_manager.get(entity.id)
    if get_result.is_ok() and get_result.data:
        print(f"✅ Retrieved via optimized manager: {get_result.data.name}")
        print("   Demonstrates shared DAO across different manager implementations")

    print()


def demonstrate_batch_operations():
    """Demonstrate transactional batch operations."""
    print("=== Batch Operations Demo ===")

    managers = create_managers("default")
    batch_manager = managers["batch_operation_manager"]
    entity_manager = managers["entity_manager"]

    # Create test data
    entities_data = [
        {
            "id": "batch_person_1",
            "name": "John Batch",
            "entity_type": "PERSON",
            "confidence": 0.9
        },
        {
            "id": "batch_person_2",
            "name": "Jane Batch",
            "entity_type": "PERSON",
            "confidence": 0.85
        },
        {
            "id": "batch_org_1",
            "name": "Batch Corp",
            "entity_type": "ORGANIZATION",
            "confidence": 0.8
        }
    ]

    print("1. Creating batch operations...")

    # Begin batch context
    batch_result = batch_manager.begin_batch()
    if not batch_result.is_ok():
        print(f"❌ Failed to create batch: {batch_result.error_message}")
        return

    batch_context = batch_result.data

    # Add operations to batch
    for entity_data in entities_data:
        operation = create_entity_batch_operation(
            BatchOperationType.ADD,
            entity_data
        )
        batch_context.add_operation(operation)

    print(f"   Added {len(entities_data)} operations to batch")

    # Commit batch
    print("\n2. Committing batch...")
    commit_result = batch_manager.commit_batch(batch_context)

    if commit_result.is_ok():
        summary = commit_result.data
        print(f"✅ Batch committed successfully:")
        print(f"   Total operations: {summary['total_operations']}")
        print(f"   Successful: {summary['successful_operations']}")
        print(f"   Failed: {summary['failed_operations']}")
        print(f"   Execution time: {summary['execution_time']:.4f}s")
    else:
        print(f"❌ Batch commit failed: {commit_result.error_message}")

    # Verify entities were created
    print("\n3. Verifying batch results...")
    for entity_data in entities_data:
        result = entity_manager.get(entity_data["id"])
        if result.is_ok() and result.data:
            print(f"   ✅ {result.data.name} exists")

    print()


def demonstrate_performance_monitoring():
    """Demonstrate performance metrics collection."""
    print("=== Performance Monitoring Demo ===")

    managers = create_managers("default")
    entity_manager = managers["entity_manager"]

    print("1. Performing operations to generate metrics...")

    # Perform various operations
    entities = []
    for i in range(100):
        entity = Entity(
            id=f"perf_entity_{i}",
            name=f"Performance Entity {i}",
            entity_type=EntityType.PERSON,
            confidence=0.8
        )
        entities.append(entity)
        entity_manager.add(entity)

    # Perform some searches
    for i in range(10):
        entity_manager.search(f"Entity {i}")

    # Get some statistics
    for i in range(5):
        entity_manager.list_by_type(EntityType.PERSON)

    print(f"   Completed {len(entities)} adds, 10 searches, 5 type queries")

    # Get performance statistics
    print("\n2. Collecting performance metrics...")
    stats_result = entity_manager.get_statistics()

    if stats_result.is_ok():
        stats = stats_result.data
        metrics = stats["manager_metrics"]

        print(f"✅ Performance Metrics:")
        print(f"   Total operations: {metrics['operations_count']}")
        print(f"   Error count: {metrics['errors_count']}")
        print(f"   Average response time: {metrics['average_response_time']:.4f}s")
        print(f"   Cache hits: {metrics['cache_hits']}")
        print(f"   Cache misses: {metrics['cache_misses']}")

        print(f"\n📊 Entity Statistics:")
        print(f"   Total entities: {stats['total_entities']}")
        print(f"   Type distribution: {stats['type_distribution']}")

    print()


def demonstrate_comprehensive_workflow():
    """Demonstrate a comprehensive workflow using all features."""
    print("=== Comprehensive Workflow Demo ===")

    # Create managers
    managers = create_managers("optimized")
    entity_manager = managers["entity_manager"]
    relation_manager = managers["relation_manager"]
    cluster_manager = managers["cluster_manager"]
    text_chunk_manager = managers["text_chunk_manager"]
    batch_manager = managers["batch_operation_manager"]

    print("1. Setting up knowledge graph data...")

    # Create entities
    alice = Entity(
        id="alice_smith",
        name="Alice Smith",
        entity_type=EntityType.PERSON,
        confidence=0.95,
        properties={"role": "CEO", "company": "Tech Innovations"}
    )

    tech_innovations = Entity(
        id="tech_innovations",
        name="Tech Innovations Inc.",
        entity_type=EntityType.ORGANIZATION,
        confidence=0.9,
        properties={"industry": "Technology", "founded": "2020"}
    )

    # Add entities
    entity_manager.add(alice)
    entity_manager.add(tech_innovations)

    # Create relation
    works_at_relation = Relation(
        id="alice_works_at_tech",
        head_entity=alice,
        tail_entity=tech_innovations,
        relation_type=RelationType.BELONGS_TO,
        confidence=0.9,
        properties={"position": "CEO", "since": "2020"}
    )

    relation_manager.add(works_at_relation)

    # Create cluster
    tech_cluster = Cluster(
        id="tech_industry_cluster",
        cluster_type=ClusterType.TOPIC,
        properties={"name": "Technology Industry", "description": "Tech companies and personnel"}
    )

    cluster_manager.add(tech_cluster)
    cluster_manager.add_entity_to_cluster(tech_cluster.id, alice.id)
    cluster_manager.add_entity_to_cluster(tech_cluster.id, tech_innovations.id)

    # Create text chunk
    news_chunk = TextChunk(
        id="tech_news_1",
        text="Alice Smith, CEO of Tech Innovations Inc., announced the company's expansion into AI research.",
        source="tech_news.html",
        properties={"date": "2024-01-15", "category": "business"}
    )

    text_chunk_manager.add(news_chunk)

    print("✅ Created knowledge graph with entities, relations, clusters, and text chunks")

    # Demonstrate complex queries
    print("\n2. Performing complex queries...")

    # Find related entities
    related_result = entity_manager.get_related_entities(alice.id)
    if related_result.is_ok():
        print(f"   Related to Alice: {[e.name for e in related_result.data]}")

    # Find path between entities
    path_result = relation_manager.find_path(alice.id, tech_innovations.id)
    if path_result.is_ok():
        print(f"   Path length: {len(path_result.data)} relations")

    # Get cluster entities
    cluster_entities_result = cluster_manager.get_cluster_entities(tech_cluster.id)
    if cluster_entities_result.is_ok():
        print(f"   Cluster entities: {[e.name for e in cluster_entities_result.data]}")

    # Search text chunks
    chunk_search_result = text_chunk_manager.search("Alice Smith")
    if chunk_search_result.is_ok():
        print(f"   Found {len(chunk_search_result.data)} text chunks mentioning Alice")

    print("\n3. Using Result chaining for complex operations...")

    # Chain multiple operations
    final_result = (entity_manager.get(alice.id)
                   .flat_map(lambda e: relation_manager.get_entity_relations(e.id) if e else Result.ok([]))
                   .map(lambda relations: len(relations))
                   .filter(lambda count: count > 0))

    if final_result.is_ok():
        print(f"✅ Alice has {final_result.data} relations")

    print()


def main():
    """Run all demonstrations."""
    print("🚀 AGraph Unified Architecture Demo")
    print("=" * 50)

    start_time = time.time()

    try:
        demonstrate_result_based_apis()
        demonstrate_manager_decoupling()
        demonstrate_batch_operations()
        demonstrate_performance_monitoring()
        demonstrate_comprehensive_workflow()

        end_time = time.time()
        print(f"✅ All demonstrations completed successfully in {end_time - start_time:.2f}s")

    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
