"""
Tests for IndexManager - comprehensive coverage of index operations.

This test module covers all IndexManager functionality including:
- Index creation and deletion
- Entity type indexing operations
- Relation-entity index operations
- Entity-relation index operations
- Entity-cluster index operations
- Entity-text chunk index operations
- Bulk operations
- Index statistics and performance monitoring
- Thread safety
"""

import threading
import time
import unittest

from agraph.base.core.types import EntityType
from agraph.base.graphs.optimized import OptimizedKnowledgeGraph
from agraph.base.infrastructure.indexes import IndexManager, IndexType
from agraph.base.models.clusters import Cluster
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.base.models.text import TextChunk


class TestIndexManager(unittest.TestCase):
    """Test cases for IndexManager functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.index_manager = IndexManager()

    def test_initialization(self):
        """Test IndexManager initialization."""
        self.assertIsInstance(self.index_manager._entity_type_index, dict)
        self.assertIsInstance(self.index_manager._relation_entity_index, dict)
        self.assertIsInstance(self.index_manager._entity_relations_index, dict)
        self.assertIsInstance(self.index_manager._entity_clusters_index, dict)
        self.assertIsInstance(self.index_manager._cluster_entities_index, dict)
        self.assertIsInstance(self.index_manager._entity_text_chunks_index, dict)
        self.assertIsInstance(self.index_manager._text_chunk_entities_index, dict)

        stats = self.index_manager.get_statistics()
        self.assertEqual(stats["total_indexes"], 0)
        self.assertEqual(stats["index_hits"], 0)
        self.assertEqual(stats["index_misses"], 0)

    def test_entity_type_index_operations(self):
        """Test entity type index add/remove/get operations."""
        entity_id = "entity_1"
        entity_type = EntityType.PERSON

        # Test adding entity to type index
        self.index_manager.add_entity_to_type_index(entity_id, entity_type)
        entities = self.index_manager.get_entities_by_type(entity_type)
        self.assertIn(entity_id, entities)

        # Test statistics update
        stats = self.index_manager.get_statistics()
        self.assertEqual(stats["total_indexes"], 1)
        self.assertEqual(stats["index_hits"], 1)

        # Test removing entity from type index
        self.index_manager.remove_entity_from_type_index(entity_id, entity_type)
        entities = self.index_manager.get_entities_by_type(entity_type)
        self.assertNotIn(entity_id, entities)

        # Test empty type index cleanup
        stats = self.index_manager.get_statistics()
        self.assertEqual(stats["entity_types_count"], 0)

    def test_entity_type_index_multiple_entities(self):
        """Test entity type index with multiple entities of same type."""
        entity_ids = ["entity_1", "entity_2", "entity_3"]
        entity_type = EntityType.ORGANIZATION

        # Add multiple entities
        for entity_id in entity_ids:
            self.index_manager.add_entity_to_type_index(entity_id, entity_type)

        # Get all entities of type
        entities = self.index_manager.get_entities_by_type(entity_type)
        self.assertEqual(len(entities), 3)
        for entity_id in entity_ids:
            self.assertIn(entity_id, entities)

        # Remove one entity
        self.index_manager.remove_entity_from_type_index(entity_ids[0], entity_type)
        entities = self.index_manager.get_entities_by_type(entity_type)
        self.assertEqual(len(entities), 2)
        self.assertNotIn(entity_ids[0], entities)

    def test_relation_entity_index_operations(self):
        """Test relation-entity index operations."""
        relation_id = "relation_1"
        head_entity_id = "entity_1"
        tail_entity_id = "entity_2"

        # Test adding relation to index
        self.index_manager.add_relation_to_index(relation_id, head_entity_id, tail_entity_id)

        # Test getting relation entities
        entities = self.index_manager.get_relation_entities(relation_id)
        self.assertEqual(entities, (head_entity_id, tail_entity_id))

        # Test getting entity relations
        head_relations = self.index_manager.get_entity_relations(head_entity_id)
        tail_relations = self.index_manager.get_entity_relations(tail_entity_id)
        self.assertIn(relation_id, head_relations)
        self.assertIn(relation_id, tail_relations)

        # Test removing relation from index
        removed_entities = self.index_manager.remove_relation_from_index(relation_id)
        self.assertEqual(removed_entities, (head_entity_id, tail_entity_id))

        # Verify removal
        entities = self.index_manager.get_relation_entities(relation_id)
        self.assertIsNone(entities)

        head_relations = self.index_manager.get_entity_relations(head_entity_id)
        tail_relations = self.index_manager.get_entity_relations(tail_entity_id)
        self.assertNotIn(relation_id, head_relations)
        self.assertNotIn(relation_id, tail_relations)

    def test_entity_cluster_index_operations(self):
        """Test entity-cluster index operations."""
        entity_id = "entity_1"
        cluster_id = "cluster_1"

        # Test adding entity to cluster index
        self.index_manager.add_entity_to_cluster_index(entity_id, cluster_id)

        # Test getting entity clusters
        clusters = self.index_manager.get_entity_clusters(entity_id)
        self.assertIn(cluster_id, clusters)

        # Test getting cluster entities
        entities = self.index_manager.get_cluster_entities(cluster_id)
        self.assertIn(entity_id, entities)

        # Test removing entity from cluster index
        self.index_manager.remove_entity_from_cluster_index(entity_id, cluster_id)

        # Verify removal
        clusters = self.index_manager.get_entity_clusters(entity_id)
        entities = self.index_manager.get_cluster_entities(cluster_id)
        self.assertNotIn(cluster_id, clusters)
        self.assertNotIn(entity_id, entities)

    def test_entity_text_chunk_index_operations(self):
        """Test entity-text chunk index operations."""
        entity_id = "entity_1"
        text_chunk_id = "chunk_1"

        # Test adding entity to text chunk index
        self.index_manager.add_entity_to_text_chunk_index(entity_id, text_chunk_id)

        # Test getting entity text chunks
        chunks = self.index_manager.get_entity_text_chunks(entity_id)
        self.assertIn(text_chunk_id, chunks)

        # Test getting text chunk entities
        entities = self.index_manager.get_text_chunk_entities(text_chunk_id)
        self.assertIn(entity_id, entities)

        # Test removing entity from text chunk index
        self.index_manager.remove_entity_from_text_chunk_index(entity_id, text_chunk_id)

        # Verify removal
        chunks = self.index_manager.get_entity_text_chunks(entity_id)
        entities = self.index_manager.get_text_chunk_entities(text_chunk_id)
        self.assertNotIn(text_chunk_id, chunks)
        self.assertNotIn(entity_id, entities)

    def test_bulk_entity_removal(self):
        """Test removing entity from all indexes."""
        entity_id = "entity_1"
        relation_id = "relation_1"
        cluster_id = "cluster_1"
        text_chunk_id = "chunk_1"
        other_entity_id = "entity_2"

        # Set up entity in all indexes
        self.index_manager.add_entity_to_type_index(entity_id, EntityType.PERSON)
        self.index_manager.add_relation_to_index(relation_id, entity_id, other_entity_id)
        self.index_manager.add_entity_to_cluster_index(entity_id, cluster_id)
        self.index_manager.add_entity_to_text_chunk_index(entity_id, text_chunk_id)

        # Remove entity from all indexes
        removed_data = self.index_manager.remove_entity_from_all_indexes(
            entity_id, EntityType.PERSON
        )

        # Verify removal
        self.assertIn(relation_id, removed_data["relations"])
        self.assertIn(cluster_id, removed_data["clusters"])
        self.assertIn(text_chunk_id, removed_data["text_chunks"])

        # Verify entity no longer in any indexes
        type_entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertNotIn(entity_id, type_entities)

        entity_relations = self.index_manager.get_entity_relations(entity_id)
        self.assertEqual(len(entity_relations), 0)

    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        # Initial state
        stats = self.index_manager.get_statistics()
        self.assertEqual(stats["index_hits"], 0)
        self.assertEqual(stats["index_misses"], 0)

        # Add entity and query
        entity_id = "entity_1"
        entity_type = EntityType.CONCEPT
        self.index_manager.add_entity_to_type_index(entity_id, entity_type)

        # Hit case
        entities = self.index_manager.get_entities_by_type(entity_type)
        self.assertIn(entity_id, entities)

        # Miss case
        entities = self.index_manager.get_entities_by_type(EntityType.LOCATION)
        self.assertEqual(len(entities), 0)

        # Check statistics
        stats = self.index_manager.get_statistics()
        self.assertGreater(stats["index_hits"], 0)
        self.assertGreater(stats["index_misses"], 0)
        self.assertGreater(stats["hit_ratio"], 0)

    def test_clear_all_indexes(self):
        """Test clearing all indexes."""
        # Set up some data
        self.index_manager.add_entity_to_type_index("entity_1", EntityType.PERSON)
        self.index_manager.add_relation_to_index("relation_1", "entity_1", "entity_2")

        # Verify data exists
        stats_before = self.index_manager.get_statistics()
        self.assertGreater(stats_before["total_indexes"], 0)

        # Clear all indexes
        self.index_manager.clear_all_indexes()

        # Verify everything is cleared
        stats_after = self.index_manager.get_statistics()
        self.assertEqual(stats_after["total_indexes"], 0)
        self.assertEqual(stats_after["entity_types_count"], 0)
        self.assertEqual(stats_after["relations_count"], 0)

    def test_rebuild_indexes(self):
        """Test rebuilding indexes from knowledge graph data."""
        # Create a knowledge graph with data
        kg = OptimizedKnowledgeGraph()

        # Add test entities
        entity1 = Entity(id="entity_1", name="Test Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Test Entity 2", entity_type=EntityType.ORGANIZATION)
        kg.entities["entity_1"] = entity1
        kg.entities["entity_2"] = entity2

        # Add test relation
        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="WORKS_FOR"
        )
        kg.relations["relation_1"] = relation

        # Add test cluster
        cluster = Cluster(id="cluster_1", entities={"entity_1"})
        kg.clusters["cluster_1"] = cluster

        # Add test text chunk
        text_chunk = TextChunk(id="chunk_1", content="Test content", entities={"entity_1"})
        kg.text_chunks["chunk_1"] = text_chunk

        # Clear current indexes and rebuild
        self.index_manager.clear_all_indexes()
        self.index_manager.rebuild_indexes(kg)

        # Verify indexes were rebuilt correctly
        person_entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertIn("entity_1", person_entities)

        org_entities = self.index_manager.get_entities_by_type(EntityType.ORGANIZATION)
        self.assertIn("entity_2", org_entities)

        relation_entities = self.index_manager.get_relation_entities("relation_1")
        self.assertEqual(relation_entities, ("entity_1", "entity_2"))

        entity_clusters = self.index_manager.get_entity_clusters("entity_1")
        self.assertIn("cluster_1", entity_clusters)

        entity_chunks = self.index_manager.get_entity_text_chunks("entity_1")
        self.assertIn("chunk_1", entity_chunks)

    def test_index_consistency(self):
        """Test consistency between different index views."""
        relation_id = "relation_1"
        head_entity_id = "entity_1"
        tail_entity_id = "entity_2"

        # Add relation to index
        self.index_manager.add_relation_to_index(relation_id, head_entity_id, tail_entity_id)

        # Verify consistency between relation->entities and entity->relations indexes
        relation_entities = self.index_manager.get_relation_entities(relation_id)
        head_relations = self.index_manager.get_entity_relations(head_entity_id)
        tail_relations = self.index_manager.get_entity_relations(tail_entity_id)

        self.assertEqual(relation_entities, (head_entity_id, tail_entity_id))
        self.assertIn(relation_id, head_relations)
        self.assertIn(relation_id, tail_relations)

        # Remove and verify consistency is maintained
        removed_entities = self.index_manager.remove_relation_from_index(relation_id)
        self.assertEqual(removed_entities, (head_entity_id, tail_entity_id))

        # Verify removal from both indexes
        relation_entities = self.index_manager.get_relation_entities(relation_id)
        head_relations = self.index_manager.get_entity_relations(head_entity_id)
        tail_relations = self.index_manager.get_entity_relations(tail_entity_id)

        self.assertIsNone(relation_entities)
        self.assertEqual(len(head_relations), 0)
        self.assertEqual(len(tail_relations), 0)

    def test_thread_safety(self):
        """Test thread safety of index operations."""
        entity_ids = [f"entity_{i}" for i in range(100)]
        results = []
        errors = []

        def add_entities():
            try:
                for entity_id in entity_ids:
                    self.index_manager.add_entity_to_type_index(entity_id, EntityType.PERSON)
                results.append("add_success")
            except Exception as e:
                errors.append(f"add_error: {e}")

        def query_entities():
            try:
                for _ in range(50):
                    entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
                    results.append(f"query_result: {len(entities)}")
                    time.sleep(0.001)  # Small delay to interleave operations
            except Exception as e:
                errors.append(f"query_error: {e}")

        # Run concurrent operations
        threads = [
            threading.Thread(target=add_entities),
            threading.Thread(target=query_entities),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")

        # Verify final state
        final_entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertEqual(len(final_entities), 100)

    def test_nonexistent_keys(self):
        """Test behavior with nonexistent keys."""
        # Test getting nonexistent entity type
        entities = self.index_manager.get_entities_by_type(EntityType.LOCATION)
        self.assertEqual(len(entities), 0)

        # Test getting nonexistent relation
        relation_entities = self.index_manager.get_relation_entities("nonexistent_relation")
        self.assertIsNone(relation_entities)

        # Test getting nonexistent entity relations
        entity_relations = self.index_manager.get_entity_relations("nonexistent_entity")
        self.assertEqual(len(entity_relations), 0)

        # Test removing nonexistent relation
        removed_entities = self.index_manager.remove_relation_from_index("nonexistent_relation")
        self.assertIsNone(removed_entities)

        # Verify miss statistics
        stats = self.index_manager.get_statistics()
        self.assertGreater(stats["index_misses"], 0)

    def test_index_statistics_accuracy(self):
        """Test accuracy of index statistics."""
        # Add some test data
        entity_ids = ["entity_1", "entity_2", "entity_3"]
        for entity_id in entity_ids:
            self.index_manager.add_entity_to_type_index(entity_id, EntityType.PERSON)

        self.index_manager.add_relation_to_index("relation_1", "entity_1", "entity_2")
        self.index_manager.add_entity_to_cluster_index("entity_1", "cluster_1")
        self.index_manager.add_entity_to_text_chunk_index("entity_1", "chunk_1")

        # Get statistics
        stats = self.index_manager.get_statistics()

        # Verify counts
        self.assertEqual(stats["entity_types_count"], 1)  # One type (PERSON)
        self.assertEqual(stats["relations_count"], 1)
        self.assertEqual(stats["entity_relations_count"], 2)  # Both entities have relations
        self.assertEqual(stats["entity_clusters_count"], 1)
        self.assertEqual(stats["cluster_entities_count"], 1)
        self.assertEqual(stats["entity_text_chunks_count"], 1)
        self.assertEqual(stats["text_chunk_entities_count"], 1)

    def test_complex_removal_scenario(self):
        """Test complex entity removal affecting multiple indexes."""
        entity_id = "central_entity"

        # Set up entity in multiple indexes
        self.index_manager.add_entity_to_type_index(entity_id, EntityType.PERSON)

        # Add multiple relations
        relation_ids = ["rel_1", "rel_2", "rel_3"]
        other_entities = ["entity_2", "entity_3", "entity_4"]
        for i, relation_id in enumerate(relation_ids):
            self.index_manager.add_relation_to_index(relation_id, entity_id, other_entities[i])

        # Add to multiple clusters
        cluster_ids = ["cluster_1", "cluster_2"]
        for cluster_id in cluster_ids:
            self.index_manager.add_entity_to_cluster_index(entity_id, cluster_id)

        # Add to text chunks
        chunk_ids = ["chunk_1", "chunk_2"]
        for chunk_id in chunk_ids:
            self.index_manager.add_entity_to_text_chunk_index(entity_id, chunk_id)

        # Remove entity from all indexes
        removed_data = self.index_manager.remove_entity_from_all_indexes(
            entity_id, EntityType.PERSON
        )

        # Verify all relationships were removed
        self.assertEqual(len(removed_data["relations"]), 3)
        self.assertEqual(len(removed_data["clusters"]), 2)
        self.assertEqual(len(removed_data["text_chunks"]), 2)

        # Verify entity is completely removed from all indexes
        type_entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertNotIn(entity_id, type_entities)

        for relation_id in relation_ids:
            relation_entities = self.index_manager.get_relation_entities(relation_id)
            self.assertIsNone(relation_entities)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test empty string keys
        self.index_manager.add_entity_to_type_index("", EntityType.PERSON)
        entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertIn("", entities)

        # Test removing non-existent entity from type index
        self.index_manager.remove_entity_from_type_index("nonexistent", EntityType.LOCATION)
        # Should not raise exception

        # Test multiple additions of same entity-cluster pair
        self.index_manager.add_entity_to_cluster_index("entity_1", "cluster_1")
        self.index_manager.add_entity_to_cluster_index("entity_1", "cluster_1")

        clusters = self.index_manager.get_entity_clusters("entity_1")
        entities = self.index_manager.get_cluster_entities("cluster_1")
        self.assertEqual(len(clusters), 1)  # Should only contain one instance
        self.assertEqual(len(entities), 1)

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation in statistics."""
        # Add some data
        self.index_manager.add_entity_to_type_index("entity_1", EntityType.PERSON)

        # Generate hits and misses
        self.index_manager.get_entities_by_type(EntityType.PERSON)  # Hit
        self.index_manager.get_entities_by_type(EntityType.PERSON)  # Hit
        self.index_manager.get_entities_by_type(EntityType.LOCATION)  # Miss

        stats = self.index_manager.get_statistics()
        expected_ratio = 2 / 3  # 2 hits out of 3 queries
        self.assertAlmostEqual(stats["hit_ratio"], expected_ratio, places=2)

    def test_index_type_enum(self):
        """Test IndexType enum values."""
        self.assertEqual(IndexType.ENTITY_TYPE.value, "entity_type")
        self.assertEqual(IndexType.RELATION_ENTITY.value, "relation_entity")
        self.assertEqual(IndexType.ENTITY_RELATIONS.value, "entity_relations")
        self.assertEqual(IndexType.ENTITY_CLUSTERS.value, "entity_clusters")
        self.assertEqual(IndexType.ENTITY_TEXT_CHUNKS.value, "entity_text_chunks")
        self.assertEqual(IndexType.CLUSTER_ENTITIES.value, "cluster_entities")


class TestIndexManagerIntegration(unittest.TestCase):
    """Integration tests for IndexManager with knowledge graph components."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()
        self.index_manager = self.kg.index_manager

    def test_integration_with_optimized_knowledge_graph(self):
        """Test IndexManager integration with OptimizedKnowledgeGraph."""
        # Add entities through knowledge graph
        entity1 = Entity(id="entity_1", name="Test Person", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Test Org", entity_type=EntityType.ORGANIZATION)

        self.kg.add_entity(entity1)
        self.kg.add_entity(entity2)

        # Verify indexes are updated
        person_entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        org_entities = self.index_manager.get_entities_by_type(EntityType.ORGANIZATION)

        self.assertIn("entity_1", person_entities)
        self.assertIn("entity_2", org_entities)

        # Add relation through knowledge graph
        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="WORKS_FOR"
        )
        self.kg.add_relation(relation)

        # Verify relation indexes are updated
        relation_entities = self.index_manager.get_relation_entities("relation_1")
        self.assertEqual(relation_entities, ("entity_1", "entity_2"))

    def test_performance_with_large_dataset(self):
        """Test performance characteristics with larger dataset."""
        # Create large number of entities
        num_entities = 1000
        start_time = time.time()

        for i in range(num_entities):
            entity_id = f"entity_{i}"
            entity_type = EntityType.PERSON if i % 2 == 0 else EntityType.ORGANIZATION
            self.index_manager.add_entity_to_type_index(entity_id, entity_type)

        index_creation_time = time.time() - start_time

        # Test query performance
        start_time = time.time()
        person_entities = self.index_manager.get_entities_by_type(EntityType.PERSON)
        query_time = time.time() - start_time

        # Verify results
        self.assertEqual(len(person_entities), 500)  # Half should be PERSON type

        # Performance should be fast (sub-millisecond for indexed lookup)
        self.assertLess(query_time, 0.01)  # Should be very fast

        # Index creation should be reasonable
        self.assertLess(index_creation_time, 1.0)  # Should complete within 1 second


if __name__ == "__main__":
    unittest.main()
