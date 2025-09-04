"""
Tests for core infrastructure components - IndexManager, CacheManager, and OptimizedKnowledgeGraph.

This is a simplified test suite focusing on the most critical functionality.
"""

import time
import unittest

from agraph.base.graphs.optimized import OptimizedKnowledgeGraph
from agraph.base.infrastructure.cache import CacheManager, CacheStrategy
from agraph.base.infrastructure.indexes import IndexManager
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation


class TestIndexManagerCore(unittest.TestCase):
    """Core IndexManager functionality tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.index_manager = IndexManager()

    def test_entity_type_indexing(self):
        """Test entity type indexing operations."""
        # Add entity to index
        self.index_manager.add_entity_to_type_index("entity_1", "person")

        # Retrieve entities by type
        entities = self.index_manager.get_entities_by_type("person")
        self.assertIn("entity_1", entities)

        # Remove entity from index
        self.index_manager.remove_entity_from_type_index("entity_1", "person")
        entities = self.index_manager.get_entities_by_type("person")
        self.assertNotIn("entity_1", entities)

    def test_relation_indexing(self):
        """Test relation indexing operations."""
        # Add relation to index
        self.index_manager.add_relation_to_index("rel_1", "entity_1", "entity_2")

        # Test relation -> entities mapping
        entities = self.index_manager.get_relation_entities("rel_1")
        self.assertEqual(entities, ("entity_1", "entity_2"))

        # Test entity -> relations mapping
        relations = self.index_manager.get_entity_relations("entity_1")
        self.assertIn("rel_1", relations)

    def test_bulk_removal(self):
        """Test bulk entity removal from indexes."""
        entity_id = "entity_1"

        # Set up entity in multiple indexes
        self.index_manager.add_entity_to_type_index(entity_id, "person")
        self.index_manager.add_relation_to_index("rel_1", entity_id, "entity_2")

        # Remove from all indexes
        removed_data = self.index_manager.remove_entity_from_all_indexes(entity_id, "person")

        # Verify removal
        type_entities = self.index_manager.get_entities_by_type("person")
        self.assertNotIn(entity_id, type_entities)


class TestCacheManagerCore(unittest.TestCase):
    """Core CacheManager functionality tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager(max_size=100)

    def test_basic_operations(self):
        """Test basic cache put/get operations."""
        # Cache miss
        result = self.cache_manager.get("test_key")
        self.assertIsNone(result)

        # Cache put and hit
        self.cache_manager.put("test_key", "test_value")
        result = self.cache_manager.get("test_key")
        self.assertEqual(result, "test_value")

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Put with short TTL
        self.cache_manager.put("test_key", "test_value", ttl=0.1)

        # Should be available immediately
        result = self.cache_manager.get("test_key")
        self.assertEqual(result, "test_value")

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        result = self.cache_manager.get("test_key")
        self.assertIsNone(result)

    def test_invalidation(self):
        """Test cache invalidation."""
        # Put with tags
        self.cache_manager.put("key1", "value1", tags={"tag1"})
        self.cache_manager.put("key2", "value2", tags={"tag2"})

        # Invalidate by tag
        count = self.cache_manager.invalidate_by_tags({"tag1"})
        self.assertEqual(count, 1)

        # Verify selective invalidation
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNotNone(self.cache_manager.get("key2"))


class TestOptimizedKnowledgeGraphCore(unittest.TestCase):
    """Core OptimizedKnowledgeGraph functionality tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

    def test_entity_operations(self):
        """Test entity CRUD operations."""
        # Add entity
        entity = Entity(id="entity_1", name="Test Entity", entity_type="person")
        self.kg.add_entity(entity)

        # Verify addition
        self.assertIn("entity_1", self.kg.entities)

        # Test retrieval by type
        persons = self.kg.get_entities_by_type("person")
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].id, "entity_1")

        # Test removal
        result = self.kg.remove_entity("entity_1")
        self.assertTrue(result)
        self.assertNotIn("entity_1", self.kg.entities)

    def test_relation_operations(self):
        """Test relation CRUD operations."""
        # Add entities first
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type="person")
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type="organization")
        self.kg.add_entity(entity1)
        self.kg.add_entity(entity2)

        # Add relation
        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="works_for"
        )
        self.kg.add_relation(relation)

        # Verify addition
        self.assertIn("relation_1", self.kg.relations)

        # Test retrieval
        retrieved = self.kg.get_relation("relation_1")
        self.assertEqual(retrieved, relation)

        # Test entity relations
        entity_relations = self.kg.get_entity_relations("entity_1")
        self.assertEqual(len(entity_relations), 1)
        self.assertEqual(entity_relations[0].id, "relation_1")

    def test_graph_statistics(self):
        """Test graph statistics calculation."""
        # Add test data
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type="person")
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type="organization")
        self.kg.add_entity(entity1)
        self.kg.add_entity(entity2)

        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="works_for"
        )
        self.kg.add_relation(relation)

        # Get statistics
        stats = self.kg.get_graph_statistics()

        # Verify statistics
        self.assertEqual(stats["total_entities"], 2)
        self.assertEqual(stats["total_relations"], 1)
        self.assertEqual(stats["entity_types"]["person"], 1)
        self.assertEqual(stats["entity_types"]["organization"], 1)

    def test_performance_optimization(self):
        """Test performance optimization features."""
        # Add some data
        entity = Entity(id="entity_1", name="Test Entity", entity_type="person")
        self.kg.add_entity(entity)

        # Get performance metrics
        metrics = self.kg.get_performance_metrics()
        self.assertIn("graph_metrics", metrics)
        self.assertIn("index_statistics", metrics)
        self.assertIn("cache_statistics", metrics)

        # Test optimization
        optimization_summary = self.kg.optimize_performance()
        self.assertIn("cache_cleanup", optimization_summary)

    def test_serialization(self):
        """Test graph serialization and deserialization."""
        # Add test data
        entity = Entity(id="entity_1", name="Test Entity", entity_type="person")
        self.kg.add_entity(entity)

        # Serialize
        data = self.kg.to_dict()
        self.assertIn("entities", data)
        self.assertEqual(len(data["entities"]), 1)

        # Deserialize
        new_kg = OptimizedKnowledgeGraph.from_dict(data)
        self.assertEqual(len(new_kg.entities), 1)
        self.assertIn("entity_1", new_kg.entities)

        # Verify indexes were rebuilt
        persons = new_kg.get_entities_by_type("person")
        self.assertEqual(len(persons), 1)


if __name__ == "__main__":
    unittest.main()
