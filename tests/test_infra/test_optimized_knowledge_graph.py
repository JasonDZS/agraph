"""
Tests for OptimizedKnowledgeGraph - comprehensive coverage of optimized graph functionality.

This test module covers all OptimizedKnowledgeGraph functionality including:
- Graph initialization and configuration
- Entity management operations (CRUD)
- Relation management operations (CRUD)
- Cluster management operations
- Text chunk management operations
- Graph statistics calculation
- Connected components analysis
- Performance optimization functionality
- Cache integration testing
- Index integration testing
- Data serialization/deserialization
- Graph merging functionality
- Integrity validation
"""

import time
import unittest
from datetime import datetime

from agraph.base.core.types import ClusterType, EntityType, RelationType
from agraph.base.graphs.optimized import OptimizedKnowledgeGraph
from agraph.base.infrastructure.cache import CacheStrategy
from agraph.base.models.clusters import Cluster
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.base.models.text import TextChunk


class TestOptimizedKnowledgeGraphInitialization(unittest.TestCase):
    """Test OptimizedKnowledgeGraph initialization and configuration."""

    def test_default_initialization(self):
        """Test default initialization of OptimizedKnowledgeGraph."""
        kg = OptimizedKnowledgeGraph()

        # Verify basic properties
        self.assertIsNotNone(kg.id)
        self.assertEqual(kg.name, "")
        self.assertEqual(kg.description, "")
        self.assertEqual(len(kg.entities), 0)
        self.assertEqual(len(kg.relations), 0)
        self.assertEqual(len(kg.clusters), 0)
        self.assertEqual(len(kg.text_chunks), 0)

        # Verify optimization components
        self.assertIsNotNone(kg.index_manager)
        self.assertIsNotNone(kg.cache_manager)
        self.assertIsNotNone(kg._entity_manager)
        self.assertIsNotNone(kg._relation_manager)

        # Verify timestamps
        self.assertIsInstance(kg.created_at, datetime)
        self.assertIsInstance(kg.updated_at, datetime)

    def test_custom_initialization(self):
        """Test custom initialization with parameters."""
        custom_name = "Test Graph"
        custom_description = "A test knowledge graph"
        custom_metadata = {"test": True, "version": "1.0"}

        kg = OptimizedKnowledgeGraph(
            name=custom_name, description=custom_description, metadata=custom_metadata
        )

        self.assertEqual(kg.name, custom_name)
        self.assertEqual(kg.description, custom_description)
        self.assertEqual(kg.metadata, custom_metadata)

    def test_initialization_with_existing_data(self):
        """Test initialization with existing data rebuilds indexes."""
        # Create entities directly
        entity1 = Entity(id="entity_1", name="Test Entity 1", entity_type="person")
        entity2 = Entity(id="entity_2", name="Test Entity 2", entity_type="organization")
        entities = {"entity_1": entity1, "entity_2": entity2}

        # Create relation
        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="WORKS_FOR"
        )
        relations = {"relation_1": relation}

        # Initialize with existing data
        kg = OptimizedKnowledgeGraph(entities=entities, relations=relations)

        # Verify indexes were built
        person_entities = kg.index_manager.get_entities_by_type("person")
        self.assertIn("entity_1", person_entities)

        org_entities = kg.index_manager.get_entities_by_type("organization")
        self.assertIn("entity_2", org_entities)

        relation_entities = kg.index_manager.get_relation_entities("relation_1")
        self.assertEqual(relation_entities, ("entity_1", "entity_2"))


class TestOptimizedKnowledgeGraphEntityManagement(unittest.TestCase):
    """Test entity management operations in OptimizedKnowledgeGraph."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

    def test_add_entity(self):
        """Test adding entities to the knowledge graph."""
        entity = Entity(id="entity_1", name="Test Entity", entity_type="person")

        self.kg.add_entity(entity)

        # Verify entity was added
        self.assertIn("entity_1", self.kg.entities)
        self.assertEqual(self.kg.entities["entity_1"], entity)

        # Verify index was updated
        person_entities = self.kg.index_manager.get_entities_by_type("person")
        self.assertIn("entity_1", person_entities)

        # Verify performance metrics updated
        metrics = self.kg.get_performance_metrics()
        self.assertGreater(metrics["graph_metrics"]["total_operations"], 0)

    def test_remove_entity(self):
        """Test removing entities from the knowledge graph."""
        # Add entity first
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity)

        # Create related data
        other_entity = Entity(
            id="entity_2", name="Other Entity", entity_type=EntityType.ORGANIZATION
        )
        self.kg.add_entity(other_entity)

        relation = Relation(
            id="relation_1", head_entity=entity, tail_entity=other_entity, relation_type="WORKS_FOR"
        )
        self.kg.add_relation(relation)

        # Remove entity
        result = self.kg.remove_entity("entity_1")
        self.assertTrue(result)

        # Verify entity was removed
        self.assertNotIn("entity_1", self.kg.entities)

        # Verify cascading removal of relations
        self.assertEqual(len(self.kg.relations), 0)

        # Verify indexes were updated
        person_entities = self.kg.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertNotIn("entity_1", person_entities)

    def test_get_entity(self):
        """Test getting entities by ID."""
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity)

        # Test successful get
        retrieved = self.kg.get_entity("entity_1")
        self.assertEqual(retrieved, entity)

        # Test get non-existent entity
        retrieved = self.kg.get_entity("nonexistent")
        self.assertIsNone(retrieved)

    def test_get_entities_by_type(self):
        """Test getting entities by type using indexes."""
        # Add entities of different types
        person1 = Entity(id="person_1", name="Person 1", entity_type=EntityType.PERSON)
        person2 = Entity(id="person_2", name="Person 2", entity_type=EntityType.PERSON)
        org = Entity(id="org_1", name="Organization 1", entity_type=EntityType.ORGANIZATION)

        self.kg.add_entity(person1)
        self.kg.add_entity(person2)
        self.kg.add_entity(org)

        # Test getting persons
        persons = self.kg.get_entities_by_type(EntityType.PERSON)
        self.assertEqual(len(persons), 2)
        person_ids = {p.id for p in persons}
        self.assertEqual(person_ids, {"person_1", "person_2"})

        # Test getting organizations
        orgs = self.kg.get_entities_by_type(EntityType.ORGANIZATION)
        self.assertEqual(len(orgs), 1)
        self.assertEqual(orgs[0].id, "org_1")

    def test_search_entities(self):
        """Test entity search functionality."""
        # Add entities with searchable content
        entity1 = Entity(id="entity_1", name="Alice Smith", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Bob Jones", entity_type=EntityType.PERSON)
        entity3 = Entity(id="entity_3", name="ACME Corp", entity_type=EntityType.ORGANIZATION)

        self.kg.add_entity(entity1)
        self.kg.add_entity(entity2)
        self.kg.add_entity(entity3)

        # Test search by name
        results = self.kg.search_entities("Alice", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "entity_1")

        # Test partial match
        results = self.kg.search_entities("Corp", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "entity_3")

        # Test limit functionality
        results = self.kg.search_entities("", limit=2)  # Should match all
        self.assertLessEqual(len(results), 2)


class TestOptimizedKnowledgeGraphRelationManagement(unittest.TestCase):
    """Test relation management operations in OptimizedKnowledgeGraph."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

        # Add test entities
        self.entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        self.entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        self.kg.add_entity(self.entity1)
        self.kg.add_entity(self.entity2)

    def test_add_relation(self):
        """Test adding relations to the knowledge graph."""
        relation = Relation(
            id="relation_1",
            head_entity=self.entity1,
            tail_entity=self.entity2,
            relation_type="WORKS_FOR",
        )

        self.kg.add_relation(relation)

        # Verify relation was added
        self.assertIn("relation_1", self.kg.relations)
        self.assertEqual(self.kg.relations["relation_1"], relation)

        # Verify index was updated
        relation_entities = self.kg.index_manager.get_relation_entities("relation_1")
        self.assertEqual(relation_entities, ("entity_1", "entity_2"))

        entity_relations = self.kg.index_manager.get_entity_relations("entity_1")
        self.assertIn("relation_1", entity_relations)

    def test_remove_relation(self):
        """Test removing relations from the knowledge graph."""
        relation = Relation(
            id="relation_1",
            head_entity=self.entity1,
            tail_entity=self.entity2,
            relation_type="WORKS_FOR",
        )
        self.kg.add_relation(relation)

        # Remove relation
        result = self.kg.remove_relation("relation_1")
        self.assertTrue(result)

        # Verify removal
        self.assertNotIn("relation_1", self.kg.relations)

        # Verify index was updated
        relation_entities = self.kg.index_manager.get_relation_entities("relation_1")
        self.assertIsNone(relation_entities)

    def test_get_relation(self):
        """Test getting relations by ID."""
        relation = Relation(
            id="relation_1",
            head_entity=self.entity1,
            tail_entity=self.entity2,
            relation_type="WORKS_FOR",
        )
        self.kg.add_relation(relation)

        # Test successful get
        retrieved = self.kg.get_relation("relation_1")
        self.assertEqual(retrieved, relation)

        # Test get non-existent relation
        retrieved = self.kg.get_relation("nonexistent")
        self.assertIsNone(retrieved)

    def test_get_relations_by_type(self):
        """Test getting relations by type."""
        # Add relations of different types
        relation1 = Relation(
            id="relation_1",
            head_entity=self.entity1,
            tail_entity=self.entity2,
            relation_type="WORKS_FOR",
        )
        relation2 = Relation(
            id="relation_2",
            head_entity=self.entity1,
            tail_entity=self.entity2,
            relation_type="RELATED_TO",
        )

        self.kg.add_relation(relation1)
        self.kg.add_relation(relation2)

        # Test getting by type
        works_for_relations = self.kg.get_relations_by_type("WORKS_FOR")
        self.assertEqual(len(works_for_relations), 1)
        self.assertEqual(works_for_relations[0].id, "relation_1")

    def test_get_entity_relations(self):
        """Test getting entity relations with direction."""
        # Add multiple relations
        entity3 = Entity(id="entity_3", name="Entity 3", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity3)

        # entity1 -> entity2
        relation1 = Relation(
            id="relation_1",
            head_entity=self.entity1,
            tail_entity=self.entity2,
            relation_type="WORKS_FOR",
        )
        # entity3 -> entity1
        relation2 = Relation(
            id="relation_2",
            head_entity=entity3,
            tail_entity=self.entity1,
            relation_type="RELATED_TO",
        )

        self.kg.add_relation(relation1)
        self.kg.add_relation(relation2)

        # Test getting all relations for entity1
        entity1_relations = self.kg.get_entity_relations("entity_1", direction="both")
        self.assertEqual(len(entity1_relations), 2)

        relation_ids = {r.id for r in entity1_relations}
        self.assertEqual(relation_ids, {"relation_1", "relation_2"})


class TestOptimizedKnowledgeGraphClusterManagement(unittest.TestCase):
    """Test cluster management in OptimizedKnowledgeGraph."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

        # Add test entities
        self.entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        self.entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.PERSON)
        self.kg.add_entity(self.entity1)
        self.kg.add_entity(self.entity2)

    def test_add_cluster(self):
        """Test adding clusters to the knowledge graph."""
        cluster = Cluster(
            id="cluster_1", entities={"entity_1", "entity_2"}, cluster_type=ClusterType.COMMUNITY
        )

        self.kg.add_cluster(cluster)

        # Verify cluster was added
        self.assertIn("cluster_1", self.kg.clusters)
        self.assertEqual(self.kg.clusters["cluster_1"], cluster)

        # Verify indexes were updated
        entity1_clusters = self.kg.index_manager.get_entity_clusters("entity_1")
        entity2_clusters = self.kg.index_manager.get_entity_clusters("entity_2")
        self.assertIn("cluster_1", entity1_clusters)
        self.assertIn("cluster_1", entity2_clusters)

        cluster_entities = self.kg.index_manager.get_cluster_entities("cluster_1")
        self.assertEqual(cluster_entities, {"entity_1", "entity_2"})

    def test_remove_cluster(self):
        """Test removing clusters from the knowledge graph."""
        cluster = Cluster(
            id="cluster_1", entities={"entity_1", "entity_2"}, cluster_type=ClusterType.COMMUNITY
        )
        self.kg.add_cluster(cluster)

        # Remove cluster
        result = self.kg.remove_cluster("cluster_1")
        self.assertTrue(result)

        # Verify removal
        self.assertNotIn("cluster_1", self.kg.clusters)

        # Verify indexes were updated
        entity1_clusters = self.kg.index_manager.get_entity_clusters("entity_1")
        self.assertNotIn("cluster_1", entity1_clusters)

    def test_get_cluster(self):
        """Test getting clusters by ID."""
        cluster = Cluster(id="cluster_1", entities={"entity_1"})
        self.kg.add_cluster(cluster)

        # Test successful get
        retrieved = self.kg.get_cluster("cluster_1")
        self.assertEqual(retrieved, cluster)

        # Test get non-existent cluster
        retrieved = self.kg.get_cluster("nonexistent")
        self.assertIsNone(retrieved)

    def test_get_clusters_by_type(self):
        """Test getting clusters by type."""
        cluster1 = Cluster(id="cluster_1", cluster_type=ClusterType.COMMUNITY)
        cluster2 = Cluster(id="cluster_2", cluster_type=ClusterType.HIERARCHICAL)
        cluster3 = Cluster(id="cluster_3", cluster_type=ClusterType.COMMUNITY)

        self.kg.add_cluster(cluster1)
        self.kg.add_cluster(cluster2)
        self.kg.add_cluster(cluster3)

        # Test getting by type
        community_clusters = self.kg.get_clusters_by_type(ClusterType.COMMUNITY)
        self.assertEqual(len(community_clusters), 2)

        cluster_ids = {c.id for c in community_clusters}
        self.assertEqual(cluster_ids, {"cluster_1", "cluster_3"})

    def test_cluster_hierarchy_management(self):
        """Test cluster parent-child relationship management."""
        parent_cluster = Cluster(id="parent_cluster")
        child_cluster = Cluster(id="child_cluster", parent_cluster_id="parent_cluster")
        parent_cluster.add_child_cluster("child_cluster")

        self.kg.add_cluster(parent_cluster)
        self.kg.add_cluster(child_cluster)

        # Remove parent cluster
        result = self.kg.remove_cluster("parent_cluster")
        self.assertTrue(result)

        # Verify child cluster's parent reference was cleared
        remaining_child = self.kg.get_cluster("child_cluster")
        self.assertEqual(remaining_child.parent_cluster_id, "")


class TestOptimizedKnowledgeGraphTextChunkManagement(unittest.TestCase):
    """Test text chunk management in OptimizedKnowledgeGraph."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

        # Add test entities
        self.entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        self.kg.add_entity(self.entity1)

    def test_add_text_chunk(self):
        """Test adding text chunks to the knowledge graph."""
        text_chunk = TextChunk(
            id="chunk_1",
            content="This is test content about Entity 1.",
            title="Test Chunk",
            entities={"entity_1"},
        )

        self.kg.add_text_chunk(text_chunk)

        # Verify text chunk was added
        self.assertIn("chunk_1", self.kg.text_chunks)
        self.assertEqual(self.kg.text_chunks["chunk_1"], text_chunk)

        # Verify indexes were updated
        entity_chunks = self.kg.index_manager.get_entity_text_chunks("entity_1")
        self.assertIn("chunk_1", entity_chunks)

        chunk_entities = self.kg.index_manager.get_text_chunk_entities("chunk_1")
        self.assertIn("entity_1", chunk_entities)

    def test_remove_text_chunk(self):
        """Test removing text chunks from the knowledge graph."""
        text_chunk = TextChunk(id="chunk_1", content="Test content", entities={"entity_1"})
        self.kg.add_text_chunk(text_chunk)

        # Remove text chunk
        result = self.kg.remove_text_chunk("chunk_1")
        self.assertTrue(result)

        # Verify removal
        self.assertNotIn("chunk_1", self.kg.text_chunks)

        # Verify indexes were updated
        entity_chunks = self.kg.index_manager.get_entity_text_chunks("entity_1")
        self.assertNotIn("chunk_1", entity_chunks)

    def test_get_text_chunk(self):
        """Test getting text chunks by ID."""
        text_chunk = TextChunk(id="chunk_1", content="Test content")
        self.kg.add_text_chunk(text_chunk)

        # Test successful get
        retrieved = self.kg.get_text_chunk("chunk_1")
        self.assertEqual(retrieved, text_chunk)

        # Test get non-existent chunk
        retrieved = self.kg.get_text_chunk("nonexistent")
        self.assertIsNone(retrieved)

    def test_search_text_chunks(self):
        """Test text chunk search functionality."""
        chunk1 = TextChunk(
            id="chunk_1", content="Python programming tutorial", title="Python Guide"
        )
        chunk2 = TextChunk(id="chunk_2", content="Machine learning basics", title="ML Introduction")
        chunk3 = TextChunk(
            id="chunk_3", content="Advanced Python techniques", title="Advanced Programming"
        )

        self.kg.add_text_chunk(chunk1)
        self.kg.add_text_chunk(chunk2)
        self.kg.add_text_chunk(chunk3)

        # Test content search
        results = self.kg.search_text_chunks("Python", limit=5)
        self.assertEqual(len(results), 2)

        result_ids = {c.id for c in results}
        self.assertEqual(result_ids, {"chunk_1", "chunk_3"})

        # Test title search
        results = self.kg.search_text_chunks("Guide", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "chunk_1")


class TestOptimizedKnowledgeGraphAnalysis(unittest.TestCase):
    """Test graph analysis functionality in OptimizedKnowledgeGraph."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.kg = OptimizedKnowledgeGraph()

        # Create test data
        self.entities = []
        for i in range(5):
            entity = Entity(
                id=f"entity_{i}",
                name=f"Entity {i}",
                entity_type=EntityType.PERSON if i % 2 == 0 else EntityType.ORGANIZATION,
            )
            self.entities.append(entity)
            self.kg.add_entity(entity)

        # Add relations to create connected components
        relation1 = Relation(
            id="relation_1",
            head_entity=self.entities[0],
            tail_entity=self.entities[1],
            relation_type="WORKS_FOR",
        )
        relation2 = Relation(
            id="relation_2",
            head_entity=self.entities[1],
            tail_entity=self.entities[2],
            relation_type="RELATED_TO",
        )
        # entity_3 and entity_4 remain isolated

        self.kg.add_relation(relation1)
        self.kg.add_relation(relation2)

    def test_get_graph_statistics(self):
        """Test graph statistics calculation (cached)."""
        # First call - should calculate and cache
        stats1 = self.kg.get_graph_statistics()

        # Verify statistics content
        self.assertEqual(stats1["total_entities"], 5)
        self.assertEqual(stats1["total_relations"], 2)
        self.assertEqual(stats1["total_clusters"], 0)
        self.assertEqual(stats1["total_text_chunks"], 0)

        # Check entity type breakdown
        self.assertEqual(stats1["entity_types"]["PERSON"], 3)
        self.assertEqual(stats1["entity_types"]["ORGANIZATION"], 2)

        # Check relation type breakdown
        self.assertEqual(stats1["relation_types"]["WORKS_FOR"], 1)
        self.assertEqual(stats1["relation_types"]["COLLABORATES_WITH"], 1)

        # Second call - should use cache
        start_time = time.time()
        stats2 = self.kg.get_graph_statistics()
        cache_time = time.time() - start_time

        # Should be very fast due to caching
        self.assertLess(cache_time, 0.001)
        self.assertEqual(stats1, stats2)

    def test_get_connected_components(self):
        """Test connected components analysis (cached)."""
        # First call - should calculate and cache
        components = self.kg.get_connected_components()

        # Should have 3 components: {0,1,2}, {3}, {4}
        self.assertEqual(len(components), 3)

        # Find the large component
        large_component = max(components, key=len)
        self.assertEqual(len(large_component), 3)
        self.assertEqual(large_component, {"entity_0", "entity_1", "entity_2"})

        # Verify isolated entities
        isolated_components = [comp for comp in components if len(comp) == 1]
        self.assertEqual(len(isolated_components), 2)

    def test_average_degree_calculation(self):
        """Test average degree calculation."""
        stats = self.kg.get_graph_statistics()

        # entity_0: 1 relation, entity_1: 2 relations, entity_2: 1 relation
        # entity_3: 0 relations, entity_4: 0 relations
        # Average = (1 + 2 + 1 + 0 + 0) / 5 = 0.8
        expected_avg_degree = 4.0 / 5  # Total connections / entities
        self.assertAlmostEqual(stats["average_entity_degree"], expected_avg_degree, places=2)


class TestOptimizedKnowledgeGraphPerformance(unittest.TestCase):
    """Test performance optimization features."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

    def test_get_performance_metrics(self):
        """Test performance metrics collection."""
        # Perform some operations
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity)

        # Get performance metrics
        metrics = self.kg.get_performance_metrics()

        # Verify structure
        self.assertIn("graph_metrics", metrics)
        self.assertIn("entity_manager", metrics)
        self.assertIn("relation_manager", metrics)
        self.assertIn("index_statistics", metrics)
        self.assertIn("cache_statistics", metrics)
        self.assertIn("optimization_summary", metrics)

        # Verify operation count updated
        self.assertGreater(metrics["graph_metrics"]["total_operations"], 0)

    def test_optimize_performance(self):
        """Test performance optimization operations."""
        # Add some cached data with short TTL
        self.kg.cache_manager.put("test_key", "test_value", ttl=0.1)

        # Wait for expiration
        time.sleep(0.15)

        # Run optimization
        optimization_summary = self.kg.optimize_performance()

        # Verify optimization performed
        self.assertIn("cache_cleanup", optimization_summary)
        self.assertIn("index_rebuild", optimization_summary)
        self.assertIn("memory_freed", optimization_summary)

        # Should have cleaned up expired entries
        self.assertGreater(optimization_summary["cache_cleanup"], 0)

    def test_clear_caches(self):
        """Test cache clearing functionality."""
        # Add some cached data
        self.kg.cache_manager.put("test_key", "test_value")
        self.assertEqual(self.kg.cache_manager.get_statistics()["size"], 1)

        # Clear caches
        self.kg.clear_caches()

        # Verify caches are cleared
        self.assertEqual(self.kg.cache_manager.get_statistics()["size"], 0)

    def test_rebuild_indexes(self):
        """Test manual index rebuilding."""
        # Add data through direct manipulation (bypassing managers)
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.entities["entity_1"] = entity

        # Clear indexes
        self.kg.index_manager.clear_all_indexes()

        # Verify index is empty
        person_entities = self.kg.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertEqual(len(person_entities), 0)

        # Rebuild indexes
        self.kg.rebuild_indexes()

        # Verify index was rebuilt
        person_entities = self.kg.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertIn("entity_1", person_entities)


class TestOptimizedKnowledgeGraphValidation(unittest.TestCase):
    """Test validation functionality in OptimizedKnowledgeGraph."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

    def test_is_valid_basic(self):
        """Test basic graph validation."""
        # Empty graph should be valid
        self.assertTrue(self.kg.is_valid())

        # Add valid entity
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity)
        self.assertTrue(self.kg.is_valid())

    def test_is_valid_caching(self):
        """Test validation result caching."""
        # First validation call
        start_time = time.time()
        result1 = self.kg.is_valid()
        first_call_time = time.time() - start_time

        # Second validation call (should be cached)
        start_time = time.time()
        result2 = self.kg.is_valid()
        second_call_time = time.time() - start_time

        # Results should be the same
        self.assertEqual(result1, result2)

        # Second call should be faster due to caching
        self.assertLess(second_call_time, first_call_time)

    def test_validate_integrity(self):
        """Test comprehensive integrity validation."""
        # Add entities
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        self.kg.add_entity(entity1)
        self.kg.add_entity(entity2)

        # Add valid relation
        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="WORKS_FOR"
        )
        self.kg.add_relation(relation)

        # Valid graph should have no errors
        errors = self.kg.validate_integrity()
        self.assertEqual(len(errors), 0)

        # Manually create invalid reference (entity not in graph)
        invalid_entity = Entity(id="invalid_entity", name="Invalid", entity_type=EntityType.PERSON)
        invalid_relation = Relation(
            id="invalid_relation",
            head_entity=invalid_entity,  # This entity is not in the graph
            tail_entity=entity2,
            relation_type="RELATED_TO",
        )
        self.kg.relations["invalid_relation"] = invalid_relation

        # Should detect the error
        errors = self.kg.validate_integrity()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("non-existent head entity" in error for error in errors))


class TestOptimizedKnowledgeGraphSerialization(unittest.TestCase):
    """Test serialization and deserialization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph(name="Test Graph", description="Test Description")

        # Add sample data
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        self.kg.add_entity(entity1)
        self.kg.add_entity(entity2)

        relation = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="WORKS_FOR"
        )
        self.kg.add_relation(relation)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = self.kg.to_dict()

        # Verify structure
        self.assertIn("id", data)
        self.assertIn("name", data)
        self.assertIn("description", data)
        self.assertIn("entities", data)
        self.assertIn("relations", data)
        self.assertIn("clusters", data)
        self.assertIn("text_chunks", data)
        self.assertIn("metadata", data)
        self.assertIn("created_at", data)
        self.assertIn("updated_at", data)

        # Verify content
        self.assertEqual(data["name"], "Test Graph")
        self.assertEqual(data["description"], "Test Description")
        self.assertEqual(len(data["entities"]), 2)
        self.assertEqual(len(data["relations"]), 1)

    def test_from_dict(self):
        """Test creation from dictionary."""
        # Get dictionary representation
        original_data = self.kg.to_dict()

        # Create new instance from dictionary
        new_kg = OptimizedKnowledgeGraph.from_dict(original_data)

        # Verify recreation
        self.assertEqual(new_kg.name, self.kg.name)
        self.assertEqual(new_kg.description, self.kg.description)
        self.assertEqual(len(new_kg.entities), len(self.kg.entities))
        self.assertEqual(len(new_kg.relations), len(self.kg.relations))

        # Verify indexes were rebuilt
        person_entities = new_kg.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertIn("entity_1", person_entities)

        relation_entities = new_kg.index_manager.get_relation_entities("relation_1")
        self.assertEqual(relation_entities, ("entity_1", "entity_2"))

    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        # Add more complex data
        cluster = Cluster(id="cluster_1", entities={"entity_1"})
        text_chunk = TextChunk(id="chunk_1", content="Test content", entities={"entity_1"})

        self.kg.add_cluster(cluster)
        self.kg.add_text_chunk(text_chunk)

        # Perform roundtrip
        data = self.kg.to_dict()
        reconstructed_kg = OptimizedKnowledgeGraph.from_dict(data)

        # Verify complete reconstruction
        self.assertEqual(len(reconstructed_kg.entities), len(self.kg.entities))
        self.assertEqual(len(reconstructed_kg.relations), len(self.kg.relations))
        self.assertEqual(len(reconstructed_kg.clusters), len(self.kg.clusters))
        self.assertEqual(len(reconstructed_kg.text_chunks), len(self.kg.text_chunks))

        # Verify functionality is preserved
        stats_original = self.kg.get_graph_statistics()
        stats_reconstructed = reconstructed_kg.get_graph_statistics()

        self.assertEqual(stats_original["total_entities"], stats_reconstructed["total_entities"])
        self.assertEqual(stats_original["total_relations"], stats_reconstructed["total_relations"])


class TestOptimizedKnowledgeGraphMerging(unittest.TestCase):
    """Test graph merging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg1 = OptimizedKnowledgeGraph(name="Graph 1")
        self.kg2 = OptimizedKnowledgeGraph(name="Graph 2")

    def test_merge_basic(self):
        """Test basic graph merging."""
        # Add entities to first graph
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        self.kg1.add_entity(entity1)

        # Add entities to second graph
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        self.kg2.add_entity(entity2)

        # Merge second graph into first
        self.kg1.merge(self.kg2)

        # Verify merge
        self.assertEqual(len(self.kg1.entities), 2)
        self.assertIn("entity_1", self.kg1.entities)
        self.assertIn("entity_2", self.kg1.entities)

        # Verify indexes were updated
        person_entities = self.kg1.index_manager.get_entities_by_type(EntityType.PERSON)
        org_entities = self.kg1.index_manager.get_entities_by_type(EntityType.ORGANIZATION)
        self.assertIn("entity_1", person_entities)
        self.assertIn("entity_2", org_entities)

    def test_merge_with_overlapping_ids(self):
        """Test merging with overlapping entity IDs."""
        # Add entity with same ID to both graphs
        entity1_v1 = Entity(id="entity_1", name="Version 1", entity_type=EntityType.PERSON)
        entity1_v2 = Entity(id="entity_1", name="Version 2", entity_type=EntityType.PERSON)

        self.kg1.add_entity(entity1_v1)
        self.kg2.add_entity(entity1_v2)

        # Merge - second version should overwrite first
        self.kg1.merge(self.kg2)

        # Verify overwrite
        self.assertEqual(len(self.kg1.entities), 1)
        self.assertEqual(self.kg1.entities["entity_1"].name, "Version 2")

    def test_merge_complex_graph(self):
        """Test merging complex graphs with relations and clusters."""
        # Set up first graph
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        self.kg1.add_entity(entity1)
        self.kg1.add_entity(entity2)

        relation1 = Relation(
            id="relation_1", head_entity=entity1, tail_entity=entity2, relation_type="WORKS_FOR"
        )
        self.kg1.add_relation(relation1)

        # Set up second graph
        entity3 = Entity(id="entity_3", name="Entity 3", entity_type=EntityType.PERSON)
        self.kg2.add_entity(entity3)

        cluster = Cluster(id="cluster_1", entities={"entity_3"})
        self.kg2.add_cluster(cluster)

        # Merge
        self.kg1.merge(self.kg2)

        # Verify complete merge
        self.assertEqual(len(self.kg1.entities), 3)
        self.assertEqual(len(self.kg1.relations), 1)
        self.assertEqual(len(self.kg1.clusters), 1)

        # Verify indexes updated correctly
        person_entities = self.kg1.index_manager.get_entities_by_type(EntityType.PERSON)
        self.assertEqual(len(person_entities), 2)  # entity_1 and entity_3


class TestOptimizedKnowledgeGraphUtilityOperations(unittest.TestCase):
    """Test utility operations like clear, touch, etc."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

    def test_clear_operation(self):
        """Test clearing all graph data."""
        # Add test data
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity)

        relation = Relation(
            id="relation_1", head_entity=entity, tail_entity=entity, relation_type="RELATED_TO"
        )
        self.kg.add_relation(relation)

        # Verify data exists
        self.assertEqual(len(self.kg.entities), 1)
        self.assertEqual(len(self.kg.relations), 1)

        # Clear graph
        self.kg.clear()

        # Verify everything is cleared
        self.assertEqual(len(self.kg.entities), 0)
        self.assertEqual(len(self.kg.relations), 0)
        self.assertEqual(len(self.kg.clusters), 0)
        self.assertEqual(len(self.kg.text_chunks), 0)

        # Verify indexes are cleared
        stats = self.kg.index_manager.get_statistics()
        self.assertEqual(stats["total_indexes"], 0)

        # Verify caches are cleared
        cache_stats = self.kg.cache_manager.get_statistics()
        self.assertEqual(cache_stats["size"], 0)

    def test_touch_operation(self):
        """Test touch operation updates timestamp."""
        original_time = self.kg.updated_at

        # Small delay to ensure timestamp difference
        time.sleep(0.01)

        # Touch the graph
        self.kg.touch()

        # Verify timestamp was updated
        self.assertGreater(self.kg.updated_at, original_time)

    def test_metadata_management(self):
        """Test metadata operations."""
        # Add metadata
        self.kg.metadata["version"] = "1.0"
        self.kg.metadata["tags"] = ["test", "demo"]

        # Verify metadata is preserved in serialization
        data = self.kg.to_dict()
        self.assertEqual(data["metadata"]["version"], "1.0")
        self.assertEqual(data["metadata"]["tags"], ["test", "demo"])

        # Verify metadata is restored from dict
        new_kg = OptimizedKnowledgeGraph.from_dict(data)
        self.assertEqual(new_kg.metadata["version"], "1.0")
        self.assertEqual(new_kg.metadata["tags"], ["test", "demo"])


class TestOptimizedKnowledgeGraphEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.kg = OptimizedKnowledgeGraph()

    def test_removing_nonexistent_items(self):
        """Test removing non-existent items."""
        # Test removing non-existent entity
        result = self.kg.remove_entity("nonexistent")
        self.assertFalse(result)

        # Test removing non-existent relation
        result = self.kg.remove_relation("nonexistent")
        self.assertFalse(result)

        # Test removing non-existent cluster
        result = self.kg.remove_cluster("nonexistent")
        self.assertFalse(result)

        # Test removing non-existent text chunk
        result = self.kg.remove_text_chunk("nonexistent")
        self.assertFalse(result)

    def test_large_graph_performance(self):
        """Test performance with larger graph."""
        # Create larger dataset
        num_entities = 500
        start_time = time.time()

        # Add entities
        for i in range(num_entities):
            entity = Entity(
                id=f"entity_{i}",
                name=f"Entity {i}",
                entity_type=EntityType.PERSON if i % 2 == 0 else EntityType.ORGANIZATION,
            )
            self.kg.add_entity(entity)

        entity_creation_time = time.time() - start_time

        # Test indexed query performance
        start_time = time.time()
        person_entities = self.kg.get_entities_by_type(EntityType.PERSON)
        query_time = time.time() - start_time

        # Verify results
        self.assertEqual(len(person_entities), 250)  # Half should be persons

        # Query should be very fast due to indexing
        self.assertLess(query_time, 0.01)  # Sub-10ms

        # Entity creation should be reasonable
        self.assertLess(entity_creation_time, 2.0)  # Should complete within 2 seconds

    def test_cache_invalidation_on_data_changes(self):
        """Test that caches are properly invalidated when data changes."""
        # Add entity and generate cached statistics
        entity = Entity(id="entity_1", name="Test Entity", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity)

        stats1 = self.kg.get_graph_statistics()  # This will be cached
        self.assertEqual(stats1["total_entities"], 1)

        # Add another entity
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.PERSON)
        self.kg.add_entity(entity2)

        # Statistics should reflect the change (cache should be invalidated)
        stats2 = self.kg.get_graph_statistics()
        self.assertEqual(stats2["total_entities"], 2)


class TestOptimizedKnowledgeGraphIntegration(unittest.TestCase):
    """Integration tests combining multiple OptimizedKnowledgeGraph features."""

    def test_complete_workflow(self):
        """Test a complete workflow using multiple features."""
        kg = OptimizedKnowledgeGraph(name="Integration Test Graph")

        # 1. Add entities
        person = Entity(id="person_1", name="John Doe", entity_type=EntityType.PERSON)
        company = Entity(id="company_1", name="Tech Corp", entity_type=EntityType.ORGANIZATION)
        kg.add_entity(person)
        kg.add_entity(company)

        # 2. Add relation
        works_for = Relation(
            id="works_for_1", head_entity=person, tail_entity=company, relation_type="WORKS_FOR"
        )
        kg.add_relation(works_for)

        # 3. Add cluster
        team_cluster = Cluster(
            id="team_1", entities={"person_1"}, cluster_type=ClusterType.COMMUNITY
        )
        kg.add_cluster(team_cluster)

        # 4. Add text chunk
        resume_chunk = TextChunk(
            id="resume_1",
            content="John Doe works at Tech Corp as a software engineer.",
            title="John's Resume",
            entities={"person_1"},
            relations={"works_for_1"},
        )
        kg.add_text_chunk(resume_chunk)

        # 5. Test comprehensive statistics
        stats = kg.get_graph_statistics()
        self.assertEqual(stats["total_entities"], 2)
        self.assertEqual(stats["total_relations"], 1)
        self.assertEqual(stats["total_clusters"], 1)
        self.assertEqual(stats["total_text_chunks"], 1)

        # 6. Test connected components
        components = kg.get_connected_components()
        self.assertEqual(len(components), 1)  # All connected
        self.assertEqual(len(components[0]), 2)  # Both entities

        # 7. Test performance metrics
        metrics = kg.get_performance_metrics()
        self.assertGreater(metrics["graph_metrics"]["total_operations"], 0)

        # 8. Test serialization preserves everything
        data = kg.to_dict()
        reconstructed = OptimizedKnowledgeGraph.from_dict(data)

        reconstructed_stats = reconstructed.get_graph_statistics()
        self.assertEqual(stats["total_entities"], reconstructed_stats["total_entities"])
        self.assertEqual(stats["total_relations"], reconstructed_stats["total_relations"])

    def test_optimization_under_load(self):
        """Test optimization features under simulated load."""
        kg = OptimizedKnowledgeGraph()

        # Create moderate load
        for i in range(100):
            entity = Entity(id=f"entity_{i}", name=f"Entity {i}", entity_type=EntityType.PERSON)
            kg.add_entity(entity)

        # Perform many queries to generate cache data
        for _ in range(50):
            kg.get_entities_by_type(EntityType.PERSON)
            kg.get_graph_statistics()

        # Check cache performance
        cache_stats = kg.cache_manager.get_statistics()
        self.assertGreater(cache_stats["hits"], 0)
        self.assertGreater(cache_stats["hit_ratio"], 0.5)  # Should have good hit ratio

        # Test optimization
        optimization_summary = kg.optimize_performance()
        self.assertIsInstance(optimization_summary, dict)

        # Performance metrics should show optimization benefits
        metrics = kg.get_performance_metrics()
        self.assertGreater(metrics["optimization_summary"]["total_cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()
