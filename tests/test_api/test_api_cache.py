"""Tests for Cache Management API router."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestCacheRouter(unittest.TestCase):
    """Test Cache Management API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_text_chunks_success(self, mock_get_agraph):
        """Test successful retrieval of cached text chunks."""
        # Mock text chunks
        mock_chunks = {}
        for i in range(15):
            mock_chunk = MagicMock()
            mock_chunk.id = f"chunk_{i}"
            mock_chunk.content = f"This is text chunk content {i}" + " " * 200  # Long content
            mock_chunk.title = f"Title {i}"
            mock_chunk.source = f"source_{i}.txt"
            mock_chunk.start_index = i * 100
            mock_chunk.end_index = (i + 1) * 100
            mock_chunks[f"chunk_{i}"] = mock_chunk

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.text_chunks = mock_chunks

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/text-chunks?page=2&page_size=5")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify pagination
        response_data = data["data"]
        self.assertEqual(len(response_data["text_chunks"]), 5)  # page_size
        self.assertEqual(response_data["total_count"], 15)
        self.assertEqual(response_data["page"], 2)
        self.assertEqual(response_data["total_pages"], 3)  # 15 / 5 = 3

        # Verify content truncation
        first_chunk = response_data["text_chunks"][0]
        self.assertTrue(first_chunk["content"].endswith("..."))
        self.assertIn("full_content", first_chunk)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_text_chunks_no_cache(self, mock_get_agraph):
        """Test getting text chunks when no cache exists."""
        # Mock AGraph without knowledge graph
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = None
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/text-chunks")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "No text chunks found in cache")
        self.assertEqual(data["data"]["total_count"], 0)
        self.assertEqual(len(data["data"]["text_chunks"]), 0)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_text_chunks_with_filter(self, mock_get_agraph):
        """Test getting text chunks with filter."""
        # Mock text chunks with different content
        mock_chunks = {}
        contents = [
            "Python programming tutorial",
            "Java development guide",
            "Machine learning basics",
        ]

        for i, content in enumerate(contents):
            mock_chunk = MagicMock()
            mock_chunk.id = f"chunk_{i}"
            mock_chunk.content = content
            mock_chunk.title = f"Title {i}"
            mock_chunk.source = f"source_{i}.txt"
            mock_chunk.start_index = i * 100
            mock_chunk.end_index = (i + 1) * 100
            mock_chunks[f"chunk_{i}"] = mock_chunk

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.text_chunks = mock_chunks

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/text-chunks?filter_by=python")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should only return chunks containing "python" (case insensitive)
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 1)
        self.assertEqual(len(response_data["text_chunks"]), 1)
        self.assertIn("Python", response_data["text_chunks"][0]["content"])

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_entities_success(self, mock_get_agraph):
        """Test successful retrieval of cached entities."""
        # Mock entities
        mock_entities = {}
        for i in range(8):
            mock_entity = MagicMock()
            mock_entity.id = f"entity_{i}"
            mock_entity.name = f"Entity {i}"
            mock_entity.entity_type.value = "PERSON" if i % 2 == 0 else "ORGANIZATION"
            mock_entity.description = f"Description for entity {i}"
            mock_entity.properties = {"property": f"value_{i}"}
            mock_entity.confidence = 0.8 + (i * 0.01)
            mock_entity.aliases = [f"alias_{i}"]
            mock_entity.text_chunks = [f"chunk_{i}"]
            mock_entities[f"entity_{i}"] = mock_entity

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.entities = mock_entities

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/entities?page=1&page_size=5")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify pagination and data structure
        response_data = data["data"]
        self.assertEqual(len(response_data["entities"]), 5)
        self.assertEqual(response_data["total_count"], 8)
        self.assertEqual(response_data["total_pages"], 2)

        # Verify entity data structure
        first_entity = response_data["entities"][0]
        self.assertIn("id", first_entity)
        self.assertIn("name", first_entity)
        self.assertIn("entity_type", first_entity)
        self.assertIn("confidence", first_entity)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_entities_with_filter(self, mock_get_agraph):
        """Test getting entities with name filter."""
        # Mock entities with different names
        mock_entities = {}
        names = ["John Smith", "Microsoft Corporation", "Apple Inc"]

        for i, name in enumerate(names):
            mock_entity = MagicMock()
            mock_entity.id = f"entity_{i}"
            mock_entity.name = name
            mock_entity.entity_type.value = "PERSON" if "Smith" in name else "ORGANIZATION"
            mock_entity.description = f"Description for {name}"
            mock_entity.properties = {}
            mock_entity.confidence = 0.9
            mock_entity.aliases = []
            mock_entity.text_chunks = []
            mock_entities[f"entity_{i}"] = mock_entity

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.entities = mock_entities

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/entities?filter_by=microsoft")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should only return Microsoft entity
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 1)
        self.assertEqual(len(response_data["entities"]), 1)
        self.assertIn("Microsoft", response_data["entities"][0]["name"])

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_relations_success(self, mock_get_agraph):
        """Test successful retrieval of cached relations."""
        # Mock entities for relations
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity1.name = "John Smith"
        mock_entity1.entity_type.value = "PERSON"

        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"
        mock_entity2.name = "Microsoft"
        mock_entity2.entity_type.value = "ORGANIZATION"

        # Mock relations
        mock_relations = {}
        for i in range(6):
            mock_relation = MagicMock()
            mock_relation.id = f"relation_{i}"
            mock_relation.head_entity = mock_entity1 if i % 2 == 0 else mock_entity2
            mock_relation.tail_entity = mock_entity2 if i % 2 == 0 else mock_entity1
            mock_relation.relation_type.value = "WORKS_FOR" if i % 2 == 0 else "KNOWS"
            mock_relation.description = f"Relation description {i}"
            mock_relation.properties = {"strength": i}
            mock_relation.confidence = 0.8 + (i * 0.02)
            mock_relation.text_chunks = [f"chunk_{i}"]
            mock_relations[f"relation_{i}"] = mock_relation

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.relations = mock_relations

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/relations?page=1&page_size=3")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify pagination and data structure
        response_data = data["data"]
        self.assertEqual(len(response_data["relations"]), 3)
        self.assertEqual(response_data["total_count"], 6)
        self.assertEqual(response_data["total_pages"], 2)

        # Verify relation data structure
        first_relation = response_data["relations"][0]
        self.assertIn("id", first_relation)
        self.assertIn("head_entity", first_relation)
        self.assertIn("tail_entity", first_relation)
        self.assertIn("relation_type", first_relation)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_relations_with_filter(self, mock_get_agraph):
        """Test getting relations with filter."""
        # Mock entities
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity1.name = "John Smith"
        mock_entity1.entity_type.value = "PERSON"

        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"
        mock_entity2.name = "Microsoft Corporation"
        mock_entity2.entity_type.value = "ORGANIZATION"

        # Mock relations with different types
        mock_relations = {}
        relation_types = ["WORKS_FOR", "KNOWS", "COLLABORATES_WITH"]

        for i, rel_type in enumerate(relation_types):
            mock_relation = MagicMock()
            mock_relation.id = f"relation_{i}"
            mock_relation.head_entity = mock_entity1
            mock_relation.tail_entity = mock_entity2
            mock_relation.relation_type.value = rel_type
            mock_relation.description = f"Description {i}"
            mock_relation.properties = {}
            mock_relation.confidence = 0.8
            mock_relation.text_chunks = []
            mock_relations[f"relation_{i}"] = mock_relation

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.relations = mock_relations

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/relations?filter_by=works")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should only return WORKS_FOR relation
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 1)
        self.assertEqual(len(response_data["relations"]), 1)
        self.assertEqual(response_data["relations"][0]["relation_type"], "WORKS_FOR")

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_clusters_success(self, mock_get_agraph):
        """Test successful retrieval of cached clusters."""
        # Mock entities for clusters
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity1.name = "John"
        mock_entity1.entity_type.value = "PERSON"

        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"
        mock_entity2.name = "Jane"
        mock_entity2.entity_type.value = "PERSON"

        # Mock relations for clusters
        mock_relation = MagicMock()
        mock_relation.id = "r1"
        mock_relation.relation_type.value = "KNOWS"
        mock_relation.head_entity = mock_entity1
        mock_relation.tail_entity = mock_entity2

        # Mock clusters
        mock_clusters = {}
        for i in range(4):
            mock_cluster = MagicMock()
            mock_cluster.id = f"cluster_{i}"
            mock_cluster.name = f"Cluster {i}"
            mock_cluster.description = f"Description for cluster {i}"
            mock_cluster.entities = [mock_entity1, mock_entity2] if i % 2 == 0 else [mock_entity1]
            mock_cluster.relations = [mock_relation] if i == 0 else []
            mock_clusters[f"cluster_{i}"] = mock_cluster

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.clusters = mock_clusters

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/clusters?page=1&page_size=3")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify response structure
        response_data = data["data"]
        self.assertEqual(len(response_data["clusters"]), 3)
        self.assertEqual(response_data["total_count"], 4)
        self.assertEqual(response_data["total_pages"], 2)

        # Verify cluster data structure
        first_cluster = response_data["clusters"][0]
        self.assertIn("id", first_cluster)
        self.assertIn("name", first_cluster)
        self.assertIn("entities", first_cluster)
        self.assertIn("relations", first_cluster)
        self.assertIn("entity_count", first_cluster)
        self.assertIn("relation_count", first_cluster)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_clusters_with_filter(self, mock_get_agraph):
        """Test getting clusters with name filter."""
        # Mock clusters with different names
        mock_clusters = {}
        names = ["Technology Cluster", "Business Cluster", "Research Group"]

        for i, name in enumerate(names):
            mock_cluster = MagicMock()
            mock_cluster.id = f"cluster_{i}"
            mock_cluster.name = name
            mock_cluster.description = f"Description for {name}"
            mock_cluster.entities = []
            mock_cluster.relations = []
            mock_clusters[f"cluster_{i}"] = mock_cluster

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.clusters = mock_clusters

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/clusters?filter_by=technology")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should only return Technology Cluster
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 1)
        self.assertEqual(len(response_data["clusters"]), 1)
        self.assertIn("Technology", response_data["clusters"][0]["name"])

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_entities_no_cache(self, mock_get_agraph):
        """Test getting entities when no cache exists."""
        # Mock AGraph without knowledge graph
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = None
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/entities")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "No entities found in cache")
        self.assertEqual(data["data"]["total_count"], 0)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_get_cached_clusters_no_cache(self, mock_get_agraph):
        """Test getting clusters when no cache exists."""
        # Mock AGraph without knowledge graph
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = None
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/clusters")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "No clusters found in cache")
        self.assertEqual(data["data"]["total_count"], 0)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_cache_endpoints_error_handling(self, mock_get_agraph):
        """Test error handling in cache endpoints."""
        # Mock AGraph instance that raises exception
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph.text_chunks.values.side_effect = Exception("Cache error")
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/text-chunks")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Cache error")

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_entities_filter_by_description(self, mock_get_agraph):
        """Test filtering entities by description."""
        # Mock entities with different descriptions
        mock_entities = {}
        descriptions = ["Python developer", "Java developer", "Project manager"]

        for i, desc in enumerate(descriptions):
            mock_entity = MagicMock()
            mock_entity.id = f"entity_{i}"
            mock_entity.name = f"Person {i}"
            mock_entity.entity_type.value = "PERSON"
            mock_entity.description = desc
            mock_entity.properties = {}
            mock_entity.confidence = 0.8
            mock_entity.aliases = []
            mock_entity.text_chunks = []
            mock_entities[f"entity_{i}"] = mock_entity

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.entities = mock_entities

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/entities?filter_by=developer")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should return both Python and Java developers
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 2)

    @patch("agraph.api.routers.cache.get_agraph_instance_dependency")
    def test_relations_filter_by_entity_name(self, mock_get_agraph):
        """Test filtering relations by entity name."""
        # Mock entities
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity1.name = "John Smith"
        mock_entity1.entity_type.value = "PERSON"

        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"
        mock_entity2.name = "Microsoft"
        mock_entity2.entity_type.value = "ORGANIZATION"

        mock_entity3 = MagicMock()
        mock_entity3.id = "e3"
        mock_entity3.name = "Apple"
        mock_entity3.entity_type.value = "ORGANIZATION"

        # Mock relations
        mock_relations = {}
        relations_data = [
            (mock_entity1, mock_entity2, "WORKS_FOR"),
            (mock_entity1, mock_entity3, "KNOWS"),
            (mock_entity2, mock_entity3, "COMPETES_WITH"),
        ]

        for i, (head, tail, rel_type) in enumerate(relations_data):
            mock_relation = MagicMock()
            mock_relation.id = f"relation_{i}"
            mock_relation.head_entity = head
            mock_relation.tail_entity = tail
            mock_relation.relation_type.value = rel_type
            mock_relation.description = f"Description {i}"
            mock_relation.properties = {}
            mock_relation.confidence = 0.8
            mock_relation.text_chunks = []
            mock_relations[f"relation_{i}"] = mock_relation

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.relations = mock_relations

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.knowledge_graph = mock_kg
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/cache/relations?filter_by=john")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should return relations involving John Smith
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 2)  # John -> Microsoft, John -> Apple

    def test_cache_endpoints_pagination_edge_cases(self):
        """Test pagination edge cases in cache endpoints."""
        with patch("agraph.api.routers.cache.get_agraph_instance_dependency") as mock_get_agraph:
            # Mock empty cache
            mock_kg = MagicMock()
            mock_kg.text_chunks = {}
            mock_agraph = MagicMock()
            mock_agraph.knowledge_graph = mock_kg
            mock_get_agraph.return_value = mock_agraph

            # Test page beyond available data
            response = self.client.get("/cache/text-chunks?page=10&page_size=5")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)

            # Should return empty results but valid pagination info
            response_data = data["data"]
            self.assertEqual(len(response_data["text_chunks"]), 0)
            self.assertEqual(response_data["page"], 10)


if __name__ == "__main__":
    unittest.main()
