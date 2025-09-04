"""Tests for Knowledge Graph API router."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus
from agraph.api.routers.knowledge_graph import _extract_text_from_document


class TestKnowledgeGraphRouter(unittest.TestCase):
    """Test Knowledge Graph API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    @patch("agraph.api.routers.knowledge_graph.get_document_manager")
    def test_build_knowledge_graph_with_texts(self, mock_doc_manager, mock_agraph_instance):
        """Test building knowledge graph with direct texts."""
        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_kg = MagicMock()
        mock_kg.name = "Test Graph"
        mock_kg.description = "Test Description"
        mock_kg.entities = {"e1": MagicMock(), "e2": MagicMock()}
        mock_kg.relations = {"r1": MagicMock()}
        mock_kg.clusters = {"c1": MagicMock()}
        mock_kg.text_chunks = {"t1": MagicMock(), "t2": MagicMock()}

        mock_agraph.build_from_texts.return_value = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test data
        test_data = {
            "texts": ["This is test text 1", "This is test text 2"],
            "graph_name": "Test Graph",
            "graph_description": "Test Description",
            "use_cache": True,
            "save_to_vector_store": True,
        }

        response = self.client.post("/knowledge-graph/build", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Knowledge graph built successfully")

        # Verify response data structure
        response_data = data["data"]
        self.assertEqual(response_data["graph_name"], "Test Graph")
        self.assertEqual(response_data["entities_count"], 2)
        self.assertEqual(response_data["relations_count"], 1)
        self.assertEqual(response_data["clusters_count"], 1)
        self.assertEqual(response_data["text_chunks_count"], 2)
        self.assertEqual(response_data["total_texts_processed"], 2)

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    @patch("agraph.api.routers.knowledge_graph.get_document_manager")
    def test_build_knowledge_graph_with_document_ids(self, mock_doc_manager, mock_agraph_instance):
        """Test building knowledge graph with document IDs."""
        # Mock document manager
        mock_manager = MagicMock()
        mock_documents = [
            {"id": "doc1", "content": "Document 1 content", "filename": "doc1.txt", "metadata": {}},
            {"id": "doc2", "content": "Document 2 content", "filename": "doc2.txt", "metadata": {}},
        ]
        mock_manager.get_documents_by_ids.return_value = mock_documents
        mock_doc_manager.return_value = mock_manager

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_kg = MagicMock()
        mock_kg.name = "Test Graph"
        mock_kg.description = "Test Description"
        mock_kg.entities = {"e1": MagicMock()}
        mock_kg.relations = {}
        mock_kg.clusters = {}
        mock_kg.text_chunks = {"t1": MagicMock()}

        mock_agraph.build_from_texts.return_value = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test data
        test_data = {
            "document_ids": ["doc1", "doc2"],
            "graph_name": "Test Graph",
            "use_cache": True,
        }

        response = self.client.post("/knowledge-graph/build", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify document manager was called
        mock_manager.get_documents_by_ids.assert_called_once_with(["doc1", "doc2"])

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    @patch("agraph.api.routers.knowledge_graph.get_document_manager")
    def test_build_knowledge_graph_no_documents_found(self, mock_doc_manager, mock_agraph_instance):
        """Test building knowledge graph when no documents are found."""
        # Mock document manager returning empty list
        mock_manager = MagicMock()
        mock_manager.get_documents_by_ids.return_value = []
        mock_doc_manager.return_value = mock_manager

        test_data = {"document_ids": ["nonexistent"], "graph_name": "Test Graph"}

        response = self.client.post("/knowledge-graph/build", json=test_data)

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["message"], "No documents found with provided IDs")

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_update_knowledge_graph_no_existing_graph(self, mock_agraph_instance):
        """Test updating knowledge graph when no existing graph exists."""
        # Mock AGraph instance without knowledge graph
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = False
        mock_agraph_instance.return_value = mock_agraph

        test_data = {"additional_texts": ["New text content"]}

        response = self.client.post("/knowledge-graph/update", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(
            data["message"], "No existing knowledge graph found. Use /build endpoint first."
        )

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    @patch("agraph.api.routers.knowledge_graph.get_document_manager")
    def test_update_knowledge_graph_success(self, mock_doc_manager, mock_agraph_instance):
        """Test successful knowledge graph update."""
        # Mock existing knowledge graph
        mock_current_kg = MagicMock()
        mock_current_kg.name = "Existing Graph"
        mock_current_kg.description = "Existing Description"
        mock_current_kg.entities = {"e1": MagicMock()}
        mock_current_kg.relations = {}
        mock_current_kg.clusters = {}
        mock_current_kg.text_chunks = {"t1": MagicMock()}
        mock_current_kg.text_chunks["t1"].content = "Existing content"

        # Mock updated knowledge graph
        mock_updated_kg = MagicMock()
        mock_updated_kg.name = "Existing Graph"
        mock_updated_kg.description = "Existing Description"
        mock_updated_kg.entities = {"e1": MagicMock(), "e2": MagicMock()}
        mock_updated_kg.relations = {"r1": MagicMock()}
        mock_updated_kg.clusters = {}
        mock_updated_kg.text_chunks = {"t1": MagicMock(), "t2": MagicMock()}

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_current_kg
        mock_agraph.build_from_texts.return_value = mock_updated_kg
        mock_agraph_instance.return_value = mock_agraph

        test_data = {"additional_texts": ["New text content"], "use_cache": True}

        response = self.client.post("/knowledge-graph/update", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify changes are calculated correctly
        changes = data["data"]["changes"]
        self.assertEqual(changes["entities_added"], 1)  # 2 - 1
        self.assertEqual(changes["relations_added"], 1)  # 1 - 0

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_knowledge_graph_status_no_graph(self, mock_agraph_instance):
        """Test getting status when no knowledge graph exists."""
        # Mock AGraph instance without knowledge graph
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = False
        mock_agraph_instance.return_value = mock_agraph

        response = self.client.get("/knowledge-graph/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "No knowledge graph currently loaded")
        self.assertFalse(data["data"]["exists"])

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_knowledge_graph_status_with_graph(self, mock_agraph_instance):
        """Test getting status when knowledge graph exists."""
        # Mock entities and relations with types
        mock_entity = MagicMock()
        mock_entity.entity_type = MagicMock()
        mock_entity.entity_type.value = "PERSON"

        mock_relation = MagicMock()
        mock_relation.relation_type = MagicMock()
        mock_relation.relation_type.value = "WORKS_FOR"

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.name = "Test Graph"
        mock_kg.description = "Test Description"
        mock_kg.entities = {"e1": mock_entity}
        mock_kg.relations = {"r1": mock_relation}
        mock_kg.clusters = {}
        mock_kg.text_chunks = {"t1": MagicMock()}

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph.is_initialized = True
        mock_agraph.vector_store_type = "chroma"
        mock_agraph.enable_knowledge_graph = True
        mock_agraph_instance.return_value = mock_agraph

        response = self.client.get("/knowledge-graph/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertTrue(data["data"]["exists"])
        self.assertEqual(data["data"]["graph_name"], "Test Graph")
        self.assertEqual(data["data"]["statistics"]["entities"], 1)
        self.assertEqual(data["data"]["entity_types"]["PERSON"], 1)
        self.assertEqual(data["data"]["relation_types"]["WORKS_FOR"], 1)

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_knowledge_graph_no_graph(self, mock_agraph_instance):
        """Test getting knowledge graph when none exists."""
        # Mock AGraph instance without knowledge graph
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = False
        mock_agraph_instance.return_value = mock_agraph

        response = self.client.get("/knowledge-graph/get")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["message"], "No knowledge graph found. Please build one first.")

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_knowledge_graph_with_limits(self, mock_agraph_instance):
        """Test getting knowledge graph with entity and relation limits."""
        # Mock entities
        mock_entities = {}
        for i in range(5):
            mock_entity = MagicMock()
            mock_entity.id = f"e{i}"
            mock_entity.name = f"Entity {i}"
            mock_entity.entity_type.value = "PERSON"
            mock_entity.description = f"Description {i}"
            mock_entity.confidence = 0.8
            mock_entity.properties = {}
            mock_entity.aliases = []
            mock_entities[f"e{i}"] = mock_entity

        # Mock relations
        mock_relations = {}
        for i in range(3):
            mock_relation = MagicMock()
            mock_relation.id = f"r{i}"
            mock_relation.head_entity = mock_entities["e0"]
            mock_relation.tail_entity = (
                mock_entities[f"e{i+1}"] if i + 1 < 5 else mock_entities["e1"]
            )
            mock_relation.relation_type.value = "KNOWS"
            mock_relation.description = f"Relation {i}"
            mock_relation.confidence = 0.7
            mock_relation.properties = {}
            mock_relations[f"r{i}"] = mock_relation

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.name = "Test Graph"
        mock_kg.description = "Test Description"
        mock_kg.entities = mock_entities
        mock_kg.relations = mock_relations
        mock_kg.clusters = {}
        mock_kg.text_chunks = {}

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test with limits
        response = self.client.get("/knowledge-graph/get?entity_limit=2&relation_limit=1")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify limits are applied
        self.assertEqual(len(data["data"]["entities"]), 2)
        self.assertEqual(len(data["data"]["relations"]), 1)

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_visualization_data_with_filters(self, mock_agraph_instance):
        """Test getting visualization data with filters."""
        # Mock entities with different types and confidence
        mock_entities = {}
        entity_types = ["PERSON", "ORGANIZATION", "LOCATION"]
        confidences = [0.9, 0.5, 0.8]

        for i, (entity_type, confidence) in enumerate(zip(entity_types, confidences)):
            mock_entity = MagicMock()
            mock_entity.id = f"e{i}"
            mock_entity.name = f"Entity {i}"
            mock_entity.entity_type.value = entity_type
            mock_entity.confidence = confidence
            mock_entity.properties = {}
            mock_entities[f"e{i}"] = mock_entity

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.entities = mock_entities
        mock_kg.relations = {}
        mock_kg.clusters = {}
        mock_kg.text_chunks = {}

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test with filters
        test_data = {
            "entity_types": ["PERSON", "ORGANIZATION"],
            "min_confidence": 0.7,
            "max_entities": 10,
            "max_relations": 10,
        }

        response = self.client.post("/knowledge-graph/visualization-data", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should only include PERSON (0.9) and ORGANIZATION (0.8) entities
        # LOCATION (0.5) should be filtered out by min_confidence
        # But ORGANIZATION (0.5) should be filtered out, only PERSON (0.9) should remain
        nodes = data["data"]["nodes"]
        self.assertEqual(len(nodes), 2)  # PERSON (0.9) and LOCATION (0.8)

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_entities_with_pagination(self, mock_agraph_instance):
        """Test getting entities with pagination."""
        # Mock entities
        mock_entities = {}
        for i in range(10):
            mock_entity = MagicMock()
            mock_entity.id = f"e{i}"
            mock_entity.name = f"Entity {i}"
            mock_entity.entity_type.value = "PERSON"
            mock_entity.description = f"Description {i}"
            mock_entity.confidence = 0.8
            mock_entity.properties = {}
            mock_entity.aliases = []
            mock_entities[f"e{i}"] = mock_entity

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.entities = mock_entities
        mock_kg.relations = {}
        mock_kg.clusters = {}
        mock_kg.text_chunks = {}

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test pagination
        response = self.client.get("/knowledge-graph/entities?limit=5&offset=2")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

        # Verify pagination
        entities = data["data"]["entities"]
        self.assertEqual(len(entities), 5)
        pagination = data["data"]["pagination"]
        self.assertEqual(pagination["total"], 10)
        self.assertEqual(pagination["limit"], 5)
        self.assertEqual(pagination["offset"], 2)
        self.assertTrue(pagination["has_more"])

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_relations_with_entity_filter(self, mock_agraph_instance):
        """Test getting relations filtered by entity ID."""
        # Mock entities
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"
        mock_entity3 = MagicMock()
        mock_entity3.id = "e3"

        # Mock relations
        mock_relations = {}
        mock_relation1 = MagicMock()
        mock_relation1.id = "r1"
        mock_relation1.head_entity = mock_entity1
        mock_relation1.tail_entity = mock_entity2
        mock_relation1.relation_type.value = "KNOWS"
        mock_relation1.description = "Relation 1"
        mock_relation1.confidence = 0.8
        mock_relation1.properties = {}
        mock_relations["r1"] = mock_relation1

        mock_relation2 = MagicMock()
        mock_relation2.id = "r2"
        mock_relation2.head_entity = mock_entity2
        mock_relation2.tail_entity = mock_entity3
        mock_relation2.relation_type.value = "WORKS_FOR"
        mock_relation2.description = "Relation 2"
        mock_relation2.confidence = 0.9
        mock_relation2.properties = {}
        mock_relations["r2"] = mock_relation2

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.entities = {"e1": mock_entity1, "e2": mock_entity2, "e3": mock_entity3}
        mock_kg.relations = mock_relations
        mock_kg.clusters = {}
        mock_kg.text_chunks = {}

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test filtering by entity ID
        response = self.client.get("/knowledge-graph/relations?entity_id=e1")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

        # Should only return relations involving e1
        relations = data["data"]["relations"]
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["id"], "r1")

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_get_text_chunks_endpoint(self, mock_agraph_instance):
        """Test getting text chunks endpoint."""
        # Mock text chunks
        mock_chunks = {}
        for i in range(5):
            mock_chunk = MagicMock()
            mock_chunk.id = f"t{i}"
            mock_chunk.content = f"Text chunk content {i}"
            mock_chunk.source = f"source{i}.txt"
            mock_chunk.start_index = i * 100
            mock_chunk.end_index = (i + 1) * 100
            mock_chunk.entities = {f"e{i}"}
            mock_chunks[f"t{i}"] = mock_chunk

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.text_chunks = mock_chunks

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        response = self.client.get("/knowledge-graph/text-chunks?limit=3")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

        # Verify response structure
        text_chunks = data["data"]["text_chunks"]
        self.assertEqual(len(text_chunks), 3)
        pagination = data["data"]["pagination"]
        self.assertEqual(pagination["total"], 5)
        self.assertTrue(pagination["has_more"])

    def test_extract_text_from_document_string_content(self):
        """Test text extraction from string content."""
        doc = {"content": "This is text content", "metadata": {}}

        result = _extract_text_from_document(doc)
        self.assertEqual(result, "This is text content")

    def test_extract_text_from_document_bytes_content(self):
        """Test text extraction from bytes content."""
        doc = {
            "content": b"This is bytes content",
            "metadata": {"original_format": True},
            "filename": "test.txt",
        }

        result = _extract_text_from_document(doc)
        self.assertEqual(result, "This is bytes content")

    def test_extract_text_from_document_invalid_bytes(self):
        """Test text extraction from invalid bytes content."""
        doc = {
            "content": b"\xff\xfe\xfd",  # Invalid UTF-8
            "metadata": {"original_format": True},
            "filename": "test.txt",
        }

        with self.assertRaises(ValueError) as context:
            _extract_text_from_document(doc)

        self.assertIn("Cannot extract text from binary content", str(context.exception))

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_search_text_chunks_with_search_query(self, mock_agraph_instance):
        """Test searching text chunks with search query."""
        # Mock text chunks
        mock_chunks = {}
        contents = ["Python programming", "Java development", "Machine learning"]

        for i, content in enumerate(contents):
            mock_chunk = MagicMock()
            mock_chunk.id = f"t{i}"
            mock_chunk.content = content
            mock_chunk.source = f"source{i}.txt"
            mock_chunk.start_index = i * 100
            mock_chunk.end_index = (i + 1) * 100
            mock_chunk.entities = {f"e{i}"}
            mock_chunks[f"t{i}"] = mock_chunk

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.text_chunks = mock_chunks

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.has_knowledge_graph = True
        mock_agraph.knowledge_graph = mock_kg
        mock_agraph_instance.return_value = mock_agraph

        # Test search
        test_data = {"search": "python", "limit": 10, "offset": 0}

        response = self.client.post("/knowledge-graph/text-chunks", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Should only return chunks containing "python" (case insensitive)
        text_chunks = data["data"]["text_chunks"]
        self.assertEqual(len(text_chunks), 1)
        self.assertIn("Python", text_chunks[0]["content"])

    @patch("agraph.api.routers.knowledge_graph.get_agraph_instance")
    def test_error_handling_in_build(self, mock_agraph_instance):
        """Test error handling in build endpoint."""
        # Mock AGraph instance that raises exception
        mock_agraph = AsyncMock()
        mock_agraph.build_from_texts.side_effect = Exception("Build failed")
        mock_agraph_instance.return_value = mock_agraph

        test_data = {"texts": ["Test text"], "graph_name": "Test Graph"}

        response = self.client.post("/knowledge-graph/build", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Build failed")

    def test_build_knowledge_graph_no_content(self):
        """Test building knowledge graph with no texts or document IDs."""
        test_data = {"graph_name": "Test Graph"}

        with (
            patch("agraph.api.routers.knowledge_graph.get_agraph_instance") as mock_agraph_instance,
            patch("agraph.api.routers.knowledge_graph.get_document_manager") as mock_doc_manager,
        ):

            # Mock empty document storage
            mock_manager = MagicMock()
            mock_manager.list_documents.return_value = ([], 0)
            mock_doc_manager.return_value = mock_manager

            response = self.client.post("/knowledge-graph/build", json=test_data)

            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn("No documents found", data["message"])


class TestTextExtraction(unittest.TestCase):
    """Test text extraction utility function."""

    def test_extract_text_string_content(self):
        """Test extracting text from string content."""
        doc = {"content": "Hello world", "metadata": {}}
        result = _extract_text_from_document(doc)
        self.assertEqual(result, "Hello world")

    def test_extract_text_bytes_utf8(self):
        """Test extracting text from UTF-8 bytes."""
        doc = {
            "content": "Hello world".encode("utf-8"),
            "metadata": {"original_format": True},
            "filename": "test.txt",
        }
        result = _extract_text_from_document(doc)
        self.assertEqual(result, "Hello world")

    def test_extract_text_non_utf8_bytes_fails(self):
        """Test that non-UTF-8 bytes raise ValueError."""
        doc = {
            "content": b"\xff\xfe\xfd",
            "metadata": {"original_format": True},
            "filename": "test.bin",
        }

        with self.assertRaises(ValueError) as context:
            _extract_text_from_document(doc)

        self.assertIn("Cannot extract text from binary content", str(context.exception))

    def test_extract_text_other_types(self):
        """Test extracting text from other content types."""
        doc = {"content": 12345, "metadata": {}}
        result = _extract_text_from_document(doc)
        self.assertEqual(result, "12345")

    @patch("agraph.api.routers.knowledge_graph.DocumentProcessorManager")
    @patch("agraph.api.routers.knowledge_graph.os.unlink")
    def test_extract_text_binary_with_processor(self, mock_unlink, mock_processor_class):
        """Test extracting text from binary content using document processor."""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.can_process.return_value = True
        mock_processor.process.return_value = "Processed text content"
        mock_processor_class.return_value = mock_processor

        doc = {
            "content": b"Binary PDF content",
            "metadata": {"original_format": True},
            "filename": "test.pdf",
        }

        result = _extract_text_from_document(doc)
        self.assertEqual(result, "Processed text content")

        # Verify processor was used
        mock_processor.can_process.assert_called_once()
        mock_processor.process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
