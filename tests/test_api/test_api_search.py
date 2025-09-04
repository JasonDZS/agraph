"""Tests for Search API router."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestSearchRouter(unittest.TestCase):
    """Test Search API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_entities_success(self, mock_get_agraph):
        """Test successful entity search."""
        # Mock entities
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity1.name = "John Smith"
        mock_entity1.entity_type.value = "PERSON"
        mock_entity1.description = "Software engineer"
        mock_entity1.confidence = 0.9

        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"
        mock_entity2.name = "Microsoft"
        mock_entity2.entity_type.value = "ORGANIZATION"
        mock_entity2.description = "Technology company"
        mock_entity2.confidence = 0.8

        # Mock search results (tuples of entity and score)
        mock_search_results = [(mock_entity1, 0.95), (mock_entity2, 0.87)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "query": "software engineer",
            "search_type": "entities",
            "top_k": 10,
            "filter_dict": {"entity_type": "PERSON"},
        }

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Search completed successfully", data["message"])

        # Verify response structure
        response_data = data["data"]
        self.assertEqual(response_data["query"], "software engineer")
        self.assertEqual(response_data["search_type"], "entities")
        self.assertEqual(response_data["total_count"], 2)
        self.assertEqual(len(response_data["results"]), 2)

        # Verify entity data structure
        first_result = response_data["results"][0]
        self.assertIn("entity", first_result)
        self.assertIn("score", first_result)
        self.assertEqual(first_result["entity"]["id"], "e1")
        self.assertEqual(first_result["entity"]["name"], "John Smith")
        self.assertEqual(first_result["score"], 0.95)

        # Verify search method was called correctly
        mock_agraph.search_entities.assert_called_once_with(
            query="software engineer", top_k=10, filter_dict={"entity_type": "PERSON"}
        )

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_relations_success(self, mock_get_agraph):
        """Test successful relation search."""
        # Mock entities for relations
        mock_entity1 = MagicMock()
        mock_entity1.id = "e1"
        mock_entity2 = MagicMock()
        mock_entity2.id = "e2"

        # Mock relations
        mock_relation = MagicMock()
        mock_relation.id = "r1"
        mock_relation.head_entity = mock_entity1
        mock_relation.tail_entity = mock_entity2
        mock_relation.relation_type.value = "WORKS_FOR"
        mock_relation.description = "Employment relationship"
        mock_relation.confidence = 0.85

        # Mock search results
        mock_search_results = [(mock_relation, 0.92)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_relations.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "employment", "search_type": "relations", "top_k": 5}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify response structure
        response_data = data["data"]
        self.assertEqual(response_data["search_type"], "relations")
        self.assertEqual(len(response_data["results"]), 1)

        # Verify relation data structure
        first_result = response_data["results"][0]
        self.assertIn("relation", first_result)
        self.assertIn("score", first_result)
        self.assertEqual(first_result["relation"]["id"], "r1")
        self.assertEqual(first_result["relation"]["relation_type"], "WORKS_FOR")
        self.assertEqual(first_result["score"], 0.92)

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_text_chunks_success(self, mock_get_agraph):
        """Test successful text chunk search."""
        # Mock text chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.id = "t1"
        mock_chunk1.content = "This is a text chunk about Python programming"
        mock_chunk1.title = "Python Tutorial"
        mock_chunk1.source = "tutorial.txt"
        mock_chunk1.start_index = 0
        mock_chunk1.end_index = 100

        mock_chunk2 = MagicMock()
        mock_chunk2.id = "t2"
        mock_chunk2.content = "Advanced Python techniques and best practices"
        mock_chunk2.title = "Advanced Python"
        mock_chunk2.source = "advanced.txt"
        mock_chunk2.start_index = 100
        mock_chunk2.end_index = 200

        # Mock search results
        mock_search_results = [(mock_chunk1, 0.88), (mock_chunk2, 0.76)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_text_chunks.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "query": "Python programming",
            "search_type": "text_chunks",
            "top_k": 15,
            "filter_dict": {"source": "tutorial"},
        }

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify response structure
        response_data = data["data"]
        self.assertEqual(response_data["search_type"], "text_chunks")
        self.assertEqual(len(response_data["results"]), 2)

        # Verify text chunk data structure
        first_result = response_data["results"][0]
        self.assertIn("text_chunk", first_result)
        self.assertIn("score", first_result)
        self.assertEqual(first_result["text_chunk"]["id"], "t1")
        self.assertEqual(first_result["text_chunk"]["title"], "Python Tutorial")
        self.assertEqual(first_result["score"], 0.88)

    def test_search_invalid_search_type(self):
        """Test search with invalid search type."""
        test_data = {
            "query": "test query",
            "search_type": "invalid_type",  # Invalid search type
            "top_k": 10,
        }

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Invalid search_type", data["message"])
        self.assertIn("Must be 'entities', 'relations', or 'text_chunks'", data["message"])

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_entities_empty_results(self, mock_get_agraph):
        """Test entity search with no results."""
        # Mock AGraph instance returning empty results
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = []
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "nonexistent entity", "search_type": "entities", "top_k": 10}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify empty results are handled correctly
        response_data = data["data"]
        self.assertEqual(response_data["total_count"], 0)
        self.assertEqual(len(response_data["results"]), 0)

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_agraph_error(self, mock_get_agraph):
        """Test search when AGraph raises exception."""
        # Mock AGraph instance that raises exception
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.side_effect = Exception("Search index error")
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "test query", "search_type": "entities", "top_k": 10}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Search index error")

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_with_filter_dict(self, mock_get_agraph):
        """Test search with filter dictionary."""
        # Mock search results
        mock_entity = MagicMock()
        mock_entity.id = "e1"
        mock_entity.name = "Filtered Entity"
        mock_entity.entity_type.value = "PERSON"
        mock_entity.description = "Description"
        mock_entity.confidence = 0.9

        mock_search_results = [(mock_entity, 0.85)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "query": "person",
            "search_type": "entities",
            "top_k": 5,
            "filter_dict": {"entity_type": "PERSON", "confidence": {">=": 0.8}},
        }

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify filter_dict was passed correctly
        mock_agraph.search_entities.assert_called_once_with(
            query="person", top_k=5, filter_dict=test_data["filter_dict"]
        )

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_without_filter_dict(self, mock_get_agraph):
        """Test search without filter dictionary."""
        # Mock search results
        mock_entity = MagicMock()
        mock_entity.id = "e1"
        mock_entity.name = "Entity"
        mock_entity.entity_type.value = "PERSON"
        mock_entity.description = "Description"
        mock_entity.confidence = 0.9

        mock_search_results = [(mock_entity, 0.85)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "query": "entity search",
            "search_type": "entities",
            # No filter_dict provided, top_k should use default
        }

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify search was called with defaults
        mock_agraph.search_entities.assert_called_once_with(
            query="entity search", top_k=10, filter_dict=None  # Default value
        )

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_relations_with_missing_entities(self, mock_get_agraph):
        """Test relation search with relations missing head/tail entities."""
        # Mock relation with missing entities
        mock_relation = MagicMock()
        mock_relation.id = "r1"
        mock_relation.head_entity = None  # Missing head entity
        mock_relation.tail_entity = None  # Missing tail entity
        mock_relation.relation_type.value = "ORPHANED"
        mock_relation.description = "Orphaned relation"
        mock_relation.confidence = 0.7

        mock_search_results = [(mock_relation, 0.6)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_relations.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "orphaned", "search_type": "relations", "top_k": 5}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify relation with missing entities is handled correctly
        first_result = data["data"]["results"][0]
        relation = first_result["relation"]
        self.assertIsNone(relation["head_entity_id"])
        self.assertIsNone(relation["tail_entity_id"])
        self.assertEqual(relation["relation_type"], "ORPHANED")

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_text_chunks_complete_structure(self, mock_get_agraph):
        """Test text chunk search with complete data structure."""
        # Mock text chunk with all fields
        mock_chunk = MagicMock()
        mock_chunk.id = "t1"
        mock_chunk.content = "Complete text chunk content for testing search functionality"
        mock_chunk.title = "Search Test Document"
        mock_chunk.source = "test_document.pdf"
        mock_chunk.start_index = 150
        mock_chunk.end_index = 300

        mock_search_results = [(mock_chunk, 0.91)]

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_text_chunks.return_value = mock_search_results
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "search functionality", "search_type": "text_chunks", "top_k": 8}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify complete text chunk structure
        first_result = data["data"]["results"][0]
        text_chunk = first_result["text_chunk"]
        self.assertEqual(text_chunk["id"], "t1")
        self.assertEqual(text_chunk["content"], mock_chunk.content)
        self.assertEqual(text_chunk["title"], "Search Test Document")
        self.assertEqual(text_chunk["source"], "test_document.pdf")
        self.assertEqual(text_chunk["start_index"], 150)
        self.assertEqual(text_chunk["end_index"], 300)

    def test_search_request_validation(self):
        """Test search request validation."""
        # Missing required query field
        test_data = {"search_type": "entities", "top_k": 10}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_search_invalid_top_k_values(self):
        """Test search with various top_k values."""
        test_cases = [
            {"top_k": 0, "should_pass": True},  # Edge case: 0 results
            {"top_k": -1, "should_pass": True},  # Negative (API accepts, AGraph may handle)
            {"top_k": 1000, "should_pass": True},  # Very large number
        ]

        for case in test_cases:
            with patch(
                "agraph.api.routers.search.get_agraph_instance_dependency"
            ) as mock_get_agraph:
                mock_agraph = AsyncMock()
                mock_agraph.search_entities.return_value = []
                mock_get_agraph.return_value = mock_agraph

                test_data = {"query": "test", "search_type": "entities", "top_k": case["top_k"]}

                response = self.client.post("/search", json=test_data)

                if case["should_pass"]:
                    # API should accept and let AGraph handle validation
                    self.assertIn(response.status_code, [200, 500])  # 500 if AGraph rejects
                else:
                    self.assertEqual(response.status_code, 422)

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_empty_query(self, mock_get_agraph):
        """Test search with empty query string."""
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = []
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "", "search_type": "entities"}  # Empty query

        response = self.client.post("/search", json=test_data)

        # API should accept empty query and let AGraph handle it
        self.assertEqual(response.status_code, 200)

        # Verify empty query was passed
        mock_agraph.search_entities.assert_called_once()
        call_kwargs = mock_agraph.search_entities.call_args.kwargs
        self.assertEqual(call_kwargs["query"], "")

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_all_types_sequentially(self, mock_get_agraph):
        """Test searching all supported types."""
        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_get_agraph.return_value = mock_agraph

        search_types = ["entities", "relations", "text_chunks"]

        for search_type in search_types:
            # Reset mock for each iteration
            mock_agraph.reset_mock()

            # Set up appropriate mock method
            if search_type == "entities":
                mock_agraph.search_entities.return_value = []
            elif search_type == "relations":
                mock_agraph.search_relations.return_value = []
            else:  # text_chunks
                mock_agraph.search_text_chunks.return_value = []

            test_data = {"query": f"test {search_type}", "search_type": search_type, "top_k": 5}

            response = self.client.post("/search", json=test_data)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)
            self.assertEqual(data["data"]["search_type"], search_type)

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_complex_filter_dict(self, mock_get_agraph):
        """Test search with complex filter dictionary."""
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = []
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "query": "complex search",
            "search_type": "entities",
            "top_k": 10,
            "filter_dict": {
                "entity_type": {"in": ["PERSON", "ORGANIZATION"]},
                "confidence": {">=": 0.7, "<=": 0.9},
                "properties": {"location": "California", "active": True},
            },
        }

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify complex filter was passed correctly
        call_kwargs = mock_agraph.search_entities.call_args.kwargs
        self.assertEqual(call_kwargs["filter_dict"], test_data["filter_dict"])

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_dependency_injection_error(self, mock_get_agraph):
        """Test search when dependency injection fails."""
        mock_get_agraph.side_effect = Exception("Failed to get AGraph instance")

        test_data = {"query": "test", "search_type": "entities"}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Failed to get AGraph instance")

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_results_score_ordering(self, mock_get_agraph):
        """Test that search results maintain score ordering."""
        # Mock entities with different scores
        entities_with_scores = []
        for i, score in enumerate([0.95, 0.88, 0.76, 0.65]):
            mock_entity = MagicMock()
            mock_entity.id = f"e{i}"
            mock_entity.name = f"Entity {i}"
            mock_entity.entity_type.value = "PERSON"
            mock_entity.description = f"Description {i}"
            mock_entity.confidence = 0.8
            entities_with_scores.append((mock_entity, score))

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = entities_with_scores
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "ordered search", "search_type": "entities", "top_k": 10}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify results maintain score ordering
        results = data["data"]["results"]
        self.assertEqual(len(results), 4)

        # Check that scores are in the expected order
        scores = [result["score"] for result in results]
        self.assertEqual(scores, [0.95, 0.88, 0.76, 0.65])

    def test_search_model_validation_edge_cases(self):
        """Test SearchRequest model validation with edge cases."""
        # Test with all valid combinations
        valid_test_cases = [
            {"query": "a", "search_type": "entities"},  # Single character
            {
                "query": "very " * 100 + "long query",  # Very long query
                "search_type": "text_chunks",
                "top_k": 1,
            },
            {
                "query": "Unicode test: 测试中文",  # Unicode characters
                "search_type": "relations",
                "top_k": 50,
            },
        ]

        for test_data in valid_test_cases:
            with patch(
                "agraph.api.routers.search.get_agraph_instance_dependency"
            ) as mock_get_agraph:
                mock_agraph = AsyncMock()
                # Set appropriate return value based on search type
                if test_data["search_type"] == "entities":
                    mock_agraph.search_entities.return_value = []
                elif test_data["search_type"] == "relations":
                    mock_agraph.search_relations.return_value = []
                else:
                    mock_agraph.search_text_chunks.return_value = []

                mock_get_agraph.return_value = mock_agraph

                response = self.client.post("/search", json=test_data)

                # All should be valid at API level
                self.assertEqual(response.status_code, 200)

    @patch("agraph.api.routers.search.get_agraph_instance_dependency")
    def test_search_none_filter_dict(self, mock_get_agraph):
        """Test search with explicit None filter_dict."""
        mock_agraph = AsyncMock()
        mock_agraph.search_entities.return_value = []
        mock_get_agraph.return_value = mock_agraph

        test_data = {"query": "test", "search_type": "entities", "filter_dict": None}

        response = self.client.post("/search", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify None filter_dict is passed correctly
        call_kwargs = mock_agraph.search_entities.call_args.kwargs
        self.assertIsNone(call_kwargs["filter_dict"])


class TestSearchIntegration(unittest.TestCase):
    """Integration tests for search functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_search_endpoint_exists(self):
        """Test that search endpoint is properly registered."""
        # Test with invalid data to ensure endpoint exists
        response = self.client.post("/search", json={})

        # Should not return 404 (endpoint exists), but may return validation error
        self.assertNotEqual(response.status_code, 404)

    def test_search_openapi_documentation(self):
        """Test that search endpoint is documented in OpenAPI schema."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)

        schema = response.json()
        # Verify search endpoint is in the schema
        self.assertIn("/search", schema["paths"])

        # Verify the endpoint has proper documentation
        search_endpoint = schema["paths"]["/search"]["post"]
        self.assertIn("summary", search_endpoint)
        self.assertIn("requestBody", search_endpoint)


if __name__ == "__main__":
    unittest.main()
