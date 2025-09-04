"""Tests for System Management API router."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestSystemRouter(unittest.TestCase):
    """Test System Management API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_stats_success(self, mock_get_agraph):
        """Test successful system statistics retrieval."""
        # Mock statistics data
        mock_stats = {
            "system": {
                "memory_usage": {"used": 512, "total": 1024, "percentage": 50.0},
                "cpu_usage": 25.5,
                "disk_usage": {"used": 2048, "total": 8192, "percentage": 25.0},
            },
            "knowledge_graph": {
                "entities_count": 150,
                "relations_count": 89,
                "clusters_count": 12,
                "text_chunks_count": 245,
            },
            "cache": {"hits": 1250, "misses": 89, "hit_rate": 93.4, "size": 45},
            "vector_store": {"collections": 3, "total_documents": 245, "index_size_mb": 128},
        }

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.get_stats.return_value = mock_stats
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/stats")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Statistics retrieved successfully")
        self.assertEqual(data["data"], mock_stats)

        # Verify get_stats was called
        mock_agraph.get_stats.assert_called_once()

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_stats_agraph_error(self, mock_get_agraph):
        """Test get stats when AGraph raises exception."""
        # Mock AGraph instance that raises exception
        mock_agraph = AsyncMock()
        mock_agraph.get_stats.side_effect = Exception("Stats collection failed")
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/stats")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Stats collection failed")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_build_status_with_builder(self, mock_get_agraph):
        """Test getting build status when builder is available."""
        # Mock build status data
        mock_build_status = {
            "current_step": "entity_extraction",
            "total_steps": 5,
            "completed_steps": 2,
            "progress_percentage": 40.0,
            "estimated_remaining_time": 120,
            "status": "in_progress",
            "started_at": "2024-01-01T10:00:00",
            "current_document": "document_3.pdf",
        }

        # Mock builder
        mock_builder = MagicMock()
        mock_builder.get_build_status.return_value = mock_build_status

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/build-status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Build status retrieved successfully")
        self.assertEqual(data["data"], mock_build_status)

        # Verify builder method was called
        mock_builder.get_build_status.assert_called_once()

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_build_status_no_builder(self, mock_get_agraph):
        """Test getting build status when no builder is initialized."""
        # Mock AGraph instance without builder
        mock_agraph = MagicMock()
        mock_agraph.builder = None
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/build-status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Builder not initialized")
        self.assertEqual(data["data"]["status"], "not_initialized")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_cache_info_with_builder(self, mock_get_agraph):
        """Test getting cache info when builder is available."""
        # Mock cache info data
        mock_cache_info = {
            "cache_type": "LRU+TTL",
            "max_size": 1000,
            "current_size": 456,
            "hit_rate": 89.5,
            "total_hits": 2340,
            "total_misses": 267,
            "cache_utilization": 45.6,
            "ttl_seconds": 3600,
            "oldest_entry_age": 1234,
            "newest_entry_age": 5,
        }

        # Mock builder
        mock_builder = MagicMock()
        mock_builder.get_cache_info.return_value = mock_cache_info

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/cache-info")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Cache info retrieved successfully")
        self.assertEqual(data["data"], mock_cache_info)

        # Verify builder method was called
        mock_builder.get_cache_info.assert_called_once()

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_cache_info_no_builder(self, mock_get_agraph):
        """Test getting cache info when no builder is initialized."""
        # Mock AGraph instance without builder
        mock_agraph = MagicMock()
        mock_agraph.builder = None
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/cache-info")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Builder not initialized")
        self.assertEqual(data["data"]["cache"], "not_available")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_cache_with_builder(self, mock_get_agraph):
        """Test clearing cache when builder is available."""
        # Mock builder
        mock_builder = MagicMock()
        mock_builder.clear_cache = MagicMock()

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        test_data = {"from_step": "entity_extraction"}

        response = self.client.post("/system/clear-cache", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Cache cleared successfully")

        # Verify clear_cache was called with correct parameter
        mock_builder.clear_cache.assert_called_once_with(from_step="entity_extraction")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_cache_no_from_step(self, mock_get_agraph):
        """Test clearing cache without from_step parameter."""
        # Mock builder
        mock_builder = MagicMock()
        mock_builder.clear_cache = MagicMock()

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        test_data = {}  # No from_step provided

        response = self.client.post("/system/clear-cache", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify clear_cache was called with None
        mock_builder.clear_cache.assert_called_once_with(from_step=None)

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_cache_no_builder(self, mock_get_agraph):
        """Test clearing cache when no builder is initialized."""
        # Mock AGraph instance without builder
        mock_agraph = MagicMock()
        mock_agraph.builder = None
        mock_get_agraph.return_value = mock_agraph

        test_data = {"from_step": "relation_extraction"}

        response = self.client.post("/system/clear-cache", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Builder not initialized, no cache to clear")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_all_data_success(self, mock_get_agraph):
        """Test successful clearing of all data."""
        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.clear_all.return_value = True
        mock_get_agraph.return_value = mock_agraph

        response = self.client.post("/system/clear-all")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "All data cleared successfully")

        # Verify clear_all was called
        mock_agraph.clear_all.assert_called_once()

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_all_data_failure(self, mock_get_agraph):
        """Test clearing all data when operation fails."""
        # Mock AGraph instance that returns False
        mock_agraph = AsyncMock()
        mock_agraph.clear_all.return_value = False
        mock_get_agraph.return_value = mock_agraph

        response = self.client.post("/system/clear-all")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.ERROR)
        self.assertEqual(data["message"], "Failed to clear all data")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_all_data_exception(self, mock_get_agraph):
        """Test clearing all data when exception is raised."""
        # Mock AGraph instance that raises exception
        mock_agraph = AsyncMock()
        mock_agraph.clear_all.side_effect = Exception("Clear operation failed")
        mock_get_agraph.return_value = mock_agraph

        response = self.client.post("/system/clear-all")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Clear operation failed")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_system_endpoints_dependency_error(self, mock_get_agraph):
        """Test system endpoints when dependency injection fails."""
        mock_get_agraph.side_effect = Exception("Failed to get AGraph instance")

        # Test all endpoints with dependency issues
        endpoints = [
            ("GET", "/system/stats"),
            ("GET", "/system/build-status"),
            ("GET", "/system/cache-info"),
            ("POST", "/system/clear-cache", {"from_step": "test"}),
            ("POST", "/system/clear-all", {}),
        ]

        for method, url, *json_data in endpoints:
            if method == "GET":
                response = self.client.get(url)
            else:  # POST
                response = self.client.post(url, json=json_data[0] if json_data else {})

            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["message"], "Failed to get AGraph instance")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_build_status_builder_error(self, mock_get_agraph):
        """Test get build status when builder method raises exception."""
        # Mock builder that raises exception
        mock_builder = MagicMock()
        mock_builder.get_build_status.side_effect = Exception("Builder status error")

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/build-status")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Builder status error")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_get_cache_info_builder_error(self, mock_get_agraph):
        """Test get cache info when builder method raises exception."""
        # Mock builder that raises exception
        mock_builder = MagicMock()
        mock_builder.get_cache_info.side_effect = Exception("Cache info error")

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/cache-info")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Cache info error")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_cache_builder_error(self, mock_get_agraph):
        """Test clear cache when builder method raises exception."""
        # Mock builder that raises exception
        mock_builder = MagicMock()
        mock_builder.clear_cache.side_effect = Exception("Cache clear error")

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        test_data = {"from_step": "clustering"}

        response = self.client.post("/system/clear-cache", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Cache clear error")

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_clear_cache_different_from_steps(self, mock_get_agraph):
        """Test clearing cache with different from_step values."""
        # Mock builder
        mock_builder = MagicMock()
        mock_builder.clear_cache = MagicMock()

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        # Test different from_step values
        test_cases = [
            {"from_step": "text_processing"},
            {"from_step": "entity_extraction"},
            {"from_step": "relation_extraction"},
            {"from_step": "clustering"},
            {"from_step": None},
            {},  # No from_step provided
        ]

        for test_data in test_cases:
            mock_builder.reset_mock()

            response = self.client.post("/system/clear-cache", json=test_data)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)

            # Verify clear_cache was called with correct parameter
            expected_from_step = test_data.get("from_step", None)
            mock_builder.clear_cache.assert_called_once_with(from_step=expected_from_step)

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_system_stats_comprehensive_data(self, mock_get_agraph):
        """Test that system stats endpoint handles comprehensive data structure."""
        # Mock comprehensive stats
        comprehensive_stats = {
            "timestamp": "2024-01-01T12:00:00Z",
            "uptime_seconds": 3600,
            "version": "1.0.0",
            "system": {
                "platform": "darwin",
                "python_version": "3.9.0",
                "memory": {"rss": 128, "vms": 256},
                "cpu_count": 8,
            },
            "knowledge_graph": {
                "entities": {"total": 150, "by_type": {"PERSON": 50, "ORG": 100}},
                "relations": {"total": 89, "by_type": {"WORKS_FOR": 45, "KNOWS": 44}},
                "clusters": {"total": 12, "avg_size": 12.5},
                "text_chunks": {"total": 245, "avg_length": 512},
            },
            "performance": {
                "avg_query_time_ms": 45.2,
                "avg_build_time_s": 120.5,
                "cache_hit_rate": 0.934,
            },
            "storage": {"vector_db_size_mb": 128, "cache_size_mb": 64, "documents_size_mb": 512},
        }

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.get_stats.return_value = comprehensive_stats
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/system/stats")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify comprehensive data is returned correctly
        self.assertEqual(data["data"], comprehensive_stats)
        self.assertIn("knowledge_graph", data["data"])
        self.assertIn("performance", data["data"])
        self.assertIn("storage", data["data"])

    def test_clear_cache_request_validation(self):
        """Test ClearCacheRequest validation."""
        # Test various valid from_step values
        valid_steps = [
            "text_processing",
            "entity_extraction",
            "relation_extraction",
            "clustering",
            "vectorization",
            None,
        ]

        for step in valid_steps:
            with patch(
                "agraph.api.routers.system.get_agraph_instance_dependency"
            ) as mock_get_agraph:
                mock_builder = MagicMock()
                mock_agraph = MagicMock()
                mock_agraph.builder = mock_builder
                mock_get_agraph.return_value = mock_agraph

                test_data = {"from_step": step} if step is not None else {}

                response = self.client.post("/system/clear-cache", json=test_data)

                # All should be valid at API level
                self.assertEqual(response.status_code, 200)

    def test_system_endpoints_exist(self):
        """Test that all system endpoints are properly registered."""
        # Test that endpoints exist (don't return 404)
        endpoints = [
            ("GET", "/system/stats"),
            ("GET", "/system/build-status"),
            ("GET", "/system/cache-info"),
            ("POST", "/system/clear-cache"),
            ("POST", "/system/clear-all"),
        ]

        for method, url in endpoints:
            if method == "GET":
                response = self.client.get(url)
            else:  # POST
                response = self.client.post(url, json={})

            # Should not return 404 (endpoint exists)
            self.assertNotEqual(response.status_code, 404)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for system management functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_system_endpoints_openapi_documentation(self):
        """Test that system endpoints are documented in OpenAPI schema."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)

        schema = response.json()
        paths = schema["paths"]

        # Verify system endpoints are documented
        expected_endpoints = [
            "/system/stats",
            "/system/build-status",
            "/system/cache-info",
            "/system/clear-cache",
            "/system/clear-all",
        ]

        for endpoint in expected_endpoints:
            self.assertIn(endpoint, paths)

    @patch("agraph.api.routers.system.get_agraph_instance_dependency")
    def test_system_workflow_cache_management(self, mock_get_agraph):
        """Test typical cache management workflow."""
        # Mock builder with cache operations
        mock_builder = MagicMock()
        mock_cache_info = {"size": 100, "hit_rate": 0.85, "entries": 50}
        mock_builder.get_cache_info.return_value = mock_cache_info
        mock_builder.clear_cache = MagicMock()

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.builder = mock_builder
        mock_get_agraph.return_value = mock_agraph

        # Step 1: Get cache info
        response = self.client.get("/system/cache-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"], mock_cache_info)

        # Step 2: Clear cache
        response = self.client.post("/system/clear-cache", json={"from_step": "entity_extraction"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Cache cleared successfully")

        # Verify operations were called
        mock_builder.get_cache_info.assert_called_once()
        mock_builder.clear_cache.assert_called_once_with(from_step="entity_extraction")


if __name__ == "__main__":
    unittest.main()
