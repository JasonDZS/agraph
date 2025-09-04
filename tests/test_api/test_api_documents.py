"""Tests for Document Management API router."""

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestDocumentsRouter(unittest.TestCase):
    """Test Document Management API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.documents.get_document_manager")
    @patch("agraph.api.routers.documents.DocumentProcessorManager")
    def test_upload_documents_success(self, mock_processor_class, mock_get_doc_manager):
        """Test successful document upload."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_doc_manager.store_document.return_value = "doc_123"
        mock_get_doc_manager.return_value = mock_doc_manager

        # Mock document processor
        mock_processor = MagicMock()
        mock_processor.can_process.return_value = True
        mock_processor.extract_metadata.return_value = {"pages": 1}
        mock_processor_class.return_value = mock_processor

        # Create test file
        test_content = b"Test document content"
        test_file = BytesIO(test_content)

        # Test upload
        response = self.client.post(
            "/documents/upload",
            files={"files": ("test.txt", test_file, "text/plain")},
            data={
                "metadata": json.dumps({"author": "test"}),
                "tags": json.dumps(["test", "document"]),
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Successfully uploaded", data["message"])

        # Verify response data
        response_data = data["data"]
        self.assertEqual(response_data["total_uploaded"], 1)
        self.assertEqual(len(response_data["uploaded_documents"]), 1)

        uploaded_doc = response_data["uploaded_documents"][0]
        self.assertEqual(uploaded_doc["id"], "doc_123")
        self.assertEqual(uploaded_doc["filename"], "test.txt")
        self.assertEqual(uploaded_doc["size"], len(test_content))

    @patch("agraph.api.routers.documents.get_document_manager")
    @patch("agraph.api.routers.documents.DocumentProcessorManager")
    def test_upload_documents_unsupported_file_type(
        self, mock_processor_class, mock_get_doc_manager
    ):
        """Test upload with unsupported file type."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_get_doc_manager.return_value = mock_doc_manager

        # Mock document processor that can't process the file
        mock_processor = MagicMock()
        mock_processor.can_process.return_value = False
        mock_processor.get_supported_extensions.return_value = [".txt", ".pdf", ".docx"]
        mock_processor_class.return_value = mock_processor

        # Create test file with unsupported extension
        test_content = b"Binary content"
        test_file = BytesIO(test_content)

        response = self.client.post(
            "/documents/upload",
            files={"files": ("test.xyz", test_file, "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("not supported", data["message"])
        self.assertIn("Supported extensions", data["message"])

    def test_upload_documents_invalid_metadata(self):
        """Test upload with invalid JSON metadata."""
        test_content = b"Test document content"
        test_file = BytesIO(test_content)

        response = self.client.post(
            "/documents/upload",
            files={"files": ("test.txt", test_file, "text/plain")},
            data={"metadata": "invalid json"},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Invalid JSON", data["message"])

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_upload_texts_success(self, mock_get_doc_manager):
        """Test successful text upload."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_doc_manager.store_document.side_effect = ["doc_1", "doc_2"]
        mock_get_doc_manager.return_value = mock_doc_manager

        test_data = {
            "texts": ["First text content", "Second text content"],
            "metadata": {"source": "test"},
            "tags": ["test", "upload"],
        }

        response = self.client.post("/documents/from-text", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Successfully uploaded 2 text documents", data["message"])

        # Verify response data
        response_data = data["data"]
        self.assertEqual(response_data["total_uploaded"], 2)
        self.assertEqual(len(response_data["uploaded_documents"]), 2)

    def test_upload_texts_no_content(self):
        """Test text upload with no texts provided."""
        test_data = {"texts": []}

        response = self.client.post("/documents/from-text", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["message"], "No texts provided")

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_list_documents_success(self, mock_get_doc_manager):
        """Test successful document listing."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_documents = [
            {"id": "doc1", "filename": "test1.txt", "tags": ["test"]},
            {"id": "doc2", "filename": "test2.txt", "tags": ["test", "example"]},
        ]
        mock_doc_manager.list_documents.return_value = (mock_documents, 2)
        mock_get_doc_manager.return_value = mock_doc_manager

        response = self.client.get("/documents/list?page=1&page_size=10")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify response data
        response_data = data["data"]
        self.assertEqual(len(response_data["documents"]), 2)
        self.assertEqual(response_data["pagination"]["total_count"], 2)
        self.assertEqual(response_data["pagination"]["page"], 1)
        self.assertEqual(response_data["pagination"]["page_size"], 10)

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_list_documents_with_filters(self, mock_get_doc_manager):
        """Test document listing with tag filter and search query."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_documents = [{"id": "doc1", "filename": "test1.txt"}]
        mock_doc_manager.list_documents.return_value = (mock_documents, 1)
        mock_get_doc_manager.return_value = mock_doc_manager

        tag_filter = json.dumps(["test"])
        search_query = "test"

        response = self.client.get(
            f"/documents/list?tag_filter={tag_filter}&search_query={search_query}"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify filters were passed correctly
        mock_doc_manager.list_documents.assert_called_once()
        call_args = mock_doc_manager.list_documents.call_args[1]
        self.assertEqual(call_args["tag_filter"], ["test"])
        self.assertEqual(call_args["search_query"], "test")

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_get_document_success(self, mock_get_doc_manager):
        """Test successful document retrieval."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_document = {
            "id": "doc_123",
            "filename": "test.txt",
            "content": "Test content",
            "metadata": {"author": "test"},
        }
        mock_doc_manager.get_document.return_value = mock_document
        mock_get_doc_manager.return_value = mock_doc_manager

        response = self.client.get("/documents/doc_123")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Document retrieved successfully")
        self.assertEqual(data["data"], mock_document)

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_get_document_not_found(self, mock_get_doc_manager):
        """Test document retrieval when document doesn't exist."""
        # Mock document manager returning None
        mock_doc_manager = MagicMock()
        mock_doc_manager.get_document.return_value = None
        mock_get_doc_manager.return_value = mock_doc_manager

        response = self.client.get("/documents/nonexistent")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("Document not found", data["message"])

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_delete_documents_success(self, mock_get_doc_manager):
        """Test successful document deletion."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_doc_manager.delete_documents.return_value = {"doc1": True, "doc2": True, "doc3": False}
        mock_get_doc_manager.return_value = mock_doc_manager

        test_data = {"document_ids": ["doc1", "doc2", "doc3"]}

        response = self.client.post("/documents/delete", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify response data
        response_data = data["data"]
        self.assertEqual(len(response_data["deleted_documents"]), 2)
        self.assertEqual(len(response_data["failed_deletes"]), 1)
        self.assertEqual(response_data["total_requested"], 3)
        self.assertEqual(response_data["total_deleted"], 2)

        # Verify message includes failure info
        self.assertIn("failed to delete 1 documents", data["message"])

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_get_document_stats_success(self, mock_get_doc_manager):
        """Test successful document statistics retrieval."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_stats = {
            "total_documents": 10,
            "total_size": 1024000,
            "document_types": {"txt": 5, "pdf": 3, "docx": 2},
            "tags": {"test": 8, "example": 5},
        }
        mock_doc_manager.get_stats.return_value = mock_stats
        mock_get_doc_manager.return_value = mock_doc_manager

        response = self.client.get("/documents/stats/summary")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Document statistics retrieved successfully")
        self.assertEqual(data["data"], mock_stats)

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_project_specific_document_operations(self, mock_get_doc_manager):
        """Test that project-specific operations use correct document manager."""
        # Mock different document managers for different projects
        mock_default_manager = MagicMock()
        mock_project_manager = MagicMock()

        def mock_get_manager_side_effect(project_name=None):
            if project_name == "test_project":
                return mock_project_manager
            return mock_default_manager

        mock_get_doc_manager.side_effect = mock_get_manager_side_effect

        # Mock stats for project manager
        mock_project_manager.get_stats.return_value = {"total_documents": 5}

        # Test with project name
        response = self.client.get("/documents/stats/summary?project_name=test_project")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["data"]["total_documents"], 5)

        # Verify project manager was called
        mock_project_manager.get_stats.assert_called_once()

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_document_manager_error_handling(self, mock_get_doc_manager):
        """Test error handling when document manager fails."""
        # Mock document manager that raises exception
        mock_doc_manager = MagicMock()
        mock_doc_manager.get_stats.side_effect = Exception("Database error")
        mock_get_doc_manager.return_value = mock_doc_manager

        response = self.client.get("/documents/stats/summary")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Database error")

    @patch("agraph.api.routers.documents.get_document_manager")
    @patch("agraph.api.routers.documents.DocumentProcessorManager")
    def test_upload_multiple_files(self, mock_processor_class, mock_get_doc_manager):
        """Test uploading multiple files at once."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_doc_manager.store_document.side_effect = ["doc_1", "doc_2", "doc_3"]
        mock_get_doc_manager.return_value = mock_doc_manager

        # Mock document processor
        mock_processor = MagicMock()
        mock_processor.can_process.return_value = True
        mock_processor.extract_metadata.return_value = {"type": "text"}
        mock_processor_class.return_value = mock_processor

        # Create multiple test files
        files = [
            ("files", ("file1.txt", BytesIO(b"Content 1"), "text/plain")),
            ("files", ("file2.txt", BytesIO(b"Content 2"), "text/plain")),
            ("files", ("file3.txt", BytesIO(b"Content 3"), "text/plain")),
        ]

        response = self.client.post("/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify all files were uploaded
        response_data = data["data"]
        self.assertEqual(response_data["total_uploaded"], 3)
        self.assertEqual(len(response_data["uploaded_documents"]), 3)

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_upload_file_without_filename(self, mock_get_doc_manager):
        """Test uploading file without filename."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_get_doc_manager.return_value = mock_doc_manager

        # Create file without filename (empty string)
        test_file = BytesIO(b"Content")

        response = self.client.post(
            "/documents/upload", files={"files": ("", test_file, "text/plain")}
        )

        # Should still return 200 but with no uploaded documents
        self.assertEqual(response.status_code, 200)
        data = response.json()
        response_data = data["data"]
        self.assertEqual(response_data["total_uploaded"], 0)

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_list_documents_with_pagination(self, mock_get_doc_manager):
        """Test document listing with pagination parameters."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_documents = [{"id": f"doc{i}", "filename": f"file{i}.txt"} for i in range(5)]
        mock_doc_manager.list_documents.return_value = (mock_documents, 25)  # 25 total documents
        mock_get_doc_manager.return_value = mock_doc_manager

        response = self.client.get("/documents/list?page=2&page_size=5")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify pagination calculation
        pagination = data["data"]["pagination"]
        self.assertEqual(pagination["page"], 2)
        self.assertEqual(pagination["page_size"], 5)
        self.assertEqual(pagination["total_count"], 25)
        self.assertEqual(pagination["total_pages"], 5)  # 25 / 5 = 5

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_list_documents_invalid_tag_filter(self, mock_get_doc_manager):
        """Test document listing with invalid tag filter JSON."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_doc_manager.list_documents.return_value = ([], 0)
        mock_get_doc_manager.return_value = mock_doc_manager

        # Invalid JSON should be handled gracefully
        response = self.client.get("/documents/list?tag_filter=invalid_json")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Should work with empty tag filter
        mock_doc_manager.list_documents.assert_called_once()
        call_args = mock_doc_manager.list_documents.call_args[1]
        self.assertIsNone(call_args["tag_filter"])

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_delete_documents_partial_success(self, mock_get_doc_manager):
        """Test document deletion with partial success."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_doc_manager.delete_documents.return_value = {
            "doc1": True,
            "doc2": False,  # Failed to delete
            "doc3": True,
        }
        mock_get_doc_manager.return_value = mock_doc_manager

        test_data = {"document_ids": ["doc1", "doc2", "doc3"]}

        response = self.client.post("/documents/delete", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify partial success is handled correctly
        response_data = data["data"]
        self.assertEqual(len(response_data["deleted_documents"]), 2)
        self.assertEqual(len(response_data["failed_deletes"]), 1)
        self.assertIn("doc2", response_data["failed_deletes"])

    @patch("agraph.api.routers.documents.get_document_manager")
    def test_document_operations_with_project_name(self, mock_get_doc_manager):
        """Test document operations with explicit project name."""
        # Mock different behavior for different projects
        mock_default_manager = MagicMock()
        mock_project_manager = MagicMock()

        def side_effect(project_name=None):
            if project_name == "my_project":
                return mock_project_manager
            return mock_default_manager

        mock_get_doc_manager.side_effect = side_effect
        mock_project_manager.get_stats.return_value = {"project_docs": 42}

        # Test with project name parameter
        response = self.client.get("/documents/stats/summary?project_name=my_project")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["data"]["project_docs"], 42)

    def test_upload_texts_with_project_name(self):
        """Test text upload with project name parameter."""
        with patch("agraph.api.routers.documents.get_document_manager") as mock_get_doc_manager:
            # Mock project-specific document manager
            mock_project_manager = MagicMock()
            mock_project_manager.store_document.return_value = "project_doc_1"

            def side_effect(project_name=None):
                if project_name == "test_project":
                    return mock_project_manager
                return MagicMock()

            mock_get_doc_manager.side_effect = side_effect

            test_data = {
                "texts": ["Project specific text"],
                "metadata": {"project": "test_project"},
            }

            response = self.client.post(
                "/documents/from-text?project_name=test_project", json=test_data
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)

            # Verify project manager was used
            mock_project_manager.store_document.assert_called_once()

    @patch("agraph.api.routers.documents.get_document_manager")
    @patch("agraph.api.routers.documents.DocumentProcessorManager")
    def test_upload_documents_processor_error(self, mock_processor_class, mock_get_doc_manager):
        """Test upload when document processor fails."""
        # Mock document manager
        mock_doc_manager = MagicMock()
        mock_get_doc_manager.return_value = mock_doc_manager

        # Mock document processor that raises exception
        mock_processor = MagicMock()
        mock_processor.can_process.return_value = True
        mock_processor.extract_metadata.side_effect = Exception("Processor failed")
        mock_processor_class.return_value = mock_processor

        test_content = b"Test content"
        test_file = BytesIO(test_content)

        response = self.client.post(
            "/documents/upload", files={"files": ("test.txt", test_file, "text/plain")}
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Failed to process file", data["message"])


if __name__ == "__main__":
    unittest.main()
