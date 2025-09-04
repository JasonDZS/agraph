"""Tests for Project Management API router."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestProjectsRouter(unittest.TestCase):
    """Test Project Management API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_current_project")
    def test_list_all_projects(self, mock_get_current, mock_list_projects):
        """Test listing all projects."""
        # Mock return values
        mock_list_projects.return_value = ["project1", "project2", "project3"]
        mock_get_current.return_value = "project1"

        response = self.client.get("/projects/list")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(len(data["projects"]), 3)
        self.assertEqual(data["data"]["current_project"], "project1")
        self.assertEqual(data["data"]["total_count"], 3)

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_current_project")
    def test_list_projects_empty(self, mock_get_current, mock_list_projects):
        """Test listing projects when no projects exist."""
        mock_list_projects.return_value = []
        mock_get_current.return_value = None

        response = self.client.get("/projects/list")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(len(data["projects"]), 0)
        self.assertIsNone(data["data"]["current_project"])

    @patch("agraph.api.routers.projects.create_project")
    def test_create_new_project_success(self, mock_create_project):
        """Test successful project creation."""
        # Mock return value
        mock_config = {
            "name": "test_project",
            "description": "Test project",
            "created_at": "2024-01-01T00:00:00",
        }
        mock_create_project.return_value = mock_config

        test_data = {"name": "test_project", "description": "Test project description"}

        response = self.client.post("/projects/create", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["project_name"], "test_project")
        self.assertEqual(data["data"], mock_config)
        self.assertIn("created successfully", data["message"])

        # Verify create_project was called with correct arguments
        mock_create_project.assert_called_once_with("test_project", "Test project description")

    @patch("agraph.api.routers.projects.create_project")
    def test_create_project_validation_error(self, mock_create_project):
        """Test project creation with validation error."""
        # Mock ValueError for invalid input
        mock_create_project.side_effect = ValueError("Invalid project name")

        test_data = {
            "name": "",  # Empty name should cause validation error
            "description": "Test description",
        }

        response = self.client.post("/projects/create", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["message"], "Invalid project name")

    @patch("agraph.api.routers.projects.get_current_project")
    def test_get_current_project_info_no_project(self, mock_get_current):
        """Test getting current project info when no project is set."""
        mock_get_current.return_value = None

        response = self.client.get("/projects/current")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("No current project", data["message"])
        self.assertIsNone(data["data"]["current_project"])
        self.assertTrue(data["data"]["using_default"])

    @patch("agraph.api.routers.projects.get_current_project")
    @patch("agraph.api.routers.projects.get_project_paths")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_get_current_project_info_with_config(
        self, mock_exists, mock_file, mock_get_paths, mock_get_current
    ):
        """Test getting current project info with existing config file."""
        # Mock return values
        mock_get_current.return_value = "test_project"
        mock_paths = {
            "project_dir": "/path/to/test_project",
            "config_file": "/path/to/test_project/config.json",
            "document_storage": "/path/to/test_project/documents",
            "vector_db": "/path/to/test_project/vectordb",
            "cache": "/path/to/test_project/cache",
        }
        mock_get_paths.return_value = mock_paths
        mock_exists.return_value = True

        # Mock config file content
        mock_config = {"name": "test_project", "description": "Test project", "version": "1.0"}
        mock_file.return_value.read.return_value = json.dumps(mock_config)

        response = self.client.get("/projects/current")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["project_name"], "test_project")
        self.assertEqual(data["data"]["current_project"], "test_project")
        self.assertEqual(data["data"]["paths"], mock_paths)
        self.assertEqual(data["data"]["name"], "test_project")

    @patch("agraph.api.routers.projects.reset_instances")
    @patch("agraph.api.routers.projects.set_current_project")
    @patch("agraph.api.routers.projects.get_current_project")
    def test_switch_project_success(self, mock_get_current, mock_set_current, mock_reset):
        """Test successful project switching."""
        # Mock settings object
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"current_project": "new_project"}
        mock_set_current.return_value = mock_settings
        mock_get_current.return_value = "old_project"

        test_data = {"project_name": "new_project"}

        response = self.client.post("/projects/switch", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["project_name"], "new_project")
        self.assertIn("Switched to project: new_project", data["message"])

        # Verify functions were called
        mock_reset.assert_called_once()
        mock_set_current.assert_called_once_with("new_project")

    @patch("agraph.api.routers.projects.reset_instances")
    @patch("agraph.api.routers.projects.set_current_project")
    def test_switch_to_default_workspace(self, mock_set_current, mock_reset):
        """Test switching to default workspace (no project)."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"current_project": None}
        mock_set_current.return_value = mock_settings

        test_data = {"project_name": None}

        response = self.client.post("/projects/switch", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIsNone(data["project_name"])
        self.assertIn("default workspace", data["message"])

    @patch("agraph.api.routers.projects.set_current_project")
    def test_switch_project_validation_error(self, mock_set_current):
        """Test project switching with validation error."""
        mock_set_current.side_effect = ValueError("Invalid project name")

        test_data = {"project_name": "invalid-project"}

        response = self.client.post("/projects/switch", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["message"], "Invalid project name")

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_project_paths")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_get_project_info_success(
        self, mock_exists, mock_file, mock_get_paths, mock_list_projects
    ):
        """Test getting specific project information."""
        # Mock return values
        mock_list_projects.return_value = ["test_project", "other_project"]
        mock_paths = {
            "project_dir": "/path/to/test_project",
            "config_file": "/path/to/test_project/config.json",
            "document_storage": "/path/to/test_project/documents",
            "vector_db": "/path/to/test_project/vectordb",
            "cache": "/path/to/test_project/cache",
        }
        mock_get_paths.return_value = mock_paths
        mock_exists.return_value = True

        # Mock config file
        mock_config = {"name": "test_project", "description": "Test"}
        mock_file.return_value.read.return_value = json.dumps(mock_config)

        # Mock Path objects and their methods
        with patch("pathlib.Path") as mock_path_class:
            # Mock directory existence and file operations
            mock_project_dir = MagicMock()
            mock_project_dir.exists.return_value = True
            mock_project_dir.rglob.return_value = []  # No files for size calculation

            mock_doc_dir = MagicMock()
            mock_doc_dir.exists.return_value = True
            mock_doc_dir.glob.return_value = []  # No documents

            mock_vector_dir = MagicMock()
            mock_vector_dir.exists.return_value = False

            mock_cache_dir = MagicMock()
            mock_cache_dir.exists.return_value = False

            def path_side_effect(path_str):
                if "config.json" in path_str:
                    return MagicMock()
                elif "documents" in path_str:
                    return mock_doc_dir
                elif "vectordb" in path_str:
                    return mock_vector_dir
                elif "cache" in path_str:
                    return mock_cache_dir
                else:
                    return mock_project_dir

            mock_path_class.side_effect = path_side_effect

            response = self.client.get("/projects/test_project")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)
            self.assertEqual(data["project_name"], "test_project")
            self.assertIn("statistics", data["data"])

    @patch("agraph.api.routers.projects.list_projects")
    def test_get_project_info_not_found(self, mock_list_projects):
        """Test getting project info for non-existent project."""
        mock_list_projects.return_value = ["existing_project"]

        response = self.client.get("/projects/nonexistent_project")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("not found", data["message"])

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_project_paths")
    @patch("agraph.api.routers.projects.delete_project")
    @patch("agraph.api.routers.projects.reset_instances")
    def test_delete_project_success(
        self, mock_reset, mock_delete, mock_get_paths, mock_list_projects
    ):
        """Test successful project deletion."""
        # Mock return values
        mock_list_projects.return_value = ["test_project", "other_project"]
        mock_paths = {
            "project_dir": "/path/to/test_project",
            "config_file": "/path/to/test_project/config.json",
        }
        mock_get_paths.return_value = mock_paths
        mock_delete.return_value = True

        test_data = {"project_name": "test_project", "confirm": True}

        response = self.client.post("/projects/delete", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["project_name"], "test_project")
        self.assertIn("deleted successfully", data["message"])
        self.assertEqual(data["data"]["deleted_paths"], mock_paths)

        # Verify functions were called
        mock_delete.assert_called_once_with("test_project")
        mock_reset.assert_called_once()

    @patch("agraph.api.routers.projects.list_projects")
    def test_delete_project_without_confirmation(self, mock_list_projects):
        """Test project deletion without confirmation."""
        mock_list_projects.return_value = ["test_project"]

        test_data = {"project_name": "test_project", "confirm": False}

        response = self.client.post("/projects/delete", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("requires confirmation", data["message"])

    @patch("agraph.api.routers.projects.list_projects")
    def test_delete_project_not_found(self, mock_list_projects):
        """Test deleting non-existent project."""
        mock_list_projects.return_value = ["existing_project"]

        test_data = {"project_name": "nonexistent", "confirm": True}

        response = self.client.post("/projects/delete", json=test_data)

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("not found", data["message"])

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_project_paths")
    @patch("agraph.api.routers.projects.delete_project")
    def test_delete_project_failure(self, mock_delete, mock_get_paths, mock_list_projects):
        """Test project deletion failure."""
        mock_list_projects.return_value = ["test_project"]
        mock_paths = {"project_dir": "/path/to/test_project"}
        mock_get_paths.return_value = mock_paths
        mock_delete.return_value = False  # Deletion failed

        test_data = {"project_name": "test_project", "confirm": True}

        response = self.client.post("/projects/delete", json=test_data)

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("not found or already deleted", data["message"])

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_current_project")
    def test_get_projects_overview_without_stats(self, mock_get_current, mock_list_projects):
        """Test getting projects overview without statistics."""
        mock_list_projects.return_value = ["proj1", "proj2"]
        mock_get_current.return_value = "proj1"

        response = self.client.get("/projects/?include_stats=false")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(len(data["projects"]), 2)
        self.assertNotIn("project_statistics", data["data"])

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_current_project")
    @patch("agraph.api.routers.projects.get_project_paths")
    @patch("pathlib.Path")
    def test_get_projects_overview_with_stats(
        self, mock_path_class, mock_get_paths, mock_get_current, mock_list_projects
    ):
        """Test getting projects overview with statistics."""
        mock_list_projects.return_value = ["test_project"]
        mock_get_current.return_value = "test_project"
        mock_paths = {
            "project_dir": "/path/to/test_project",
            "document_storage": "/path/to/test_project/documents",
            "vector_db": "/path/to/test_project/vectordb",
            "cache": "/path/to/test_project/cache",
        }
        mock_get_paths.return_value = mock_paths

        # Mock Path objects
        mock_project_dir = MagicMock()
        mock_project_dir.exists.return_value = True
        mock_project_dir.rglob.return_value = []  # No files

        mock_doc_dir = MagicMock()
        mock_doc_dir.exists.return_value = True
        mock_doc_dir.glob.return_value = []  # No documents

        mock_vector_dir = MagicMock()
        mock_vector_dir.exists.return_value = True
        mock_vector_dir.iterdir.return_value = [MagicMock()]  # Has content

        def path_side_effect(path_str):
            if "documents" in path_str:
                return mock_doc_dir
            elif "vectordb" in path_str:
                return mock_vector_dir
            elif "cache" in path_str:
                mock_cache_dir = MagicMock()
                mock_cache_dir.exists.return_value = False
                return mock_cache_dir
            else:
                return mock_project_dir

        mock_path_class.side_effect = path_side_effect

        response = self.client.get("/projects/?include_stats=true")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("project_statistics", data["data"])

        stats = data["data"]["project_statistics"]["test_project"]
        self.assertEqual(stats["document_count"], 0)
        self.assertTrue(stats["has_vector_db"])

    @patch("agraph.api.routers.projects.list_projects")
    def test_projects_overview_error_handling(self, mock_list_projects):
        """Test error handling in projects overview."""
        mock_list_projects.side_effect = Exception("Database error")

        response = self.client.get("/projects/")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Database error")

    @patch("agraph.api.routers.projects.create_project")
    def test_create_project_general_error(self, mock_create_project):
        """Test project creation with general error."""
        mock_create_project.side_effect = Exception("Filesystem error")

        test_data = {"name": "test_project", "description": "Test description"}

        response = self.client.post("/projects/create", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Filesystem error")

    def test_create_project_invalid_json(self):
        """Test project creation with invalid JSON payload."""
        # Send invalid JSON (missing required fields)
        response = self.client.post("/projects/create", json={})

        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_create_project_name_validation(self):
        """Test project creation with invalid name length."""
        # Test with name that's too long (> 50 characters)
        long_name = "a" * 51
        test_data = {"name": long_name, "description": "Test description"}

        response = self.client.post("/projects/create", json=test_data)

        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    @patch("agraph.api.routers.projects.get_current_project")
    @patch("agraph.api.routers.projects.get_project_paths")
    def test_get_current_project_config_file_error(self, mock_get_paths, mock_get_current):
        """Test getting current project when config file has issues."""
        mock_get_current.return_value = "test_project"
        mock_paths = {
            "project_dir": "/path/to/test_project",
            "config_file": "/path/to/test_project/config.json",
        }
        mock_get_paths.return_value = mock_paths

        # Mock file that raises exception when read
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", side_effect=IOError("Cannot read file")),
        ):

            response = self.client.get("/projects/current")

            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("Cannot read file", data["message"])

    @patch("agraph.api.routers.projects.list_projects")
    @patch("agraph.api.routers.projects.get_current_project")
    @patch("agraph.api.routers.projects.get_project_paths")
    def test_projects_overview_stats_error_handling(
        self, mock_get_paths, mock_get_current, mock_list_projects
    ):
        """Test projects overview when stats collection fails for some projects."""
        mock_list_projects.return_value = ["good_project", "bad_project"]
        mock_get_current.return_value = None

        def paths_side_effect(project_name):
            if project_name == "bad_project":
                raise Exception("Path error")
            return {
                "project_dir": f"/path/to/{project_name}",
                "document_storage": f"/path/to/{project_name}/documents",
                "vector_db": f"/path/to/{project_name}/vectordb",
                "cache": f"/path/to/{project_name}/cache",
            }

        mock_get_paths.side_effect = paths_side_effect

        with patch("pathlib.Path") as mock_path_class:
            # Mock successful path for good_project
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.glob.return_value = []
            mock_dir.rglob.return_value = []
            mock_dir.iterdir.return_value = []
            mock_path_class.return_value = mock_dir

            response = self.client.get("/projects/?include_stats=true")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)

            # Should include error for bad_project
            stats = data["data"]["project_statistics"]
            self.assertIn("good_project", stats)
            self.assertIn("bad_project", stats)
            self.assertIn("error", stats["bad_project"])


if __name__ == "__main__":
    unittest.main()
