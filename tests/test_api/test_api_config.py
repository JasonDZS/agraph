"""Tests for Configuration Management API router."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestConfigRouter(unittest.TestCase):
    """Test Configuration Management API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.config.get_settings")
    @patch("agraph.api.routers.config.get_agraph_instance")
    def test_get_config_global(self, mock_get_agraph, mock_get_settings):
        """Test getting global configuration."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {
            "workdir": "/tmp/agraph",
            "llm": {"model": "gpt-3.5-turbo", "temperature": 0.7},
            "openai": {"api_key": "test-key"},
        }
        mock_get_settings.return_value = mock_settings

        # Mock AGraph instance
        mock_agraph = MagicMock()
        mock_agraph.collection_name = "agraph_api"
        mock_agraph.vector_store_type = "chroma"
        mock_agraph.is_initialized = True
        mock_agraph.has_knowledge_graph = False
        mock_agraph.config = None
        mock_agraph.close = MagicMock()
        mock_get_agraph.return_value = mock_agraph

        response = self.client.get("/config")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Configuration retrieved successfully")

        # Verify runtime info is included
        self.assertIn("runtime_info", data["data"])
        self.assertEqual(data["data"]["runtime_info"]["collection_name"], "agraph_api")

    @patch("agraph.api.routers.config.load_project_settings")
    @patch("agraph.api.routers.config.get_project_info")
    @patch("agraph.api.routers.config.get_agraph_instance")
    def test_get_config_project_specific(
        self, mock_get_agraph, mock_get_project_info, mock_load_project
    ):
        """Test getting project-specific configuration."""
        # Mock project settings
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"llm": {"model": "gpt-4", "temperature": 0.5}}
        mock_load_project.return_value = mock_settings

        # Mock project info
        mock_project_info = {
            "name": "test_project",
            "description": "Test project",
            "created_at": "2024-01-01",
        }
        mock_get_project_info.return_value = mock_project_info

        # Mock AGraph instance that raises exception (to test fallback)
        mock_get_agraph.side_effect = Exception("Runtime error")

        response = self.client.get("/config?project_name=test_project")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify project-specific data is included
        self.assertIn("project_info", data["data"])
        self.assertEqual(data["data"]["project_info"], mock_project_info)
        self.assertIn("settings", data["data"])
        self.assertIsNone(data["data"]["runtime_info"])  # Should be None due to exception

    @patch("agraph.api.routers.config.update_settings")
    def test_update_config_global(self, mock_update_settings):
        """Test updating global configuration."""
        # Mock updated settings
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"llm": {"model": "gpt-4", "temperature": 0.8}}
        mock_update_settings.return_value = mock_settings

        test_data = {"llm_model": "gpt-4", "llm_temperature": 0.8, "openai_api_key": "new-key"}

        response = self.client.post("/config", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Configuration updated successfully", data["message"])
        self.assertIn("3 sections modified", data["message"])

        # Verify update_settings was called with correct structure
        mock_update_settings.assert_called_once()
        call_args = mock_update_settings.call_args[0][0]
        self.assertIn("llm", call_args)
        self.assertIn("openai", call_args)
        self.assertEqual(call_args["llm"]["model"], "gpt-4")
        self.assertEqual(call_args["llm"]["temperature"], 0.8)
        self.assertEqual(call_args["openai"]["api_key"], "new-key")

    @patch("agraph.api.routers.config.update_project_settings")
    def test_update_config_project_specific(self, mock_update_project):
        """Test updating project-specific configuration."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"updated": True}
        mock_update_project.return_value = mock_settings

        test_data = {"llm_model": "gpt-4", "max_chunk_size": 1000}

        response = self.client.post("/config?project_name=test_project", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Project configuration updated", data["message"])

        # Verify project update was called
        mock_update_project.assert_called_once_with("test_project", unittest.mock.ANY)

    @patch("agraph.api.routers.config.get_settings")
    def test_update_config_no_changes(self, mock_get_settings):
        """Test updating configuration with no changes."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"unchanged": True}
        mock_get_settings.return_value = mock_settings

        # Empty request
        test_data = {}

        response = self.client.post("/config", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "No configuration changes provided")

    @patch("agraph.api.routers.config.reset_settings")
    def test_reset_config_global(self, mock_reset_settings):
        """Test resetting global configuration."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"reset": True}
        mock_reset_settings.return_value = mock_settings

        response = self.client.post("/config/reset")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Configuration reset to default values successfully")

        mock_reset_settings.assert_called_once()

    @patch("agraph.api.routers.config.reset_project_settings")
    def test_reset_config_project_specific(self, mock_reset_project):
        """Test resetting project-specific configuration."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"reset": True}
        mock_reset_project.return_value = mock_settings

        response = self.client.post("/config/reset?project_name=test_project")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Project configuration reset", data["message"])

        mock_reset_project.assert_called_once_with("test_project")

    @patch("agraph.api.routers.config.save_settings_to_file")
    @patch("agraph.api.routers.config.get_settings")
    def test_save_config_to_file(self, mock_get_settings, mock_save_to_file):
        """Test saving configuration to file."""
        mock_save_to_file.return_value = "/path/to/config.json"
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"saved": True}
        mock_get_settings.return_value = mock_settings

        test_data = {"file_path": "/custom/path/config.json"}

        response = self.client.post("/config/save", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Configuration saved successfully")
        self.assertEqual(data["file_path"], "/path/to/config.json")

        mock_save_to_file.assert_called_once_with("/custom/path/config.json", project_name=None)

    @patch("agraph.api.routers.config.load_settings_from_file")
    @patch("agraph.api.routers.config.get_config_file_path")
    def test_load_config_from_file(self, mock_get_file_path, mock_load_from_file):
        """Test loading configuration from file."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"loaded": True}
        mock_load_from_file.return_value = mock_settings
        mock_get_file_path.return_value = "/default/config.json"

        test_data = {"file_path": "/custom/config.json"}

        response = self.client.post("/config/load", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Configuration loaded successfully")

        mock_load_from_file.assert_called_once_with("/custom/config.json", project_name=None)

    @patch("agraph.api.routers.config.load_settings_from_file")
    def test_load_config_file_not_found(self, mock_load_from_file):
        """Test loading configuration from non-existent file."""
        mock_load_from_file.side_effect = FileNotFoundError("Config file not found")

        test_data = {"file_path": "/nonexistent/config.json"}

        response = self.client.post("/config/load", json=test_data)

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("Config file not found", data["message"])

    @patch("agraph.api.routers.config.get_config_file_path")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.parent")
    def test_get_config_file_path_info(self, mock_parent, mock_exists, mock_get_file_path):
        """Test getting configuration file path information."""
        mock_get_file_path.return_value = "/path/to/config.json"
        mock_exists.return_value = True

        mock_parent_path = MagicMock()
        mock_parent_path.exists.return_value = True
        mock_parent.return_value = mock_parent_path

        response = self.client.get("/config/file-path")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["file_path"], "/path/to/config.json")
        self.assertTrue(data["data"]["exists"])
        self.assertTrue(data["data"]["writable"])

    @patch("agraph.api.routers.config.copy_project_settings")
    def test_copy_project_config_success(self, mock_copy_settings):
        """Test successful project configuration copying."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"copied": True}
        mock_copy_settings.return_value = mock_settings

        response = self.client.post("/config/projects/target_project/copy-from/source_project")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertIn("Configuration successfully copied", data["message"])

        mock_copy_settings.assert_called_once_with("source_project", "target_project")

    @patch("agraph.api.routers.config.copy_project_settings")
    def test_copy_project_config_source_not_found(self, mock_copy_settings):
        """Test copying project configuration when source doesn't exist."""
        mock_copy_settings.side_effect = FileNotFoundError("Source project not found")

        response = self.client.post("/config/projects/target/copy-from/nonexistent")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("Source project not found", data["message"])

    @patch("agraph.api.routers.config.get_project_info")
    def test_get_project_config_info_success(self, mock_get_project_info):
        """Test getting project configuration info."""
        mock_project_info = {
            "name": "test_project",
            "description": "Test project",
            "config_file": "/path/to/config.json",
            "paths": {"project_dir": "/path/to/test_project"},
        }
        mock_get_project_info.return_value = mock_project_info

        response = self.client.get("/config/projects/test_project/info")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["data"], mock_project_info)

    @patch("agraph.api.routers.config.get_project_info")
    def test_get_project_config_info_not_found(self, mock_get_project_info):
        """Test getting project config info for non-existent project."""
        mock_get_project_info.side_effect = FileNotFoundError("Project not found")

        response = self.client.get("/config/projects/nonexistent/info")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("Project not found", data["message"])

    def test_update_config_validation_types(self):
        """Test configuration update with various data types."""
        test_data = {
            "llm_temperature": 0.9,
            "llm_max_tokens": 2000,
            "max_chunk_size": 1500,
            "entity_types": ["PERSON", "ORGANIZATION"],
            "relation_types": ["WORKS_FOR", "KNOWS"],
        }

        with patch("agraph.api.routers.config.update_settings") as mock_update:
            mock_settings = MagicMock()
            mock_settings.to_dict.return_value = {"updated": True}
            mock_update.return_value = mock_settings

            response = self.client.post("/config", json=test_data)

            self.assertEqual(response.status_code, 200)

            # Verify the correct data structure was passed
            call_args = mock_update.call_args[0][0]
            self.assertEqual(call_args["llm"]["temperature"], 0.9)
            self.assertEqual(call_args["llm"]["max_tokens"], 2000)
            self.assertEqual(call_args["text"]["max_chunk_size"], 1500)
            self.assertEqual(call_args["graph"]["entity_types"], ["PERSON", "ORGANIZATION"])

    def test_update_config_legacy_chunk_size_support(self):
        """Test configuration update with legacy chunk_size parameter."""
        test_data = {"chunk_size": 800}  # Legacy parameter name

        with patch("agraph.api.routers.config.update_settings") as mock_update:
            mock_settings = MagicMock()
            mock_settings.to_dict.return_value = {"updated": True}
            mock_update.return_value = mock_settings

            response = self.client.post("/config", json=test_data)

            self.assertEqual(response.status_code, 200)

            # Verify legacy parameter is converted to new format
            call_args = mock_update.call_args[0][0]
            self.assertEqual(call_args["text"]["max_chunk_size"], 800)

    @patch("agraph.api.routers.config.update_settings")
    def test_update_config_error_handling(self, mock_update_settings):
        """Test configuration update error handling."""
        mock_update_settings.side_effect = Exception("Configuration error")

        test_data = {"llm_model": "gpt-4"}

        response = self.client.post("/config", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Configuration error")

    @patch("agraph.api.routers.config.get_settings")
    def test_get_config_error_handling(self, mock_get_settings):
        """Test get configuration error handling."""
        mock_get_settings.side_effect = Exception("Settings error")

        response = self.client.get("/config")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Settings error")

    @patch("agraph.api.routers.config.save_settings_to_file")
    def test_save_config_error_handling(self, mock_save_to_file):
        """Test save configuration error handling."""
        mock_save_to_file.side_effect = Exception("Save error")

        test_data = {"file_path": "/path/to/config.json"}

        response = self.client.post("/config/save", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Save error")

    @patch("agraph.api.routers.config.reset_settings")
    def test_reset_config_error_handling(self, mock_reset_settings):
        """Test reset configuration error handling."""
        mock_reset_settings.side_effect = Exception("Reset error")

        response = self.client.post("/config/reset")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Reset error")

    @patch("agraph.api.routers.config.load_project_settings")
    @patch("agraph.api.routers.config.get_project_info")
    def test_load_project_config_with_project_name(self, mock_get_project_info, mock_load_project):
        """Test loading project-specific configuration."""
        mock_settings = MagicMock()
        mock_settings.to_dict.return_value = {"project_config": True}
        mock_load_project.return_value = mock_settings

        test_data = {"file_path": "/project/config.json"}

        with (
            patch("agraph.api.routers.config.load_settings_from_file") as mock_load_from_file,
            patch("agraph.api.routers.config.get_config_file_path") as mock_get_file_path,
        ):

            mock_load_from_file.return_value = mock_settings
            mock_get_file_path.return_value = "/project/config.json"

            response = self.client.post("/config/load?project_name=test_project", json=test_data)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)
            self.assertIn("Project configuration loaded", data["message"])

    def test_update_config_all_parameters(self):
        """Test configuration update with all possible parameters."""
        test_data = {
            "workdir": "/new/workdir",
            "openai_api_key": "new-key",
            "openai_api_base": "https://custom.openai.com",
            "llm_model": "gpt-4",
            "llm_temperature": 0.8,
            "llm_max_tokens": 3000,
            "llm_provider": "openai",
            "embedding_model": "text-embedding-ada-002",
            "embedding_provider": "openai",
            "embedding_dimension": 1536,
            "embedding_max_token_size": 8191,
            "embedding_batch_size": 100,
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"],
            "relation_types": ["WORKS_FOR", "LOCATED_IN", "KNOWS"],
            "max_chunk_size": 1200,
            "chunk_overlap": 200,
            "system_prompt": "Custom system prompt",
        }

        with patch("agraph.api.routers.config.update_settings") as mock_update:
            mock_settings = MagicMock()
            mock_settings.to_dict.return_value = {"all_updated": True}
            mock_update.return_value = mock_settings

            response = self.client.post("/config", json=test_data)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], ResponseStatus.SUCCESS)

            # Verify all sections are included in the update
            call_args = mock_update.call_args[0][0]
            expected_sections = ["workdir", "openai", "llm", "embedding", "graph", "text", "rag"]

            # Check for presence of main sections (some might be nested)
            self.assertIn("openai", call_args)
            self.assertIn("llm", call_args)
            self.assertIn("embedding", call_args)
            self.assertIn("graph", call_args)
            self.assertIn("text", call_args)
            self.assertIn("rag", call_args)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_invalid_json_parameters(self):
        """Test handling of invalid parameter types."""
        # Temperature should be float, not string
        test_data = {"llm_temperature": "invalid_float"}

        response = self.client.post("/config", json=test_data)

        # Should return validation error
        self.assertEqual(response.status_code, 422)

    def test_negative_values_validation(self):
        """Test validation of negative values."""
        test_data = {"llm_max_tokens": -100, "max_chunk_size": -500}

        # This should pass validation at the API level but may fail at config level
        with patch("agraph.api.routers.config.update_settings") as mock_update:
            mock_settings = MagicMock()
            mock_settings.to_dict.return_value = {"negative_values": True}
            mock_update.return_value = mock_settings

            response = self.client.post("/config", json=test_data)

            # API should accept it and pass to config layer for validation
            self.assertEqual(response.status_code, 200)

    @patch("agraph.api.routers.config.get_config_file_path")
    def test_get_file_path_error_handling(self, mock_get_file_path):
        """Test file path retrieval error handling."""
        mock_get_file_path.side_effect = Exception("Path error")

        response = self.client.get("/config/file-path")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Path error")


if __name__ == "__main__":
    unittest.main()
