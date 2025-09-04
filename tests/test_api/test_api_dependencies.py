"""Tests for API dependency injection system."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agraph.api.dependencies import (
    _reset_local_instances,
    close_agraph_instance,
    get_agraph_instance,
    get_agraph_instance_dependency,
    get_agraph_instance_dependency_factory,
    get_document_manager,
    get_document_manager_dependency,
    get_project_agraph_instance_dependency,
    get_project_document_manager_dependency,
)


class TestAGraphInstanceManagement(unittest.TestCase):
    """Test AGraph instance management and caching."""

    def setUp(self):
        """Set up test environment."""
        # Reset instances before each test
        _reset_local_instances()

    def tearDown(self):
        """Clean up after each test."""
        # Reset instances after each test
        _reset_local_instances()

    @patch("agraph.api.dependencies.get_settings")
    @patch("agraph.api.dependencies.AGraph")
    async def test_get_agraph_instance_default_project(self, mock_agraph_class, mock_get_settings):
        """Test getting AGraph instance for default project."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.current_project = None
        mock_settings.workdir = "/tmp/agraph"
        mock_settings.text.max_chunk_size = 1000
        mock_settings.text.chunk_overlap = 200
        mock_settings.llm.provider = "openai"
        mock_settings.llm.model = "gpt-3.5-turbo"
        mock_get_settings.return_value = mock_settings

        # Mock AGraph instance
        mock_instance = AsyncMock()
        mock_instance.collection_name = "agraph_api"
        mock_instance.config = MagicMock()
        mock_instance.config.llm_model = "gpt-3.5-turbo"
        mock_instance.config.llm_provider = "openai"
        mock_instance.config.chunk_size = 1000
        mock_instance.config.chunk_overlap = 200
        mock_instance.settings = mock_settings
        mock_instance.initialize = AsyncMock()
        mock_agraph_class.return_value = mock_instance

        # First call should create instance
        result = await get_agraph_instance()

        self.assertEqual(result, mock_instance)
        mock_agraph_class.assert_called_once()
        mock_instance.initialize.assert_called_once()

        # Second call should return cached instance
        mock_agraph_class.reset_mock()
        mock_instance.initialize.reset_mock()

        result2 = await get_agraph_instance()

        self.assertEqual(result2, mock_instance)
        mock_agraph_class.assert_not_called()  # Should use cached instance
        mock_instance.initialize.assert_not_called()

    @patch("agraph.api.dependencies.load_project_settings")
    @patch("agraph.api.dependencies.get_project_paths")
    @patch("agraph.api.dependencies.AGraph")
    async def test_get_agraph_instance_specific_project(
        self, mock_agraph_class, mock_get_paths, mock_load_project
    ):
        """Test getting AGraph instance for specific project."""
        # Mock project settings
        mock_settings = MagicMock()
        mock_settings.workdir = "/tmp/agraph"
        mock_settings.text.max_chunk_size = 800
        mock_settings.text.chunk_overlap = 100
        mock_settings.llm.provider = "openai"
        mock_settings.llm.model = "gpt-4"
        mock_settings.llm.temperature = 0.7
        mock_settings.llm.max_tokens = 2000
        mock_settings.openai.api_key = "test-key"
        mock_settings.openai.api_base = "https://api.openai.com/v1"
        mock_load_project.return_value = mock_settings

        # Mock project paths
        mock_paths = {
            "cache": "/tmp/agraph/projects/test_project/cache",
            "vector_db": "/tmp/agraph/projects/test_project/vectordb",
        }
        mock_get_paths.return_value = mock_paths

        # Mock AGraph instance
        mock_instance = AsyncMock()
        mock_instance.collection_name = "agraph_test_project"
        mock_instance.config = MagicMock()
        mock_instance.config.llm_model = "gpt-4"
        mock_instance.config.llm_provider = "openai"
        mock_instance.config.chunk_size = 800
        mock_instance.config.chunk_overlap = 100
        mock_instance.settings = mock_settings
        mock_instance.initialize = AsyncMock()
        mock_agraph_class.return_value = mock_instance

        result = await get_agraph_instance("test_project")

        self.assertEqual(result, mock_instance)
        mock_load_project.assert_called_once_with("test_project")
        mock_get_paths.assert_called_once_with("test_project", "/tmp/agraph")
        mock_instance.initialize.assert_called_once()

    @patch("agraph.api.dependencies.get_settings")
    @patch("agraph.api.dependencies.AGraph")
    async def test_get_agraph_instance_configuration_change(
        self, mock_agraph_class, mock_get_settings
    ):
        """Test AGraph instance recreation when configuration changes."""
        # Mock initial settings
        mock_initial_settings = MagicMock()
        mock_initial_settings.current_project = None
        mock_initial_settings.workdir = "/tmp/agraph"
        mock_initial_settings.llm.model = "gpt-3.5-turbo"
        mock_initial_settings.llm.provider = "openai"
        mock_initial_settings.llm.temperature = 0.7
        mock_initial_settings.llm.max_tokens = 1500
        mock_initial_settings.openai.api_key = "old-key"
        mock_initial_settings.openai.api_base = "https://api.openai.com/v1"
        mock_initial_settings.text.max_chunk_size = 1000
        mock_initial_settings.text.chunk_overlap = 200

        # Mock changed settings
        mock_changed_settings = MagicMock()
        mock_changed_settings.current_project = None
        mock_changed_settings.workdir = "/tmp/agraph"
        mock_changed_settings.llm.model = "gpt-4"  # Changed
        mock_changed_settings.llm.provider = "openai"
        mock_changed_settings.llm.temperature = 0.5  # Changed
        mock_changed_settings.llm.max_tokens = 2000  # Changed
        mock_changed_settings.openai.api_key = "new-key"  # Changed
        mock_changed_settings.openai.api_base = "https://api.openai.com/v1"
        mock_changed_settings.text.max_chunk_size = 1000
        mock_changed_settings.text.chunk_overlap = 200

        # First call with initial settings
        mock_get_settings.return_value = mock_initial_settings

        mock_initial_instance = AsyncMock()
        mock_initial_instance.collection_name = "agraph_api"
        mock_initial_instance.config = MagicMock()
        mock_initial_instance.config.llm_model = "gpt-3.5-turbo"
        mock_initial_instance.config.llm_provider = "openai"
        mock_initial_instance.config.chunk_size = 1000
        mock_initial_instance.config.chunk_overlap = 200
        mock_initial_instance.settings = mock_initial_settings
        mock_initial_instance.close = AsyncMock()
        mock_initial_instance.initialize = AsyncMock()

        mock_agraph_class.return_value = mock_initial_instance

        result1 = await get_agraph_instance()
        self.assertEqual(result1, mock_initial_instance)

        # Second call with changed settings
        mock_get_settings.return_value = mock_changed_settings

        mock_new_instance = AsyncMock()
        mock_new_instance.collection_name = "agraph_api"
        mock_new_instance.config = MagicMock()
        mock_new_instance.config.llm_model = "gpt-4"
        mock_new_instance.config.llm_provider = "openai"
        mock_new_instance.config.chunk_size = 1000
        mock_new_instance.config.chunk_overlap = 200
        mock_new_instance.settings = mock_changed_settings
        mock_new_instance.initialize = AsyncMock()

        mock_agraph_class.return_value = mock_new_instance

        result2 = await get_agraph_instance()

        # Should create new instance due to configuration change
        self.assertEqual(result2, mock_new_instance)
        self.assertNotEqual(result1, result2)

        # Old instance should be closed
        mock_initial_instance.close.assert_called_once()

    async def test_close_agraph_instance_specific_project(self):
        """Test closing specific project instance."""
        # Mock instances in cache
        with patch("agraph.api.dependencies._agraph_instances") as mock_instances:
            mock_instance1 = AsyncMock()
            mock_instance2 = AsyncMock()
            mock_instances.__contains__ = lambda self, key: key in {
                "project1": mock_instance1,
                "project2": mock_instance2,
            }
            mock_instances.__getitem__ = lambda self, key: {
                "project1": mock_instance1,
                "project2": mock_instance2,
            }[key]
            mock_instances.__delitem__ = MagicMock()

            await close_agraph_instance("project1")

            # Only project1 instance should be closed
            mock_instance1.close.assert_called_once()
            mock_instance2.close.assert_not_called()

    async def test_close_agraph_instance_all_projects(self):
        """Test closing all project instances."""
        # Mock instances in cache
        with patch("agraph.api.dependencies._agraph_instances") as mock_instances:
            mock_instance1 = AsyncMock()
            mock_instance2 = AsyncMock()
            mock_instances.values.return_value = [mock_instance1, mock_instance2]
            mock_instances.clear = MagicMock()

            await close_agraph_instance()

            # All instances should be closed
            mock_instance1.close.assert_called_once()
            mock_instance2.close.assert_called_once()
            mock_instances.clear.assert_called_once()


class TestDocumentManagerManagement(unittest.TestCase):
    """Test DocumentManager instance management."""

    def setUp(self):
        """Set up test environment."""
        _reset_local_instances()

    def tearDown(self):
        """Clean up after each test."""
        _reset_local_instances()

    @patch("agraph.api.dependencies.get_settings")
    @patch("agraph.api.dependencies.DocumentManager")
    def test_get_document_manager_default_project(self, mock_doc_manager_class, mock_get_settings):
        """Test getting DocumentManager for default project."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.current_project = None
        mock_settings.workdir = "/tmp/agraph"
        mock_get_settings.return_value = mock_settings

        # Mock DocumentManager instance
        mock_manager = MagicMock()
        mock_manager.project_name = "default"
        mock_doc_manager_class.return_value = mock_manager

        result = get_document_manager()

        self.assertEqual(result, mock_manager)
        mock_doc_manager_class.assert_called_once_with(
            "/tmp/agraph/document_storage", project_name="default"
        )

        # Second call should return cached instance
        mock_doc_manager_class.reset_mock()
        result2 = get_document_manager()

        self.assertEqual(result2, mock_manager)
        mock_doc_manager_class.assert_not_called()

    @patch("agraph.api.dependencies.load_project_settings")
    @patch("agraph.api.dependencies.get_project_paths")
    @patch("agraph.api.dependencies.DocumentManager")
    def test_get_document_manager_specific_project(
        self, mock_doc_manager_class, mock_get_paths, mock_load_project
    ):
        """Test getting DocumentManager for specific project."""
        # Mock project settings
        mock_settings = MagicMock()
        mock_settings.workdir = "/tmp/agraph"
        mock_load_project.return_value = mock_settings

        # Mock project paths
        mock_paths = {"document_storage": "/tmp/agraph/projects/test_project/documents"}
        mock_get_paths.return_value = mock_paths

        # Mock DocumentManager instance
        mock_manager = MagicMock()
        mock_manager.project_name = "test_project"
        mock_doc_manager_class.return_value = mock_manager

        result = get_document_manager("test_project")

        self.assertEqual(result, mock_manager)
        mock_doc_manager_class.assert_called_once_with(
            "/tmp/agraph/projects/test_project/documents", project_name="test_project"
        )

    def test_get_document_manager_dependency_function(self):
        """Test DocumentManager dependency function."""
        with patch("agraph.api.dependencies.get_document_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            result = get_document_manager_dependency()

            self.assertEqual(result, mock_manager)
            mock_get_manager.assert_called_once_with(None)

    async def test_get_agraph_instance_dependency_function(self):
        """Test AGraph instance dependency function."""
        with patch("agraph.api.dependencies.get_agraph_instance") as mock_get_instance:
            mock_instance = AsyncMock()
            mock_get_instance.return_value = mock_instance

            result = await get_agraph_instance_dependency()

            self.assertEqual(result, mock_instance)
            mock_get_instance.assert_called_once_with()

    async def test_project_specific_dependency_functions(self):
        """Test project-specific dependency functions."""
        with (
            patch("agraph.api.dependencies.get_agraph_instance") as mock_get_agraph,
            patch("agraph.api.dependencies.get_document_manager") as mock_get_doc,
        ):

            mock_agraph = AsyncMock()
            mock_doc_manager = MagicMock()
            mock_get_agraph.return_value = mock_agraph
            mock_get_doc.return_value = mock_doc_manager

            # Test project AGraph dependency
            agraph_result = await get_project_agraph_instance_dependency("test_project")
            self.assertEqual(agraph_result, mock_agraph)
            mock_get_agraph.assert_called_with("test_project")

            # Test project DocumentManager dependency
            doc_result = get_project_document_manager_dependency("test_project")
            self.assertEqual(doc_result, mock_doc_manager)
            mock_get_doc.assert_called_with("test_project")

    def test_agraph_instance_dependency_factory(self):
        """Test AGraph instance dependency factory."""
        factory_func = get_agraph_instance_dependency_factory("test_project")

        # Verify it returns a callable
        self.assertTrue(callable(factory_func))

        # Test the generated function
        with patch("agraph.api.dependencies.get_agraph_instance") as mock_get_instance:
            mock_instance = AsyncMock()
            mock_get_instance.return_value = mock_instance

            import asyncio

            result = asyncio.run(factory_func())

            self.assertEqual(result, mock_instance)
            mock_get_instance.assert_called_once_with("test_project")


class TestInstanceCaching(unittest.TestCase):
    """Test instance caching behavior."""

    def setUp(self):
        """Set up test environment."""
        _reset_local_instances()

    def tearDown(self):
        """Clean up after each test."""
        _reset_local_instances()

    @patch("agraph.api.dependencies.get_settings")
    @patch("agraph.api.dependencies.DocumentManager")
    def test_document_manager_project_mismatch_replacement(
        self, mock_doc_manager_class, mock_get_settings
    ):
        """Test DocumentManager replacement when project name mismatches."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.current_project = None
        mock_settings.workdir = "/tmp/agraph"
        mock_get_settings.return_value = mock_settings

        # First manager with wrong project name
        mock_old_manager = MagicMock()
        mock_old_manager.project_name = "wrong_project"

        # New manager with correct project name
        mock_new_manager = MagicMock()
        mock_new_manager.project_name = "default"

        mock_doc_manager_class.return_value = mock_new_manager

        # Manually add old manager to cache to simulate mismatch
        from agraph.api.dependencies import _document_managers

        _document_managers["default"] = mock_old_manager

        result = get_document_manager()

        # Should create new manager due to project name mismatch
        self.assertEqual(result, mock_new_manager)
        mock_doc_manager_class.assert_called_once()

    def test_reset_local_instances_specific_project(self):
        """Test resetting instances for specific project."""
        from agraph.api.dependencies import _agraph_instances, _document_managers

        # Add mock instances to cache
        _agraph_instances["project1"] = MagicMock()
        _agraph_instances["project2"] = MagicMock()
        _document_managers["project1"] = MagicMock()
        _document_managers["project2"] = MagicMock()

        # Reset only project1
        _reset_local_instances("project1")

        # Verify only project1 instances were removed
        self.assertNotIn("project1", _agraph_instances)
        self.assertIn("project2", _agraph_instances)
        self.assertNotIn("project1", _document_managers)
        self.assertIn("project2", _document_managers)

    def test_reset_local_instances_all_projects(self):
        """Test resetting all instances."""
        from agraph.api.dependencies import _agraph_instances, _document_managers

        # Add mock instances to cache
        _agraph_instances["project1"] = MagicMock()
        _agraph_instances["project2"] = MagicMock()
        _document_managers["project1"] = MagicMock()
        _document_managers["project2"] = MagicMock()

        # Reset all
        _reset_local_instances()

        # Verify all instances were removed
        self.assertEqual(len(_agraph_instances), 0)
        self.assertEqual(len(_document_managers), 0)


class TestDependencyIntegration(unittest.TestCase):
    """Test dependency system integration."""

    def setUp(self):
        """Set up test environment."""
        _reset_local_instances()

    def tearDown(self):
        """Clean up after each test."""
        _reset_local_instances()

    @patch("agraph.api.dependencies.register_reset_callback")
    def test_reset_callback_registration(self, mock_register):
        """Test that reset callback is properly registered."""
        # The callback should be registered when the module is imported
        # We can verify this by checking if register_reset_callback was called

        # Import should trigger callback registration
        import agraph.api.dependencies

        # Verify callback was registered
        mock_register.assert_called_with("api_dependencies", _reset_local_instances)

    @patch("agraph.api.dependencies.get_settings")
    @patch("agraph.api.dependencies.load_project_settings")
    @patch("agraph.api.dependencies.AGraph")
    async def test_multi_project_instance_isolation(
        self, mock_agraph_class, mock_load_project, mock_get_settings
    ):
        """Test that different projects have isolated instances."""
        # Mock different settings for different projects
        mock_global_settings = MagicMock()
        mock_global_settings.current_project = None
        mock_global_settings.workdir = "/tmp/agraph"
        mock_global_settings.llm.model = "gpt-3.5-turbo"
        mock_global_settings.llm.provider = "openai"
        mock_global_settings.llm.temperature = 0.7
        mock_global_settings.llm.max_tokens = 1500
        mock_global_settings.openai.api_key = "global-key"
        mock_global_settings.openai.api_base = "https://api.openai.com/v1"
        mock_global_settings.text.max_chunk_size = 1000
        mock_global_settings.text.chunk_overlap = 200
        mock_get_settings.return_value = mock_global_settings

        mock_project_settings = MagicMock()
        mock_project_settings.workdir = "/tmp/agraph"
        mock_project_settings.llm.model = "gpt-4"
        mock_project_settings.llm.provider = "openai"
        mock_project_settings.llm.temperature = 0.5
        mock_project_settings.llm.max_tokens = 2000
        mock_project_settings.openai.api_key = "project-key"
        mock_project_settings.openai.api_base = "https://api.openai.com/v1"
        mock_project_settings.text.max_chunk_size = 800
        mock_project_settings.text.chunk_overlap = 100
        mock_load_project.return_value = mock_project_settings

        # Mock different instances
        mock_default_instance = AsyncMock()
        mock_default_instance.collection_name = "agraph_api"
        mock_default_instance.config = MagicMock()
        mock_default_instance.config.llm_model = "gpt-3.5-turbo"
        mock_default_instance.config.llm_provider = "openai"
        mock_default_instance.config.chunk_size = 1000
        mock_default_instance.config.chunk_overlap = 200
        mock_default_instance.settings = mock_global_settings
        mock_default_instance.initialize = AsyncMock()

        mock_project_instance = AsyncMock()
        mock_project_instance.collection_name = "agraph_test_project"
        mock_project_instance.config = MagicMock()
        mock_project_instance.config.llm_model = "gpt-4"
        mock_project_instance.config.llm_provider = "openai"
        mock_project_instance.config.chunk_size = 800
        mock_project_instance.config.chunk_overlap = 100
        mock_project_instance.settings = mock_project_settings
        mock_project_instance.initialize = AsyncMock()

        def agraph_side_effect(*args, **kwargs):
            collection_name = kwargs.get("collection_name", "")
            if "test_project" in collection_name:
                return mock_project_instance
            return mock_default_instance

        mock_agraph_class.side_effect = agraph_side_effect

        with patch("agraph.api.dependencies.get_project_paths") as mock_get_paths:
            mock_get_paths.return_value = {
                "cache": "/tmp/agraph/projects/test_project/cache",
                "vector_db": "/tmp/agraph/projects/test_project/vectordb",
            }

            # Get instances for different projects
            default_instance = await get_agraph_instance()
            project_instance = await get_agraph_instance("test_project")

            # Verify different instances are returned
            self.assertNotEqual(default_instance, project_instance)
            self.assertEqual(default_instance.collection_name, "agraph_api")
            self.assertEqual(project_instance.collection_name, "agraph_test_project")

    def test_dependency_error_propagation(self):
        """Test that dependency errors are properly propagated."""
        with patch("agraph.api.dependencies.get_settings") as mock_get_settings:
            mock_get_settings.side_effect = Exception("Settings error")

            # Test that exceptions propagate from dependency functions
            with self.assertRaises(Exception) as context:
                import asyncio

                asyncio.run(get_agraph_instance())

            self.assertIn("Settings error", str(context.exception))


if __name__ == "__main__":
    unittest.main()
