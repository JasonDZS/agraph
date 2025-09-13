"""Shared dependencies for AGraph API with project-based local caching."""

from pathlib import Path
from typing import Any, Optional

from ..agraph import AGraph
from ..base.infrastructure.instances import register_reset_callback
from ..config import (
    Settings,
    create_project,
    get_project_paths,
    get_settings,
    load_project_settings,
    save_project_settings,
)
from ..logger import logger
from .document_manager import DocumentManager

# Project-specific instances cache
_agraph_instances: dict[str, AGraph] = {}
_document_managers: dict[str, DocumentManager] = {}


async def get_agraph_instance(project_name: Optional[str] = None) -> AGraph:
    """Get or create AGraph instance for a specific project with local caching."""
    from ..logger import logger  # pylint: disable=import-outside-toplevel

    logger.info(f"Getting AGraph instance for project: {project_name or 'default'}")
    logger.info(f"Current instances: {list(_agraph_instances.keys())}")

    if project_name:
        # Ensure project exists and has proper configuration
        await _ensure_project_config_exists(project_name)
        settings = load_project_settings(project_name)
        current_project = project_name
        logger.info(
            f"Project settings loaded for {project_name}: model={settings.llm.model}, provider={settings.llm.provider}"
        )
    else:
        settings = get_settings()
        current_project = settings.current_project or "default"
        logger.info(f"Global settings loaded: model={settings.llm.model}, provider={settings.llm.provider}")

    # Use project-specific paths if project is specified
    if current_project != "default":
        project_paths = get_project_paths(current_project, settings.workdir)
        cache_dir = project_paths["cache"]
        persist_directory = project_paths["vector_db"]
        collection_name = f"agraph_{current_project}"
    else:
        # Use default paths for backward compatibility
        cache_dir = f"{settings.workdir}/cache"
        persist_directory = f"{settings.workdir}/agraph_vectordb"
        collection_name = "agraph_api"

    # Check if we already have an instance for this project
    if current_project in _agraph_instances:
        existing_instance = _agraph_instances[current_project]
        # Verify the instance is still valid and has the correct collection name
        if existing_instance.collection_name == collection_name:
            # Check if the configuration has changed by comparing settings
            current_settings = existing_instance.settings

            def _is_config_unchanged() -> bool:
                """Check if configuration remains unchanged."""
                return (
                    current_settings.llm.model == settings.llm.model
                    and current_settings.llm.provider == settings.llm.provider
                    and current_settings.llm.temperature == settings.llm.temperature
                    and current_settings.llm.max_tokens == settings.llm.max_tokens
                    and current_settings.text.max_chunk_size == settings.text.max_chunk_size
                    and current_settings.text.chunk_overlap == settings.text.chunk_overlap
                    and current_settings.openai.api_key == settings.openai.api_key
                    and current_settings.openai.api_base == settings.openai.api_base
                    and current_settings.embedding.provider == settings.embedding.provider
                    and current_settings.embedding.model == settings.embedding.model
                )

            if _is_config_unchanged():
                return existing_instance
        # Configuration changed or collection name mismatch, close and remove old instance
        logger.info(f"Configuration changed for project {current_project}, removing existing instance")
        logger.info(f"Old model: {current_settings.llm.model}, New model: {settings.llm.model}")
        logger.info(f"Old temperature: {current_settings.llm.temperature}, New temperature: {settings.llm.temperature}")
        await existing_instance.close()
        del _agraph_instances[current_project]

    # Create new instance for this project
    logger.info(
        f"Creating new AGraph instance for {current_project} with model: {settings.llm.model}, "
        f"temperature: {settings.llm.temperature}, api_base: {settings.openai.api_base}"
    )

    # Update settings to use project-specific paths
    if current_project != "default":
        # Create a copy of settings with project-specific configurations
        project_settings = settings.model_copy()
        project_settings.builder.cache_dir = cache_dir
    else:
        project_settings = settings
        project_settings.builder.cache_dir = cache_dir

    new_instance = AGraph(
        settings=project_settings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    await new_instance.initialize()

    # Cache the new instance
    _agraph_instances[current_project] = new_instance

    return new_instance


def get_document_manager(project_name: Optional[str] = None) -> DocumentManager:
    """Get or create DocumentManager instance for a specific project with local caching."""
    if project_name:
        # Ensure project configuration is cached locally
        try:
            settings = load_project_settings(project_name)
        except FileNotFoundError:
            # Project doesn't exist, create it with default settings
            logger.info(f"Creating project {project_name} as it doesn't exist")
            create_project(project_name, f"Auto-created project: {project_name}")
            settings = load_project_settings(project_name)
        current_project = project_name
    else:
        settings = get_settings()
        current_project = settings.current_project or "default"

    # Use project-specific paths if project is specified
    if current_project != "default":
        project_paths = get_project_paths(current_project, settings.workdir)
        storage_dir = project_paths["document_storage"]
    else:
        # Use default path for backward compatibility
        storage_dir = f"{settings.workdir}/document_storage"

    # Check if we already have a manager for this project
    if current_project in _document_managers:
        existing_manager = _document_managers[current_project]
        # Verify the manager is for the correct project
        if existing_manager.project_name == current_project:
            return existing_manager
        del _document_managers[current_project]

    # Ensure project directory structure exists
    ensure_project_directory(current_project)

    # Create new manager for this project
    new_manager = DocumentManager(storage_dir, project_name=current_project)

    # Cache the new manager
    _document_managers[current_project] = new_manager

    return new_manager


async def get_agraph_instance_dependency() -> AGraph:
    """Fastapi dependency for getting AGraph instance."""
    return await get_agraph_instance()


def get_agraph_instance_dependency_factory(project_name: Optional[str] = None) -> Any:
    """Factory function to create project-aware AGraph dependency."""

    async def _get_agraph_instance() -> AGraph:
        return await get_agraph_instance(project_name)

    return _get_agraph_instance


def get_document_manager_dependency() -> DocumentManager:
    """Fastapi dependency for getting DocumentManager instance."""
    return get_document_manager()


# Project-aware dependency functions
async def get_project_agraph_instance_dependency(project_name: Optional[str] = None) -> AGraph:
    """Fastapi dependency for getting project-specific AGraph instance."""
    return await get_agraph_instance(project_name)


def get_project_document_manager_dependency(project_name: Optional[str] = None) -> DocumentManager:
    """Fastapi dependency for getting project-specific DocumentManager instance."""
    return get_document_manager(project_name)


async def close_agraph_instance(project_name: Optional[str] = None) -> None:
    """Close AGraph instance for a specific project or all instances."""
    if project_name:
        # Close specific project instance
        if project_name in _agraph_instances:
            await _agraph_instances[project_name].close()
            del _agraph_instances[project_name]
    else:
        # Close all instances
        for instance in _agraph_instances.values():
            await instance.close()
        _agraph_instances.clear()


def _reset_local_instances(project_name: Optional[str] = None) -> None:
    """Reset local instances for a specific project or all instances."""
    if project_name:
        # Reset specific project instances
        if project_name in _agraph_instances:
            del _agraph_instances[project_name]
        if project_name in _document_managers:
            del _document_managers[project_name]
    else:
        # Reset all instances
        _agraph_instances.clear()
        _document_managers.clear()


async def _ensure_project_config_exists(project_name: str) -> None:
    """Ensure project configuration exists locally, create if necessary."""
    try:
        # Try to load existing project settings
        load_project_settings(project_name)
        logger.debug(f"Project {project_name} configuration already exists")
    except FileNotFoundError:
        # Project doesn't exist, create it with current global settings as base
        logger.info(f"Creating project {project_name} with default configuration")
        create_project(project_name, f"Auto-created project: {project_name}")
        logger.info(f"Project {project_name} created successfully")


async def save_project_config_changes(project_name: str, settings: Settings) -> str:
    """Save project configuration changes to local file."""
    try:
        config_path = save_project_settings(project_name, settings)
        logger.info(f"Saved configuration for project {project_name} to {config_path}")

        # Reset instance to use new configuration
        if project_name in _agraph_instances:
            await _agraph_instances[project_name].close()
            del _agraph_instances[project_name]
            logger.info(f"Reset AGraph instance for project {project_name} to use new config")

        if project_name in _document_managers:
            del _document_managers[project_name]
            logger.info(f"Reset DocumentManager instance for project {project_name}")

        return config_path
    except Exception as e:
        logger.error(f"Failed to save project configuration: {e}")
        raise


def get_project_config_path(project_name: str) -> str:
    """Get the path to project's configuration file."""
    project_paths = get_project_paths(project_name)
    return project_paths["config_file"]


def ensure_project_directory(project_name: str) -> dict:
    """Ensure project directory structure exists."""
    project_paths = get_project_paths(project_name)

    # Create directories if they don't exist
    for path_key, path_value in project_paths.items():
        if path_key != "config_file":  # Skip config file, only create dirs
            Path(path_value).mkdir(parents=True, exist_ok=True)

    return project_paths


# Register our reset callback
register_reset_callback("api_dependencies", _reset_local_instances)
