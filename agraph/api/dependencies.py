"""Shared dependencies for AGraph API."""

from typing import Optional

from ..agraph import AGraph
from ..config import BuilderConfig, get_project_paths, get_settings
from .document_manager import DocumentManager

# Project-specific instances cache
_agraph_instances: dict[str, AGraph] = {}
_document_managers: dict[str, DocumentManager] = {}


async def get_agraph_instance(project_name: Optional[str] = None) -> AGraph:
    """Get or create AGraph instance for a specific project."""
    settings = get_settings()
    current_project = project_name or settings.current_project or "default"

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
            return existing_instance
        # Collection name mismatch, close and remove old instance
        await existing_instance.close()
        del _agraph_instances[current_project]

    # Create new instance for this project
    config = BuilderConfig(
        chunk_size=settings.text.max_chunk_size,
        chunk_overlap=settings.text.chunk_overlap,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
        entity_confidence_threshold=0.7,
        relation_confidence_threshold=0.6,
        cache_dir=cache_dir,
    )

    new_instance = AGraph(
        collection_name=collection_name,
        persist_directory=persist_directory,
        vector_store_type="chroma",
        config=config,
        use_openai_embeddings=True,
        enable_knowledge_graph=True,
    )

    await new_instance.initialize()

    # Cache the new instance
    _agraph_instances[current_project] = new_instance

    return new_instance


def get_document_manager(project_name: Optional[str] = None) -> DocumentManager:
    """Get or create DocumentManager instance for a specific project."""
    settings = get_settings()
    current_project = project_name or settings.current_project or "default"

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

    # Create new manager for this project
    new_manager = DocumentManager(storage_dir, project_name=current_project)

    # Cache the new manager
    _document_managers[current_project] = new_manager

    return new_manager


async def get_agraph_instance_dependency() -> AGraph:
    """Fastapi dependency for getting AGraph instance."""
    return await get_agraph_instance()


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


def reset_instances(project_name: Optional[str] = None) -> None:
    """Reset instances for a specific project or all instances."""
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
