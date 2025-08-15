"""Shared dependencies for AGraph API."""

from typing import Optional

from ..agraph import AGraph
from ..config import BuilderConfig, get_project_paths, get_settings
from .document_manager import DocumentManager

# Global instances
_agraph_instance: Optional[AGraph] = None
_document_manager: Optional[DocumentManager] = None


async def get_agraph_instance(project_name: Optional[str] = None) -> AGraph:
    """Get or create AGraph instance for a specific project."""
    global _agraph_instance  # pylint: disable=global-statement

    settings = get_settings()
    current_project = project_name or settings.current_project

    # Use project-specific paths if project is specified
    if current_project:
        project_paths = get_project_paths(current_project, settings.workdir)
        cache_dir = project_paths["cache"]
        persist_directory = project_paths["vector_db"]
        collection_name = f"agraph_{current_project}"
    else:
        # Use default paths for backward compatibility
        cache_dir = f"{settings.workdir}/cache"
        persist_directory = f"{settings.workdir}/agraph_vectordb"
        collection_name = "agraph_api"

    # Create new instance if none exists or if project changed
    if _agraph_instance is None or _agraph_instance.collection_name != collection_name:

        # Close existing instance if it exists
        if _agraph_instance is not None:
            await _agraph_instance.close()

        config = BuilderConfig(
            chunk_size=settings.text.max_chunk_size,
            chunk_overlap=settings.text.chunk_overlap,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            entity_confidence_threshold=0.7,
            relation_confidence_threshold=0.6,
            cache_dir=cache_dir,
        )

        _agraph_instance = AGraph(
            collection_name=collection_name,
            persist_directory=persist_directory,
            vector_store_type="chroma",
            config=config,
            use_openai_embeddings=True,
            enable_knowledge_graph=True,
        )

        await _agraph_instance.initialize()

    return _agraph_instance


def get_document_manager(project_name: Optional[str] = None) -> DocumentManager:
    """Get or create DocumentManager instance for a specific project."""
    global _document_manager  # pylint: disable=global-statement

    settings = get_settings()
    current_project = project_name or settings.current_project

    # Use project-specific paths if project is specified
    if current_project:
        project_paths = get_project_paths(current_project, settings.workdir)
        storage_dir = project_paths["document_storage"]
    else:
        # Use default path for backward compatibility
        storage_dir = f"{settings.workdir}/document_storage"

    # Create new instance if none exists or if project changed
    if _document_manager is None or _document_manager.project_name != current_project:

        _document_manager = DocumentManager(storage_dir, project_name=current_project)

    return _document_manager


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


async def close_agraph_instance() -> None:
    """Close the global AGraph instance."""
    global _agraph_instance  # pylint: disable=global-statement
    if _agraph_instance:
        await _agraph_instance.close()
        _agraph_instance = None


def reset_instances() -> None:
    """Reset all global instances (useful for project switching)."""
    global _agraph_instance, _document_manager  # pylint: disable=global-statement
    _agraph_instance = None
    _document_manager = None
