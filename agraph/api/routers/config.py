"""Configuration management router."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ...config import (
    copy_project_settings,
    get_config_file_path,
    get_project_info,
    get_settings,
    load_project_settings,
    load_settings_from_file,
    reset_project_settings,
    reset_settings,
    save_settings_to_file,
    update_project_settings,
    update_settings,
)
from ...logger import logger
from ..dependencies import get_agraph_instance
from ..models import (
    ConfigFileRequest,
    ConfigFileResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ResponseStatus,
)

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("", response_model=ConfigResponse)
async def get_config(
    project_name: Optional[str] = Query(
        default=None, description="Project name for project-specific config"
    )
) -> ConfigResponse:
    """Get current application configuration."""
    try:
        if project_name:
            settings = load_project_settings(project_name)
            project_info = get_project_info(project_name)
            config_data: Dict[str, Any] = {
                "project_info": project_info,
                "settings": settings.to_dict(),
            }
        else:
            settings = get_settings()
            config_data = settings.to_dict()

        # Add runtime information if AGraph instance is available
        try:
            agraph = await get_agraph_instance()
            config_data["runtime_info"] = {
                "collection_name": agraph.collection_name,
                "persist_directory": agraph.persist_directory,
                "vector_store_type": agraph.vector_store_type,
                "use_openai_embeddings": agraph.use_openai_embeddings,
                "enable_knowledge_graph": agraph.enable_knowledge_graph,
                "is_initialized": agraph.is_initialized,
                "has_knowledge_graph": agraph.has_knowledge_graph,
                "builder_config": agraph.config.to_dict() if agraph.config else None,
            }
            await agraph.close()
        except Exception as runtime_error:
            logger.warning(f"Could not get runtime info: {runtime_error}")
            config_data["runtime_info"] = None

        return ConfigResponse(
            status=ResponseStatus.SUCCESS,
            message="Configuration retrieved successfully",
            data=config_data,
        )
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=ConfigResponse)
async def update_config(
    request: ConfigUpdateRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for project-specific config"
    ),
) -> ConfigResponse:
    """Update application configuration."""
    try:
        # Convert request to update dictionary
        updates: Dict[str, Any] = {}

        # Handle direct settings updates
        if request.workdir is not None:
            updates["workdir"] = request.workdir

        # Handle OpenAI updates
        openai_updates: Dict[str, str] = {}
        if request.openai_api_key is not None:
            openai_updates["api_key"] = request.openai_api_key
        if request.openai_api_base is not None:
            openai_updates["api_base"] = request.openai_api_base
        if openai_updates:
            updates["openai"] = openai_updates

        # Handle LLM updates
        llm_updates: Dict[str, Any] = {}
        if request.llm_model is not None:
            llm_updates["model"] = request.llm_model
        if request.llm_temperature is not None:
            llm_updates["temperature"] = request.llm_temperature
        if request.llm_max_tokens is not None:
            llm_updates["max_tokens"] = request.llm_max_tokens
        if request.llm_provider is not None:
            llm_updates["provider"] = request.llm_provider
        if llm_updates:
            updates["llm"] = llm_updates

        # Handle embedding updates
        embedding_updates: Dict[str, Any] = {}
        if request.embedding_model is not None:
            embedding_updates["model"] = request.embedding_model
        if request.embedding_provider is not None:
            embedding_updates["provider"] = request.embedding_provider
        if request.embedding_dimension is not None:
            embedding_updates["dimension"] = request.embedding_dimension
        if request.embedding_max_token_size is not None:
            embedding_updates["max_token_size"] = request.embedding_max_token_size
        if request.embedding_batch_size is not None:
            embedding_updates["batch_size"] = request.embedding_batch_size
        if embedding_updates:
            updates["embedding"] = embedding_updates

        # Handle graph updates
        graph_updates: Dict[str, Any] = {}
        if request.entity_types is not None:
            graph_updates["entity_types"] = request.entity_types
        if request.relation_types is not None:
            graph_updates["relation_types"] = request.relation_types
        if graph_updates:
            updates["graph"] = graph_updates

        # Handle text processing updates
        text_updates: Dict[str, int] = {}
        if request.max_chunk_size is not None:
            text_updates["max_chunk_size"] = request.max_chunk_size
        elif request.chunk_size is not None:  # Legacy support
            text_updates["max_chunk_size"] = request.chunk_size
        if request.chunk_overlap is not None:
            text_updates["chunk_overlap"] = request.chunk_overlap
        if text_updates:
            updates["text"] = text_updates

        # Handle RAG updates
        rag_updates: Dict[str, str] = {}
        if request.system_prompt is not None:
            rag_updates["system_prompt"] = request.system_prompt
        if rag_updates:
            updates["rag"] = rag_updates

        # Apply updates if any
        if updates:
            if project_name:
                new_settings = update_project_settings(project_name, updates)
                logger.info(
                    f"Project '{project_name}' configuration updated with {len(updates)} changes"
                )
                message = (
                    f"Project configuration updated successfully. {len(updates)} sections modified."
                )
            else:
                new_settings = update_settings(updates)
                logger.info(f"Global configuration updated with {len(updates)} changes")
                message = f"Configuration updated successfully. {len(updates)} sections modified."

            return ConfigResponse(
                status=ResponseStatus.SUCCESS,
                message=message,
                data=new_settings.to_dict(),
            )

        if project_name:
            settings = load_project_settings(project_name)
            message = "No project configuration changes provided"
        else:
            settings = get_settings()
            message = "No configuration changes provided"

        return ConfigResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=settings.to_dict(),
        )

    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reset", response_model=ConfigResponse)
async def reset_config(
    project_name: Optional[str] = Query(
        default=None, description="Project name for project-specific config reset"
    )
) -> ConfigResponse:
    """Reset configuration to default values."""
    try:
        if project_name:
            new_settings = reset_project_settings(project_name)
            logger.info(f"Project '{project_name}' configuration reset to default values")
            message = "Project configuration reset to default values successfully"
        else:
            new_settings = reset_settings()
            logger.info("Global configuration reset to default values")
            message = "Configuration reset to default values successfully"

        return ConfigResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=new_settings.to_dict(),
        )
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/save", response_model=ConfigFileResponse)
async def save_config_to_file(
    request: ConfigFileRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for project-specific config save"
    ),
) -> ConfigFileResponse:
    """Save current configuration to file."""
    try:
        file_path = save_settings_to_file(request.file_path, project_name=project_name)

        if project_name:
            settings = load_project_settings(project_name)
            logger.info(f"Project '{project_name}' configuration saved to: {file_path}")
            message = "Project configuration saved successfully"
        else:
            settings = get_settings()
            logger.info(f"Global configuration saved to: {file_path}")
            message = "Configuration saved successfully"

        return ConfigFileResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=settings.to_dict(),
            file_path=file_path,
        )
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/load", response_model=ConfigFileResponse)
async def load_config_from_file(
    request: ConfigFileRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for project-specific config load"
    ),
) -> ConfigFileResponse:
    """Load configuration from file."""
    try:
        new_settings = load_settings_from_file(request.file_path, project_name=project_name)

        if project_name:
            file_path = request.file_path or get_config_file_path(project_name=project_name)
            logger.info(f"Project '{project_name}' configuration loaded from: {file_path}")
            message = "Project configuration loaded successfully"
        else:
            file_path = request.file_path or get_config_file_path()
            logger.info(f"Global configuration loaded from: {file_path}")
            message = "Configuration loaded successfully"

        return ConfigFileResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=new_settings.to_dict(),
            file_path=file_path,
        )
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/file-path", response_model=ConfigFileResponse)
async def get_config_file_path_info(
    project_name: Optional[str] = Query(
        default=None, description="Project name for project-specific config file path"
    )
) -> ConfigFileResponse:
    """Get the default configuration file path."""
    try:
        file_path = get_config_file_path(project_name=project_name)
        file_exists = Path(file_path).exists()

        if project_name:
            message = f"Project configuration file path retrieved (exists: {file_exists})"
        else:
            message = f"Configuration file path retrieved (exists: {file_exists})"

        return ConfigFileResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data={"exists": file_exists, "writable": Path(file_path).parent.exists()},
            file_path=file_path,
        )
    except Exception as e:
        logger.error(f"Failed to get config file path: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/projects/{project_name}/copy-from/{source_project}", response_model=ConfigResponse)
async def copy_project_config(project_name: str, source_project: str) -> ConfigResponse:
    """Copy configuration from one project to another."""
    try:
        new_settings = copy_project_settings(source_project, project_name)
        logger.info(f"Configuration copied from '{source_project}' to '{project_name}'")

        return ConfigResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Configuration successfully copied from '{source_project}' to '{project_name}'",
            data=new_settings.to_dict(),
        )
    except FileNotFoundError as e:
        logger.error(f"Source project configuration not found: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to copy project configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/projects/{project_name}/info")
async def get_project_config_info(project_name: str) -> Any:
    """Get project configuration information."""
    try:
        project_info = get_project_info(project_name)
        logger.info(f"Project '{project_name}' configuration info retrieved")

        return {
            "status": ResponseStatus.SUCCESS,
            "message": "Project configuration info retrieved successfully",
            "data": project_info,
        }
    except FileNotFoundError as e:
        logger.error(f"Project configuration not found: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get project configuration info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
