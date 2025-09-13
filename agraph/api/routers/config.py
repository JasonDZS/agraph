"""Configuration management router with project-based local caching."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ...config import (
    Settings,
    copy_project_settings,
    get_config_file_path,
    get_project_info,
    get_project_paths,
    get_settings,
    list_projects,
    load_project_settings,
    load_settings_from_file,
    reset_project_settings,
    reset_settings,
    save_settings_to_file,
    update_settings,
)
from ...logger import logger
from ..dependencies import get_agraph_instance, get_project_config_path, save_project_config_changes
from ..models import ConfigFileRequest, ConfigFileResponse, ConfigResponse, ConfigUpdateRequest, ResponseStatus

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("", response_model=ConfigResponse)
async def get_config(
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific config")
) -> ConfigResponse:
    """Get current application configuration with local caching support."""
    try:
        if project_name:
            settings = load_project_settings(project_name)
            project_info = get_project_info(project_name)
            config_file_path = get_project_config_path(project_name)

            config_data: Dict[str, Any] = {
                "project_info": project_info,
                "settings": settings.to_dict_safe(),
                "config_source": "project_local_cache",
                "config_file_path": config_file_path,
                "is_project_specific": True,
            }
        else:
            settings = get_settings()
            config_data = settings.to_dict_safe()
            config_data["config_source"] = "global_settings"
            config_data["is_project_specific"] = False

        # Add runtime information if AGraph instance is available
        try:
            agraph = await get_agraph_instance(project_name)
            config_data["runtime_info"] = {
                "collection_name": agraph.collection_name,
                "persist_directory": agraph.persist_directory,
                "vector_store_type": agraph.vector_store_type,
                "use_openai_embeddings": agraph.use_openai_embeddings,
                "enable_knowledge_graph": agraph.enable_knowledge_graph,
                "is_initialized": agraph.is_initialized,
                "has_knowledge_graph": agraph.has_knowledge_graph,
                "builder_config": agraph.config.to_dict_safe() if agraph.config else None,
            }
            # Don't close the instance as it's cached
        except Exception as runtime_error:
            logger.warning(f"Could not get runtime info: {runtime_error}")
            config_data["runtime_info"] = None

        return ConfigResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Configuration retrieved successfully from {'project cache' if project_name else 'global settings'}",
            data=config_data,
        )
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("", response_model=ConfigResponse)
async def update_config(
    request: ConfigUpdateRequest,
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific config"),
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

        # Handle builder updates (unified via Settings.builder)
        builder_updates: Dict[str, Any] = {}
        if request.builder_enable_cache is not None:
            builder_updates["enable_cache"] = request.builder_enable_cache
        if request.builder_cache_dir is not None:
            builder_updates["cache_dir"] = request.builder_cache_dir
        if request.builder_cache_ttl is not None:
            builder_updates["cache_ttl"] = request.builder_cache_ttl
        if request.builder_auto_cleanup is not None:
            builder_updates["auto_cleanup"] = request.builder_auto_cleanup
        if request.builder_chunk_size is not None:
            builder_updates["chunk_size"] = request.builder_chunk_size
        if request.builder_chunk_overlap is not None:
            builder_updates["chunk_overlap"] = request.builder_chunk_overlap
        if request.builder_entity_confidence_threshold is not None:
            builder_updates["entity_confidence_threshold"] = request.builder_entity_confidence_threshold
        if request.builder_relation_confidence_threshold is not None:
            builder_updates["relation_confidence_threshold"] = request.builder_relation_confidence_threshold
        if request.builder_cluster_algorithm is not None:
            builder_updates["cluster_algorithm"] = request.builder_cluster_algorithm
        if request.builder_min_cluster_size is not None:
            builder_updates["min_cluster_size"] = request.builder_min_cluster_size
        if request.builder_enable_user_interaction is not None:
            builder_updates["enable_user_interaction"] = request.builder_enable_user_interaction
        if request.builder_auto_save_edits is not None:
            builder_updates["auto_save_edits"] = request.builder_auto_save_edits
        if builder_updates:
            updates["builder"] = builder_updates

        # Handle legacy field mappings for backward compatibility
        if request.entity_confidence_threshold is not None and "builder" not in updates:
            updates["builder"] = {"entity_confidence_threshold": request.entity_confidence_threshold}
        elif request.entity_confidence_threshold is not None and "builder" in updates:
            updates["builder"]["entity_confidence_threshold"] = request.entity_confidence_threshold

        if request.relation_confidence_threshold is not None and "builder" not in updates:
            updates["builder"] = {"relation_confidence_threshold": request.relation_confidence_threshold}
        elif request.relation_confidence_threshold is not None and "builder" in updates:
            updates["builder"]["relation_confidence_threshold"] = request.relation_confidence_threshold

        # Apply updates if any
        if updates:
            if project_name:
                # Use project-specific local cache update
                current_settings = load_project_settings(project_name)
                new_settings = current_settings.update_from_dict(updates)
                new_settings.current_project = project_name

                # Save to local cache and update config.json (now unified)
                config_path = await save_project_config_changes(project_name, new_settings)

                logger.info(
                    f"Project '{project_name}' configuration updated with {len(updates)} changes and saved to {config_path}"
                )
                message = f"Project configuration updated successfully and saved to local cache. {len(updates)} sections modified."

                return ConfigResponse(
                    status=ResponseStatus.SUCCESS,
                    message=message,
                    data={
                        **new_settings.to_dict_safe(),
                        "config_saved_to": config_path,
                        "config_source": "project_local_cache",
                    },
                )
            new_settings = update_settings(updates)
            logger.info(f"Global configuration updated with {len(updates)} changes")
            message = f"Configuration updated successfully. {len(updates)} sections modified."

            return ConfigResponse(
                status=ResponseStatus.SUCCESS,
                message=message,
                data=new_settings.to_dict_safe(),
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
            data=settings.to_dict_safe(),
        )

    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reset", response_model=ConfigResponse)
async def reset_config(
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific config reset")
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
            data=new_settings.to_dict_safe(),
        )
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/save", response_model=ConfigFileResponse)
async def save_config_to_file(
    request: ConfigFileRequest,
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific config save"),
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
            data=settings.to_dict_safe(),
            file_path=file_path,
        )
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/load", response_model=ConfigFileResponse)
async def load_config_from_file(
    request: ConfigFileRequest,
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific config load"),
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
            data=new_settings.to_dict_safe(),
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
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific config file path")
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
            data=new_settings.to_dict_safe(),
        )
    except FileNotFoundError as e:
        logger.error(f"Source project configuration not found: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to copy project configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/projects/{project_name}/load-from-backup", response_model=ConfigResponse)
async def load_project_config_from_backup(project_name: str) -> ConfigResponse:
    """Load and apply complete project configuration from backup config.json file."""
    try:
        # Check if project exists in the project list
        projects = list_projects()
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        # Get project paths and backup file
        project_paths = get_project_paths(project_name)
        project_config_file = Path(project_paths["config_file"])

        if not project_config_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No backup configuration file found for project '{project_name}'",
            )

        # Load backup configuration (config.json now directly contains Settings)
        with open(project_config_file, "r", encoding="utf-8") as f:
            settings_data = json.load(f)

        # Create Settings object from backup
        loaded_settings = Settings.model_validate(settings_data)
        loaded_settings.current_project = project_name

        # Save loaded settings to cache (this updates the active configuration)
        config_path = await save_project_config_changes(project_name, loaded_settings)

        logger.info(f"Project '{project_name}' configuration loaded from backup and applied")

        return ConfigResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Configuration successfully loaded from backup for project '{project_name}'",
            data={
                **loaded_settings.to_dict_safe(),
                "loaded_from": str(project_config_file),
                "config_saved_to": config_path,
                "config_source": "backup_file",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration from backup for project '{project_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/projects/{project_name}/backup-status")
async def get_project_backup_status(project_name: str) -> Any:
    """Get project backup configuration status."""
    try:
        # Check if project exists
        projects = list_projects()
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        project_paths = get_project_paths(project_name)
        project_config_file = Path(project_paths["config_file"])

        status_info = {
            "project_name": project_name,
            "backup_file_path": str(project_config_file),
            "backup_exists": project_config_file.exists(),
        }

        if project_config_file.exists():
            try:
                with open(project_config_file, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)

                # config.json now directly contains Settings
                status_info.update(
                    {
                        "has_complete_settings": True,  # Always true if file exists and is valid
                        "file_size_kb": round(project_config_file.stat().st_size / 1024, 2),
                        "settings_sections": list(settings_data.keys()),
                        "settings_count": len(settings_data),
                    }
                )

                # Check for key configurations
                status_info["config_summary"] = {
                    "has_openai_config": "openai" in settings_data,
                    "has_llm_config": "llm" in settings_data,
                    "has_builder_config": "builder" in settings_data,
                    "current_project": settings_data.get("current_project"),
                    "workdir": settings_data.get("workdir"),
                }

            except Exception as parse_error:
                status_info["parse_error"] = str(parse_error)
                status_info["is_valid"] = False
        else:
            status_info["is_valid"] = False
            status_info["reason"] = "Backup file does not exist"

        return {
            "status": ResponseStatus.SUCCESS,
            "message": f"Project '{project_name}' backup status retrieved",
            "data": status_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project backup status: {e}")
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
