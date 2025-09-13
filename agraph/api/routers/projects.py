"""Project management router with enhanced local caching."""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from ...base.infrastructure.instances import reset_instances
from ...config import (
    Settings,
    create_project,
    delete_project,
    get_current_project,
    get_project_paths,
    get_settings,
    list_projects,
    load_project_settings,
)
from ...logger import logger
from ..dependencies import ensure_project_directory, save_project_config_changes
from ..models import ProjectCreateRequest, ProjectDeleteRequest, ProjectListResponse, ProjectResponse, ResponseStatus

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/list", response_model=ProjectListResponse)
async def list_all_projects() -> ProjectListResponse:
    """List all available projects."""
    try:
        projects = list_projects()

        return ProjectListResponse(
            status=ResponseStatus.SUCCESS, message=f"Found {len(projects)} projects", projects=projects
        )
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/create", response_model=ProjectResponse)
async def create_new_project(request: ProjectCreateRequest) -> ProjectResponse:
    """Create a new project with complete Settings configuration saved locally."""
    try:
        # Create project with directory structure and configuration
        project_config = create_project(request.name, request.description)

        # Ensure directory structure is properly created
        project_paths = ensure_project_directory(request.name)

        # Create project settings based on current global settings
        base_settings = get_settings()
        project_settings = base_settings.model_copy(deep=True)
        project_settings.current_project = request.name

        # Save Settings configuration directly to config.json with secure permissions
        project_config_file = Path(project_paths["config_file"])

        project_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(project_config_file, "w", encoding="utf-8") as f:
            json.dump(project_settings.to_dict(), f, indent=2, ensure_ascii=False)

        # Set secure file permissions (readable only by owner)
        try:
            import stat

            project_config_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception as e:
            logger.warning(f"Could not set secure permissions on config file: {e}")

        config_path = str(project_config_file)

        logger.info(f"Created project: {request.name} with complete Settings at {config_path}")
        logger.info(f"Project config backup saved to: {project_config_file}")
        logger.info(f"Project paths: {project_paths}")

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Project '{request.name}' created successfully with complete Settings configuration",
            project_name=request.name,
            data={
                **project_config,
                "config_path": config_path,
                "backup_config_path": str(project_config_file),
                "project_paths": project_paths,
                "settings_saved": True,
                "complete_settings": project_settings.to_dict_safe(),
            },
        )
    except ValueError as e:
        logger.error(f"Invalid project creation request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# @router.get("/current", response_model=ProjectResponse)
# async def get_current_project_info() -> ProjectResponse:
#     """Get current active project information with local config details."""
#     try:
#         current_project = get_current_project()

#         if not current_project:
#             return ProjectResponse(
#                 status=ResponseStatus.SUCCESS,
#                 message="No current project set (using default workspace)",
#                 data={"current_project": None, "using_default": True},
#             )

#         project_paths = get_project_paths(current_project)
#         project_config_path = Path(project_paths["config_file"])

#         # Load project config.json (now directly contains Settings)
#         project_data: Dict[str, Any] = {"current_project": current_project, "paths": project_paths}

#         if project_config_path.exists():
#             with open(project_config_path, "r", encoding="utf-8") as f:
#                 settings_data = json.load(f)

#                 # config.json now directly contains Settings data
#                 project_data["settings_data"] = settings_data
#                 project_data["backup_available"] = True
#                 project_data["config_format"] = "direct_settings"
#         else:
#             project_data["backup_available"] = False

#         # Add current settings from local cache
#         try:
#             project_settings = load_project_settings(current_project)
#             project_data["current_settings"] = project_settings.to_dict_safe()
#             project_data["config_cache_status"] = "loaded_from_local"
#         except Exception as e:
#             logger.warning(f"Could not load project settings from cache: {e}")
#             project_data["config_cache_status"] = "error"

#             # Try to recover from backup if available
#             if project_data.get("backup_available") and "settings_data" in project_data:
#                 logger.info(f"Attempting to recover settings from backup for project: {current_project}")
#                 try:
#                     # Create Settings object from backup
#                     recovered_settings = Settings.model_validate(project_data["settings_data"])

#                     # Save recovered settings to cache
#                     config_path = await save_project_config_changes(current_project, recovered_settings)
#                     project_data["current_settings"] = recovered_settings.to_dict_safe()
#                     project_data["config_cache_status"] = "recovered_from_backup"
#                     project_data["recovery_log"] = f"Settings recovered from backup and saved to {config_path}"
#                     logger.info(f"Successfully recovered settings for project: {current_project}")
#                 except Exception as recovery_error:
#                     logger.error(f"Failed to recover settings from backup: {recovery_error}")
#                     project_data["recovery_error"] = str(recovery_error)

#         return ProjectResponse(
#             status=ResponseStatus.SUCCESS,
#             message=f"Current project: {current_project}",
#             project_name=current_project,
#             data=project_data,
#         )
#     except Exception as e:
#         logger.error(f"Failed to get current project: {e}")
#         raise HTTPException(status_code=500, detail=str(e)) from e


# @router.post("/switch", response_model=ProjectResponse)
# async def switch_project(request: ProjectSwitchRequest) -> ProjectResponse:
#     """Switch to a different project (or no project)."""
#     try:
#         # Reset instances to ensure clean switch
#         reset_instances()

#         new_settings = set_current_project(request.project_name)
#         logger.info(f"Switched to project: {request.project_name or 'default'}")

#         message = (
#             f"Switched to project: {request.project_name}"
#             if request.project_name
#             else "Switched to default workspace (no project)"
#         )

#         return ProjectResponse(
#             status=ResponseStatus.SUCCESS,
#             message=message,
#             project_name=request.project_name,
#             data={
#                 "previous_project": get_current_project(),
#                 "new_project": request.project_name,
#                 "settings": new_settings.to_dict_safe(),
#             },
#         )
#     except ValueError as e:
#         logger.error(f"Invalid project switch request: {e}")
#         raise HTTPException(status_code=400, detail=str(e)) from e
#     except Exception as e:
#         logger.error(f"Failed to switch project: {e}")
#         raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{project_name}", response_model=ProjectResponse)
async def get_project_info(project_name: str) -> ProjectResponse:
    """Get information about a specific project with local config details."""
    try:
        projects = list_projects()
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        project_paths = get_project_paths(project_name)
        project_config_path = Path(project_paths["config_file"])

        # Load project config.json (now directly contains Settings)
        project_data: Dict[str, Any] = {"project_name": project_name, "paths": project_paths}

        if project_config_path.exists():
            with open(project_config_path, "r", encoding="utf-8") as f:
                settings_data = json.load(f)

                # Mask sensitive data in settings_data before adding to response
                try:
                    temp_settings = Settings.model_validate(settings_data)
                    project_data["settings_data"] = temp_settings.to_dict_safe()
                except Exception:
                    # Fallback: if validation fails, just mark as available without exposing data
                    project_data["settings_data"] = {"masked": "Configuration available but not displayed for security"}

                project_data["backup_available"] = True
                project_data["config_format"] = "direct_settings"
        else:
            project_data["backup_available"] = False

        # Load current settings from local cache
        try:
            project_settings = load_project_settings(project_name)
            project_data["current_settings"] = project_settings.to_dict_safe()
            project_data["config_cache_status"] = "loaded_from_local"
            project_data["config_file_path"] = str(project_config_path)
        except Exception as e:
            logger.warning(f"Could not load project settings from cache: {e}")
            project_data["config_cache_status"] = "error"
            project_data["config_error"] = str(e)

            # Try to recover from backup if available
            if project_data.get("backup_available") and "settings_data" in project_data:
                logger.info(f"Attempting to recover settings from backup for project: {project_name}")
                try:
                    # Create Settings object from backup
                    recovered_settings = Settings.model_validate(project_data["settings_data"])

                    # Save recovered settings to cache
                    config_path = await save_project_config_changes(project_name, recovered_settings)
                    project_data["current_settings"] = recovered_settings.to_dict_safe()
                    project_data["config_cache_status"] = "recovered_from_backup"
                    project_data["recovery_log"] = f"Settings recovered from backup and saved to {config_path}"
                    logger.info(f"Successfully recovered settings for project: {project_name}")
                except Exception as recovery_error:
                    logger.error(f"Failed to recover settings from backup: {recovery_error}")
                    project_data["recovery_error"] = str(recovery_error)

        # Add directory statistics
        project_dir = Path(project_paths["project_dir"])
        doc_storage_dir = Path(project_paths["document_storage"])
        vector_db_dir = Path(project_paths["vector_db"])
        cache_dir = Path(project_paths["cache"])

        stats = {
            "project_exists": project_dir.exists(),
            "document_count": (
                len([f for f in doc_storage_dir.glob("**/*") if f.is_file()]) if doc_storage_dir.exists() else 0
            ),
            "has_vector_db": vector_db_dir.exists() and any(vector_db_dir.iterdir()),
            "cache_size": len(list(cache_dir.glob("**/*"))) if cache_dir.exists() else 0,
            "total_size_mb": (
                sum(f.stat().st_size for f in project_dir.rglob("*") if f.is_file()) // (1024 * 1024)
                if project_dir.exists()
                else 0
            ),
            "config_file_exists": project_config_path.exists(),
            "config_file_size": (project_config_path.stat().st_size if project_config_path.exists() else 0),
        }

        project_data["statistics"] = stats

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Project '{project_name}' information retrieved with local config",
            project_name=project_name,
            data=project_data,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/delete", response_model=ProjectResponse)
async def delete_project_endpoint(request: ProjectDeleteRequest) -> ProjectResponse:
    """Delete a project and all its data."""
    try:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="Project deletion requires confirmation (set confirm=true)")

        projects = list_projects()
        if request.project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found")

        # Get project info before deletion
        project_paths = get_project_paths(request.project_name)

        success = delete_project(request.project_name)
        if success:
            # Reset instances if we deleted the current project
            reset_instances()
            logger.info(f"Deleted project: {request.project_name}")

            return ProjectResponse(
                status=ResponseStatus.SUCCESS,
                message=f"Project '{request.project_name}' deleted successfully",
                project_name=request.project_name,
                data={"deleted_paths": project_paths, "current_project": get_current_project()},
            )

        raise HTTPException(
            status_code=404,
            detail=f"Project '{request.project_name}' not found or already deleted",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{project_name}/recover-settings", response_model=ProjectResponse)
async def recover_project_settings_from_backup(project_name: str) -> ProjectResponse:
    """Recover project settings from backup config.json file."""
    try:
        projects = list_projects()
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        project_paths = get_project_paths(project_name)
        project_config_path = Path(project_paths["config_file"])

        if not project_config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No backup configuration found for project '{project_name}'",
            )

        # Load backup configuration (config.json now directly contains Settings)
        with open(project_config_path, "r", encoding="utf-8") as f:
            settings_data = json.load(f)

        # Recover Settings from backup
        recovered_settings = Settings.model_validate(settings_data)
        recovered_settings.current_project = project_name

        # Save recovered settings to cache (this will also update the config.json)
        config_path = await save_project_config_changes(project_name, recovered_settings)

        logger.info(f"Successfully recovered settings for project '{project_name}' from backup")

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Settings successfully recovered from backup for project '{project_name}'",
            project_name=project_name,
            data={
                "recovery_source": str(project_config_path),
                "config_cache_path": config_path,
                "recovered_settings": recovered_settings.to_dict_safe(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recover settings from backup for project '{project_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/", response_model=ProjectListResponse)
async def get_projects_overview(
    include_stats: bool = Query(default=False, description="Include project statistics")
) -> ProjectListResponse:
    """Get overview of all projects with optional statistics."""
    try:
        projects = list_projects()
        current_project = get_current_project()

        data: Dict[str, Any] = {
            "current_project": current_project,
            "total_count": len(projects),
            "projects": projects,
        }

        if include_stats:
            project_stats: Dict[str, Dict[str, Any]] = {}
            for project_name in projects:
                try:
                    project_paths = get_project_paths(project_name)
                    project_dir = Path(project_paths["project_dir"])
                    doc_storage_dir = Path(project_paths["document_storage"])
                    vector_db_dir = Path(project_paths["vector_db"])

                    stats: Dict[str, Any] = {
                        "document_count": (
                            len([f for f in doc_storage_dir.glob("**/*") if f.is_file()])
                            if doc_storage_dir.exists()
                            else 0
                        ),
                        "has_vector_db": vector_db_dir.exists() and any(vector_db_dir.iterdir()),
                        "size_mb": (
                            sum(f.stat().st_size for f in project_dir.rglob("*") if f.is_file()) // (1024 * 1024)
                            if project_dir.exists()
                            else 0
                        ),
                    }
                    project_stats[project_name] = stats
                except Exception as e:
                    logger.warning(f"Failed to get stats for project {project_name}: {e}")
                    project_stats[project_name] = {"error": str(e)}

            data["project_statistics"] = project_stats

        return ProjectListResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Projects overview retrieved (include_stats={include_stats})",
            projects=projects,
            data=data,
        )
    except Exception as e:
        logger.error(f"Failed to get projects overview: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
