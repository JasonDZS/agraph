"""Project management router."""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from ...config import (
    create_project,
    delete_project,
    get_current_project,
    get_project_paths,
    list_projects,
    set_current_project,
)
from ...logger import logger
from ..dependencies import reset_instances
from ..models import (
    ProjectCreateRequest,
    ProjectDeleteRequest,
    ProjectListResponse,
    ProjectResponse,
    ProjectSwitchRequest,
    ResponseStatus,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/list", response_model=ProjectListResponse)
async def list_all_projects() -> ProjectListResponse:
    """List all available projects."""
    try:
        projects = list_projects()
        current_project = get_current_project()

        return ProjectListResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Found {len(projects)} projects",
            projects=projects,
            data={
                "current_project": current_project,
                "total_count": len(projects),
                "projects": projects,
            },
        )
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/create", response_model=ProjectResponse)
async def create_new_project(request: ProjectCreateRequest) -> ProjectResponse:
    """Create a new project."""
    try:
        project_config = create_project(request.name, request.description)
        logger.info(f"Created project: {request.name}")

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Project '{request.name}' created successfully",
            project_name=request.name,
            data=project_config,
        )
    except ValueError as e:
        logger.error(f"Invalid project creation request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/current", response_model=ProjectResponse)
async def get_current_project_info() -> ProjectResponse:
    """Get current active project information."""
    try:
        current_project = get_current_project()

        if not current_project:
            return ProjectResponse(
                status=ResponseStatus.SUCCESS,
                message="No current project set (using default workspace)",
                data={"current_project": None, "using_default": True},
            )

        project_paths = get_project_paths(current_project)
        project_config_path = Path(project_paths["config_file"])

        # Load project config if it exists
        project_data = {"current_project": current_project, "paths": project_paths}

        if project_config_path.exists():

            with open(project_config_path, "r", encoding="utf-8") as f:
                project_config = json.load(f)
                project_data.update(project_config)

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Current project: {current_project}",
            project_name=current_project,
            data=project_data,
        )
    except Exception as e:
        logger.error(f"Failed to get current project: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/switch", response_model=ProjectResponse)
async def switch_project(request: ProjectSwitchRequest) -> ProjectResponse:
    """Switch to a different project (or no project)."""
    try:
        # Reset instances to ensure clean switch
        reset_instances()

        new_settings = set_current_project(request.project_name)
        logger.info(f"Switched to project: {request.project_name or 'default'}")

        message = (
            f"Switched to project: {request.project_name}"
            if request.project_name
            else "Switched to default workspace (no project)"
        )

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            project_name=request.project_name,
            data={
                "previous_project": get_current_project(),
                "new_project": request.project_name,
                "settings": new_settings.to_dict(),
            },
        )
    except ValueError as e:
        logger.error(f"Invalid project switch request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to switch project: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{project_name}", response_model=ProjectResponse)
async def get_project_info(project_name: str) -> ProjectResponse:
    """Get information about a specific project."""
    try:
        projects = list_projects()
        if project_name not in projects:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

        project_paths = get_project_paths(project_name)
        project_config_path = Path(project_paths["config_file"])

        # Load project config
        project_data = {"project_name": project_name, "paths": project_paths}

        if project_config_path.exists():

            with open(project_config_path, "r", encoding="utf-8") as f:
                project_config = json.load(f)
                project_data.update(project_config)

        # Add directory statistics
        project_dir = Path(project_paths["project_dir"])
        doc_storage_dir = Path(project_paths["document_storage"])
        vector_db_dir = Path(project_paths["vector_db"])
        cache_dir = Path(project_paths["cache"])

        stats = {
            "project_exists": project_dir.exists(),
            "document_count": (
                len(list(doc_storage_dir.glob("**/*.txt"))) if doc_storage_dir.exists() else 0
            ),
            "has_vector_db": vector_db_dir.exists() and any(vector_db_dir.iterdir()),
            "cache_size": len(list(cache_dir.glob("**/*"))) if cache_dir.exists() else 0,
            "total_size_mb": (
                sum(f.stat().st_size for f in project_dir.rglob("*") if f.is_file())
                // (1024 * 1024)
                if project_dir.exists()
                else 0
            ),
        }

        project_data["statistics"] = stats

        return ProjectResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Project '{project_name}' information retrieved",
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
            raise HTTPException(
                status_code=400, detail="Project deletion requires confirmation (set confirm=true)"
            )

        projects = list_projects()
        if request.project_name not in projects:
            raise HTTPException(
                status_code=404, detail=f"Project '{request.project_name}' not found"
            )

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
                            len(list(doc_storage_dir.glob("**/*")))
                            if doc_storage_dir.exists()
                            else 0
                        ),
                        "has_vector_db": vector_db_dir.exists() and any(vector_db_dir.iterdir()),
                        "size_mb": (
                            sum(f.stat().st_size for f in project_dir.rglob("*") if f.is_file())
                            // (1024 * 1024)
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
