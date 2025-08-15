"""System management router."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...logger import logger
from ..dependencies import get_agraph_instance_dependency
from ..models import (
    BaseResponse,
    BuildStatusResponse,
    CacheInfoResponse,
    ClearCacheRequest,
    ResponseStatus,
    StatsResponse,
)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(agraph: Any = Depends(get_agraph_instance_dependency)) -> StatsResponse:
    """Get system statistics."""
    try:
        stats = await agraph.get_stats()

        return StatsResponse(
            status=ResponseStatus.SUCCESS, message="Statistics retrieved successfully", data=stats
        )

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/build-status", response_model=BuildStatusResponse)
async def get_build_status(
    agraph: Any = Depends(get_agraph_instance_dependency),
) -> BuildStatusResponse:
    """Get build status."""
    try:
        if agraph.builder:
            build_status = agraph.builder.get_build_status()
            return BuildStatusResponse(
                status=ResponseStatus.SUCCESS,
                message="Build status retrieved successfully",
                data=build_status,
            )

        return BuildStatusResponse(
            status=ResponseStatus.SUCCESS,
            message="Builder not initialized",
            data={"status": "not_initialized"},
        )

    except Exception as e:
        logger.error(f"Failed to get build status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/cache-info", response_model=CacheInfoResponse)
async def get_cache_info(
    agraph: Any = Depends(get_agraph_instance_dependency),
) -> CacheInfoResponse:
    """Get cache information."""
    try:
        if agraph.builder:
            cache_info = agraph.builder.get_cache_info()
            return CacheInfoResponse(
                status=ResponseStatus.SUCCESS,
                message="Cache info retrieved successfully",
                data=cache_info,
            )
        return CacheInfoResponse(
            status=ResponseStatus.SUCCESS,
            message="Builder not initialized",
            data={"cache": "not_available"},
        )

    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/clear-cache", response_model=BaseResponse)
async def clear_cache(
    request: ClearCacheRequest, agraph: Any = Depends(get_agraph_instance_dependency)
) -> BaseResponse:
    """Clear cache."""
    try:
        if agraph.builder:
            agraph.builder.clear_cache(from_step=request.from_step)
            return BaseResponse(status=ResponseStatus.SUCCESS, message="Cache cleared successfully")

        return BaseResponse(
            status=ResponseStatus.SUCCESS, message="Builder not initialized, no cache to clear"
        )

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/clear-all", response_model=BaseResponse)
async def clear_all_data(agraph: Any = Depends(get_agraph_instance_dependency)) -> BaseResponse:
    """Clear all system data."""
    try:
        success = await agraph.clear_all()

        if success:
            return BaseResponse(
                status=ResponseStatus.SUCCESS, message="All data cleared successfully"
            )

        return BaseResponse(status=ResponseStatus.ERROR, message="Failed to clear all data")

    except Exception as e:
        logger.error(f"Failed to clear all data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
