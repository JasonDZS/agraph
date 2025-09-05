"""FastAPI application for AGraph."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..logger import logger
from .dependencies import close_agraph_instance, get_agraph_instance
from .models import ErrorResponse, HealthResponse, ResponseStatus
from .routers import (
    cache_router,
    chat_router,
    config_router,
    documents_router,
    knowledge_graph_router,
    projects_router,
    search_router,
    system_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    # Startup
    logger.info("Starting AGraph API server...")
    await get_agraph_instance()

    yield

    # Shutdown
    logger.info("Shutting down AGraph API server...")
    await close_agraph_instance()


# Create FastAPI app
app = FastAPI(
    title="AGraph API",
    description="Knowledge Graph Construction and RAG System API",
    version="1.0.0",
    lifespan=lifespan,
)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Log HTTP requests."""
    start_time = time.time()

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Process request
    response = await call_next(request)

    # Calculate response time
    process_time = time.time() - start_time

    # Log request
    logger.info(f'{client_ip} - "{request.method} {request.url.path}" ' f"{response.status_code} - {process_time:.3f}s")

    return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_router)
app.include_router(projects_router)
app.include_router(documents_router)
app.include_router(knowledge_graph_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(cache_router)
app.include_router(system_router)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    error_response = ErrorResponse(status=ResponseStatus.ERROR, message=exc.detail, error_code=str(exc.status_code))
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(error_response))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        status=ResponseStatus.ERROR,
        message="An unexpected error occurred",
        error_details={"exception": str(exc)},
    )
    return JSONResponse(status_code=500, content=jsonable_encoder(error_response))
