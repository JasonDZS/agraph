"""API routers for AGraph."""

from .cache import router as cache_router
from .chat import router as chat_router
from .config import router as config_router
from .documents import router as documents_router
from .knowledge_graph import router as knowledge_graph_router
from .projects import router as projects_router
from .search import router as search_router
from .system import router as system_router

__all__ = [
    "cache_router",
    "chat_router",
    "config_router",
    "documents_router",
    "knowledge_graph_router",
    "projects_router",
    "search_router",
    "system_router",
]
