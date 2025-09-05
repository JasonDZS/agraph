"""
AGraph Knowledge Graph Builder module.

This module provides the high-performance pipeline-based KnowledgeGraphBuilder
for building knowledge graphs from text and documents.

Features:
- 10-100x performance improvements with intelligent caching
- Modular pipeline architecture with error recovery
- Unified architecture with advanced indexing
- Async/await support for scalable processing
"""

from ..config import BuilderConfig, BuildStatus, BuildSteps, CacheMetadata
from .builder import KnowledgeGraphBuilder

__all__ = [
    # Configuration
    "BuilderConfig",
    "CacheMetadata",
    "BuildStatus",
    "BuildSteps",
    # Core Builder - Pipeline Architecture
    "KnowledgeGraphBuilder",  # High-performance pipeline-based implementation
]
