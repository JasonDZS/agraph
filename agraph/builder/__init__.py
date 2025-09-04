"""
KnowledgeGraph Builder module.
"""

from ..config import BuilderConfig, BuildStatus, BuildSteps, CacheMetadata
from .builder_v2 import KnowledgeGraphBuilderV2 as KnowledgeGraphBuilder

# Legacy compatibility
from .compatibility import LegacyKnowledgeGraphBuilder

__all__ = [
    "BuilderConfig", 
    "CacheMetadata", 
    "BuildStatus", 
    "BuildSteps", 
    "KnowledgeGraphBuilder",  # Pipeline version (recommended)
    "LegacyKnowledgeGraphBuilder"  # Legacy compatibility wrapper
]
