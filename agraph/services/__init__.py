"""
Graph Service Module

This module provides specialized service classes following the Single Responsibility Principle.
Each service encapsulates a specific domain of graph operations:

- GraphAnalyzer: Statistical analysis and metrics computation for knowledge graphs
- GraphPathFinder: Path finding algorithms and graph traversal operations
- EntityMerger: Entity deduplication and merging operations
- GraphValidator: Graph integrity validation and quality assessment

These services operate on KnowledgeGraph instances and provide high-level operations
for graph manipulation, analysis, and maintenance.
"""

from .entity_merger import EntityMerger
from .graph_analyzer import GraphAnalyzer
from .graph_path_finder import GraphPathFinder
from .graph_validator import GraphValidator

__all__ = [
    "GraphAnalyzer",
    "GraphPathFinder",
    "EntityMerger",
    "GraphValidator",
]
