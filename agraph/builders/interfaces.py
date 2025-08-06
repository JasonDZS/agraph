"""
Builder interfaces following Interface Segregation Principle

Each interface has a single, focused responsibility:
- GraphBuilder: Core graph building functionality
- GraphUpdater: Graph update operations
- GraphMerger: Graph merging capabilities
- GraphValidator: Graph validation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..relations import Relation


class GraphBuilder(ABC):
    """Core interface for graph building - Single responsibility: build graphs"""

    @abstractmethod
    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "agraph",
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from data sources

        Args:
            texts: Text documents to process
            database_schema: Database schema information
            graph_name: Name for the resulting graph

        Returns:
            KnowledgeGraph: Built knowledge graph
        """
        pass


class GraphUpdater(ABC):
    """Interface for updating existing graphs - Single responsibility: update graphs"""

    @abstractmethod
    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """
        Update existing knowledge graph with new data

        Args:
            graph: Existing knowledge graph
            new_entities: New entities to add
            new_relations: New relations to add

        Returns:
            KnowledgeGraph: Updated knowledge graph
        """
        pass


class GraphMerger(ABC):
    """Interface for merging multiple graphs - Single responsibility: merge graphs"""

    @abstractmethod
    async def merge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Merge multiple knowledge graphs into one

        Args:
            graphs: List of graphs to merge

        Returns:
            KnowledgeGraph: Merged knowledge graph
        """
        pass


class GraphValidator(ABC):
    """Interface for graph validation - Single responsibility: validate graphs"""

    @abstractmethod
    async def validate_graph(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Validate knowledge graph quality and consistency

        Args:
            graph: Knowledge graph to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        pass


class IncrementalBuilder(ABC):
    """Interface for incremental graph building - Single responsibility: incremental updates"""

    @abstractmethod
    async def add_documents(self, documents: List[str], document_ids: Optional[List[str]] = None) -> KnowledgeGraph:
        """
        Add new documents to existing graph

        Args:
            documents: New documents to process
            document_ids: Optional document IDs for tracking

        Returns:
            KnowledgeGraph: Updated graph
        """
        pass

    @abstractmethod
    async def remove_documents(self, document_ids: List[str]) -> KnowledgeGraph:
        """
        Remove documents from existing graph

        Args:
            document_ids: Document IDs to remove

        Returns:
            KnowledgeGraph: Updated graph
        """
        pass


class GraphExporter(ABC):
    """Interface for graph export functionality - Single responsibility: export graphs"""

    @abstractmethod
    async def export_to_format(self, graph: KnowledgeGraph, format: str) -> Dict[str, Any]:
        """
        Export graph to specified format

        Args:
            graph: Knowledge graph to export
            format: Target format (json, graphml, etc.)

        Returns:
            Dict[str, Any]: Exported data
        """
        pass


# Composition interfaces for different use cases


class BasicGraphBuilder(GraphBuilder):
    """Basic graph builder interface - only core building functionality"""

    pass


class UpdatableGraphBuilder(GraphBuilder, GraphUpdater):
    """Graph builder that supports updates"""

    pass


class FullFeaturedGraphBuilder(GraphBuilder, GraphUpdater, GraphMerger, GraphValidator):
    """Full-featured graph builder with all capabilities"""

    pass


class StreamingGraphBuilder(GraphBuilder, IncrementalBuilder):
    """Graph builder optimized for streaming/incremental updates"""

    pass


class BatchGraphBuilder(GraphBuilder, GraphMerger):
    """Graph builder optimized for batch processing multiple sources"""

    pass


class ReadOnlyGraphBuilder(GraphValidator, GraphExporter):
    """Read-only interface for graph analysis and export"""

    pass
