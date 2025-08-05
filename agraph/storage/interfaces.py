"""
Storage interfaces following Single Responsibility Principle

Each interface has a single, well-defined responsibility:
- GraphConnection: Connection management
- GraphCRUD: Basic CRUD operations
- GraphQuery: Query operations
- GraphBackup: Backup and restore operations
- GraphExport: Export functionality
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..relations import Relation
from ..types import RelationType


class GraphConnection(ABC):
    """Interface for storage connection management"""

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to storage backend

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from storage backend"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to storage backend

        Returns:
            bool: True if connected
        """
        pass


class GraphCRUD(ABC):
    """Interface for basic graph CRUD operations"""

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """
        Save knowledge graph

        Args:
            graph: Knowledge graph to save

        Returns:
            bool: True if save successful
        """
        pass

    @abstractmethod
    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """
        Load knowledge graph

        Args:
            graph_id: Graph ID

        Returns:
            KnowledgeGraph: Loaded graph, None if not found
        """
        pass

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete knowledge graph

        Args:
            graph_id: Graph ID

        Returns:
            bool: True if delete successful
        """
        pass

    @abstractmethod
    def list_graphs(self) -> List[Dict[str, Any]]:
        """
        List all graphs

        Returns:
            List[Dict[str, Any]]: Graph metadata list
        """
        pass


class GraphEntityCRUD(ABC):
    """Interface for entity CRUD operations"""

    @abstractmethod
    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Add entity to graph

        Args:
            graph_id: Graph ID
            entity: Entity to add

        Returns:
            bool: True if add successful
        """
        pass

    @abstractmethod
    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Update entity in graph

        Args:
            graph_id: Graph ID
            entity: Entity to update

        Returns:
            bool: True if update successful
        """
        pass

    @abstractmethod
    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """
        Remove entity from graph

        Args:
            graph_id: Graph ID
            entity_id: Entity ID to remove

        Returns:
            bool: True if remove successful
        """
        pass


class GraphRelationCRUD(ABC):
    """Interface for relation CRUD operations"""

    @abstractmethod
    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Add relation to graph

        Args:
            graph_id: Graph ID
            relation: Relation to add

        Returns:
            bool: True if add successful
        """
        pass

    @abstractmethod
    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Update relation in graph

        Args:
            graph_id: Graph ID
            relation: Relation to update

        Returns:
            bool: True if update successful
        """
        pass

    @abstractmethod
    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """
        Remove relation from graph

        Args:
            graph_id: Graph ID
            relation_id: Relation ID to remove

        Returns:
            bool: True if remove successful
        """
        pass


class GraphQuery(ABC):
    """Interface for graph query operations"""

    @abstractmethod
    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """
        Query entities based on conditions

        Args:
            conditions: Query conditions

        Returns:
            List[Entity]: Matching entities
        """
        pass

    @abstractmethod
    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
        **kwargs: Any,
    ) -> List[Relation]:
        """
        Query relations based on conditions

        Args:
            head_entity: Head entity ID
            tail_entity: Tail entity ID
            relation_type: Relation type
            **kwargs: Additional query parameters

        Returns:
            List[Relation]: Matching relations
        """
        pass


class GraphBackup(ABC):
    """Interface for graph backup and restore operations"""

    @abstractmethod
    def backup_graph(self, graph_id: str, backup_path: str) -> bool:
        """
        Backup graph to file

        Args:
            graph_id: Graph ID
            backup_path: Path to backup file

        Returns:
            bool: True if backup successful
        """
        pass

    @abstractmethod
    def restore_graph(self, backup_path: str) -> Optional[str]:
        """
        Restore graph from backup file

        Args:
            backup_path: Path to backup file

        Returns:
            str: Restored graph ID, None if failed
        """
        pass


class GraphExport(ABC):
    """Interface for graph export operations"""

    @abstractmethod
    def export_graph(self, graph_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """
        Export graph in specified format

        Args:
            graph_id: Graph ID
            format: Export format ('json', 'csv', 'graphml', etc.)

        Returns:
            Dict[str, Any]: Exported data, None if failed
        """
        pass


class GraphStatistics(ABC):
    """Interface for graph statistics operations"""

    @abstractmethod
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get graph statistics

        Args:
            graph_id: Graph ID

        Returns:
            Dict[str, Any]: Statistics data
        """
        pass


# Composition interfaces for different use cases


class BasicGraphStorage(GraphConnection, GraphCRUD):
    """Basic graph storage interface with connection and CRUD operations"""

    pass


class QueryableGraphStorage(BasicGraphStorage, GraphQuery):
    """Graph storage with query capabilities"""

    pass


class FullGraphStorage(
    BasicGraphStorage, GraphEntityCRUD, GraphRelationCRUD, GraphQuery, GraphBackup, GraphExport, GraphStatistics
):
    """Full-featured graph storage interface"""

    pass


class ReadOnlyGraphStorage(GraphConnection, GraphQuery, GraphStatistics):
    """Read-only graph storage interface"""

    pass
