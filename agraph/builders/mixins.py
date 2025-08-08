"""
Builder mixins following Interface Segregation Principle

These mixins provide optional functionality that can be composed as needed,
following the principle that clients should not depend on interfaces they don't use.

    def _export_to_graphml(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        Export graph to GraphML format.
        from ..utils import get_type_value

        nodes = []
        edges = []

        for entity in graph.entities.values():
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                    "confidence": entity.confidence,
                }
            )'t use.
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation


class GraphMergerMixin:
    """Mixin providing graph merging functionality"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.merge_threshold = getattr(self, "merge_threshold", 0.8)

    async def merge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Merge multiple knowledge graphs

        Args:
            graphs: List of graphs to merge

        Returns:
            KnowledgeGraph: Merged graph
        """
        if not graphs:
            return KnowledgeGraph()

        if len(graphs) == 1:
            return graphs[0]

        try:
            # Create new merged graph
            merged_graph = KnowledgeGraph(name="merged_graph")
            all_entities: List[Entity] = []
            all_relations: List[Relation] = []

            # Collect all entities and relations
            for graph in graphs:
                all_entities.extend(graph.entities.values())
                all_relations.extend(graph.relations.values())

            # Align and deduplicate entities asynchronously
            aligned_entities = await self._align_entities_async(all_entities)

            # Add entities to merged graph
            entity_id_mapping = {}
            for entity in aligned_entities:
                if merged_graph.add_entity(entity):
                    entity_id_mapping[entity.id] = entity.id

            # Update relations and add to merged graph
            for relation in all_relations:
                if (
                    relation.head_entity is not None
                    and relation.tail_entity is not None
                    and relation.head_entity.id in entity_id_mapping
                    and relation.tail_entity.id in entity_id_mapping
                ):

                    # Update entity references
                    relation.head_entity = merged_graph.get_entity(entity_id_mapping[relation.head_entity.id])
                    relation.tail_entity = merged_graph.get_entity(entity_id_mapping[relation.tail_entity.id])
                    merged_graph.add_relation(relation)

            return merged_graph

        except Exception as e:
            logger.error(f"Error merging graphs: {e}")
            raise

    def _align_entities(self, entities: List[Entity]) -> List[Entity]:
        """Align entities and remove duplicates (synchronous version)"""
        aligned_entities: List[Entity] = []
        processed_names = set()

        for entity in entities:
            normalized_name = entity.name.lower().strip()

            if normalized_name in processed_names:
                # Find existing entity and merge
                existing_entity = next((e for e in aligned_entities if e.name.lower().strip() == normalized_name), None)
                if existing_entity:
                    self._merge_entity_attributes(existing_entity, entity)
            else:
                aligned_entities.append(entity)
                processed_names.add(normalized_name)

        return aligned_entities

    async def _align_entities_async(self, entities: List[Entity]) -> List[Entity]:
        """Align entities and remove duplicates (asynchronous version)"""
        # For now, delegate to synchronous version
        # In the future, this could be optimized with async processing for large datasets
        return await asyncio.get_event_loop().run_in_executor(None, self._align_entities, entities)

    def _merge_entity_attributes(self, target_entity: Entity, source_entity: Entity) -> None:
        """Merge attributes from source entity into target entity"""
        # Merge aliases
        target_entity.aliases.extend(source_entity.aliases)
        target_entity.aliases = list(set(target_entity.aliases))

        # Merge properties
        target_entity.properties.update(source_entity.properties)

        # Use higher confidence
        if source_entity.confidence > target_entity.confidence:
            target_entity.confidence = source_entity.confidence
            target_entity.description = source_entity.description or target_entity.description


class GraphValidatorMixin:
    """Mixin providing graph validation functionality"""

    async def validate_graph(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Validate knowledge graph using GraphValidator service

        Args:
            graph: Graph to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            from ..services import GraphValidator

            validator = GraphValidator()
            # Run synchronous validation in executor to make it async
            return await asyncio.get_event_loop().run_in_executor(None, validator.validate_graph, graph)
        except ImportError:
            logger.warning("GraphValidator service not available, using basic validation")
            return await self._basic_validation_async(graph)

    async def _basic_validation_async(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Basic validation when service is not available (async version)"""
        return await asyncio.get_event_loop().run_in_executor(None, self._basic_validation, graph)

    def _basic_validation(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Basic validation when service is not available"""
        issues = []

        # Check for broken relations
        for relation_id, relation in graph.relations.items():
            if not relation.head_entity or relation.head_entity.id not in graph.entities:
                issues.append({"type": "missing_head_entity", "relation_id": relation_id, "severity": "high"})

            if not relation.tail_entity or relation.tail_entity.id not in graph.entities:
                issues.append({"type": "missing_tail_entity", "relation_id": relation_id, "severity": "high"})

        return {
            "valid": len([i for i in issues if i.get("severity") == "high"]) == 0,
            "issues": issues,
            "statistics": graph.get_basic_statistics(),
        }


class GraphExporterMixin:
    """Mixin providing graph export functionality"""

    async def export_to_format(self, graph: KnowledgeGraph, format: str) -> Dict[str, Any]:
        """
        Export graph to specified format

        Args:
            graph: Knowledge graph to export
            format: Target format

        Returns:
            Dict[str, Any]: Exported data
        """
        format_lower = format.lower()

        if format_lower == "json":
            return await asyncio.get_event_loop().run_in_executor(None, graph.to_dict)
        elif format_lower == "graphml":
            return await self._export_to_graphml_async(graph)
        elif format_lower == "cytoscape":
            return await self._export_to_cytoscape_async(graph)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _export_to_graphml_async(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export to GraphML format (async version)"""
        return await asyncio.get_event_loop().run_in_executor(None, self._export_to_graphml, graph)

    async def _export_to_cytoscape_async(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export to Cytoscape format (async version)"""
        return await asyncio.get_event_loop().run_in_executor(None, self._export_to_cytoscape, graph)

    def _export_to_graphml(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export to GraphML format"""
        from ..utils import get_type_value

        nodes = []
        edges = []

        for entity in graph.entities.values():
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                    "confidence": entity.confidence,
                }
            )

        for relation in graph.relations.values():
            if relation.head_entity and relation.tail_entity:
                edges.append(
                    {
                        "id": relation.id,
                        "source": relation.head_entity.id,
                        "target": relation.tail_entity.id,
                        "label": get_type_value(relation.relation_type),
                        "confidence": relation.confidence,
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def _export_to_cytoscape(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export to Cytoscape format"""
        from ..utils import get_type_value

        elements = []

        # Add nodes
        for entity in graph.entities.values():
            elements.append(
                {
                    "data": {
                        "id": entity.id,
                        "label": entity.name,
                        "type": get_type_value(entity.entity_type),
                        "confidence": entity.confidence,
                    }
                }
            )

        # Add edges
        for relation in graph.relations.values():
            if relation.head_entity and relation.tail_entity:
                elements.append(
                    {
                        "data": {
                            "id": relation.id,
                            "source": relation.head_entity.id,
                            "target": relation.tail_entity.id,
                            "label": get_type_value(relation.relation_type),
                            "confidence": relation.confidence,
                        }
                    }
                )

        return {"elements": elements}


class IncrementalBuilderMixin:
    """Mixin providing incremental building functionality"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._document_registry: Dict[str, List[str]] = {}  # doc_id -> entity_ids
        self._current_graph: Optional[KnowledgeGraph] = None

    async def add_documents(self, documents: List[str], document_ids: Optional[List[str]] = None) -> KnowledgeGraph:
        """
        Add new documents to existing graph

        Args:
            documents: New documents to process
            document_ids: Optional document IDs for tracking

        Returns:
            KnowledgeGraph: Updated graph
        """
        if not hasattr(self, "build_graph"):
            raise NotImplementedError("This mixin requires a build_graph method")

        # Generate IDs if not provided
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]

        if len(document_ids) != len(documents):
            raise ValueError("Number of document IDs must match number of documents")

        try:
            if self._current_graph is None:
                # First time building
                self._current_graph = await self.build_graph(texts=documents)
                # Track all entities for these documents
                for doc_id in document_ids:
                    self._document_registry[doc_id] = list(self._current_graph.entities.keys())
            else:
                # Build graph for new documents
                new_graph = await self.build_graph(texts=documents)

                # Track entities for new documents
                for doc_id in document_ids:
                    self._document_registry[doc_id] = list(new_graph.entities.keys())

                # Merge with existing graph
                if hasattr(self, "merge_graphs"):
                    self._current_graph = await self.merge_graphs([self._current_graph, new_graph])
                else:
                    # Simple addition without merging
                    for entity in new_graph.entities.values():
                        self._current_graph.add_entity(entity)
                    for relation in new_graph.relations.values():
                        self._current_graph.add_relation(relation)

            # Ensure _current_graph is not None before returning
            assert self._current_graph is not None, "Current graph should not be None at this point"
            return self._current_graph

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def remove_documents(self, document_ids: List[str]) -> KnowledgeGraph:
        """
        Remove documents from existing graph

        Args:
            document_ids: Document IDs to remove

        Returns:
            KnowledgeGraph: Updated graph
        """
        if self._current_graph is None:
            raise ValueError("No graph exists to remove documents from")

        try:
            entities_to_remove = set()

            # Collect entities from documents to remove
            for doc_id in document_ids:
                if doc_id in self._document_registry:
                    entities_to_remove.update(self._document_registry[doc_id])
                    del self._document_registry[doc_id]

            # Remove entities (this will also remove related relations)
            for entity_id in entities_to_remove:
                if entity_id in self._current_graph.entities:
                    self._current_graph.remove_entity(entity_id)

            return self._current_graph

        except Exception as e:
            logger.error(f"Error removing documents: {e}")
            raise

    def get_document_registry(self) -> Dict[str, List[str]]:
        """Get the document to entity mapping"""
        return self._document_registry.copy()


class GraphStatisticsMixin:
    """Mixin providing graph statistics functionality"""

    async def get_detailed_statistics(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Get detailed graph statistics using GraphAnalyzer service

        Args:
            graph: Graph to analyze

        Returns:
            Dict[str, Any]: Detailed statistics
        """
        try:
            from ..services import GraphAnalyzer

            analyzer = GraphAnalyzer()
            # Check if analyzer has async method
            # Run synchronous method in executor
            return await asyncio.get_event_loop().run_in_executor(None, analyzer.get_statistics, graph)
        except ImportError:
            logger.warning("GraphAnalyzer service not available, using basic statistics")
            return await asyncio.get_event_loop().run_in_executor(None, graph.get_basic_statistics)

    async def compute_graph_metrics(self, graph: KnowledgeGraph) -> Dict[str, float]:
        """
        Compute various graph metrics

        Args:
            graph: Graph to analyze

        Returns:
            Dict[str, float]: Graph metrics
        """
        try:
            from ..services import GraphAnalyzer

            analyzer = GraphAnalyzer()
            # Run synchronous methods in executor
            density = await asyncio.get_event_loop().run_in_executor(None, analyzer.compute_density, graph)
            isolated_nodes = await asyncio.get_event_loop().run_in_executor(None, analyzer.find_isolated_nodes, graph)

            return {
                "density": density,
                "isolated_nodes_ratio": (len(isolated_nodes) / len(graph.entities) if graph.entities else 0),
            }
        except ImportError:
            # Basic metrics
            entity_count = len(graph.entities)
            relation_count = len(graph.relations)
            max_edges = entity_count * (entity_count - 1) if entity_count > 1 else 1

            return {
                "density": relation_count / max_edges if max_edges > 0 else 0,
                "isolated_nodes_ratio": 0,
            }
