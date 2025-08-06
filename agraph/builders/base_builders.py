import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..entities import Entity
from ..extractors.entity_extractor import BaseEntityExtractor, DatabaseEntityExtractor, TextEntityExtractor
from ..extractors.relation_extractor import BaseRelationExtractor, DatabaseRelationExtractor, TextRelationExtractor
from ..graph import KnowledgeGraph
from ..relations import Relation
from .interfaces import (
    BasicGraphBuilder,
    BatchGraphBuilder,
    FullFeaturedGraphBuilder,
    StreamingGraphBuilder,
    UpdatableGraphBuilder,
)
from .mixins import (
    GraphExporterMixin,
    GraphMergerMixin,
    GraphStatisticsMixin,
    GraphValidatorMixin,
    IncrementalBuilderMixin,
)

logger = logging.getLogger(__name__)


class MinimalGraphBuilder(BasicGraphBuilder):
    """
    Minimal graph builder - implements only core building functionality

    Perfect for clients that only need basic graph construction without
    additional features like merging, validation, or incremental updates.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "minimal_graph",
    ) -> KnowledgeGraph:
        """Build knowledge graph from texts only"""
        if not texts:
            return KnowledgeGraph(name=graph_name)

        try:
            graph = KnowledgeGraph(name=graph_name)
            all_entities = []
            all_relations = []

            # Extract entities and relations from texts asynchronously
            for text in texts:
                text_entities = await asyncio.get_event_loop().run_in_executor(
                    None, self.text_entity_extractor.extract_from_text, text
                )
                all_entities.extend(text_entities)

                text_relations = await asyncio.get_event_loop().run_in_executor(
                    None, self.text_relation_extractor.extract_from_text, text, text_entities
                )
                all_relations.extend(text_relations)

            # Add entities to graph asynchronously
            for entity in all_entities:
                await asyncio.get_event_loop().run_in_executor(None, graph.add_entity, entity)

            # Add relations to graph asynchronously
            for relation in all_relations:
                if (
                    relation.head_entity
                    and relation.head_entity.id in graph.entities
                    and relation.tail_entity
                    and relation.tail_entity.id in graph.entities
                ):
                    await asyncio.get_event_loop().run_in_executor(None, graph.add_relation, relation)

            return graph

        except Exception as e:
            logger.error(f"Error building minimal graph: {e}")
            raise


class FlexibleGraphBuilder(UpdatableGraphBuilder, GraphMergerMixin):
    """
    Flexible graph builder - supports building and updating

    Uses composition (mixin) to add merging functionality only when needed.
    Clients that don't need merging can use MinimalGraphBuilder instead.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        db_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
        db_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        super().__init__()
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.db_entity_extractor = db_entity_extractor or DatabaseEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()
        self.db_relation_extractor = db_relation_extractor or DatabaseRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "flexible_graph",
    ) -> KnowledgeGraph:
        """Build graph from multiple sources"""
        try:
            graph = KnowledgeGraph(name=graph_name)
            all_entities = []
            all_relations = []

            # Process texts asynchronously
            if texts:
                text_tasks = []
                for text in texts:
                    text_task = asyncio.create_task(self._process_text_async(text))
                    text_tasks.append(text_task)

                text_results = await asyncio.gather(*text_tasks)
                for entities, relations in text_results:
                    all_entities.extend(entities)
                    all_relations.extend(relations)

            # Process database schema asynchronously
            if database_schema:
                db_entities = await asyncio.get_event_loop().run_in_executor(
                    None, self.db_entity_extractor.extract_from_database, database_schema
                )
                all_entities.extend(db_entities)

                db_relations = await asyncio.get_event_loop().run_in_executor(
                    None, self.db_relation_extractor.extract_from_database, database_schema, db_entities
                )
                all_relations.extend(db_relations)

            # Deduplicate entities asynchronously
            unique_entities = await asyncio.get_event_loop().run_in_executor(
                None, self.text_entity_extractor.deduplicate_entities, all_entities
            )

            # Add to graph asynchronously
            entity_tasks = [asyncio.create_task(self._add_entity_async(graph, entity)) for entity in unique_entities]
            await asyncio.gather(*entity_tasks)

            relation_tasks = [
                asyncio.create_task(self._add_relation_async(graph, relation))
                for relation in all_relations
                if (
                    relation.head_entity
                    and relation.head_entity.id in graph.entities
                    and relation.tail_entity
                    and relation.tail_entity.id in graph.entities
                )
            ]
            await asyncio.gather(*relation_tasks)
            # TODO: Implement save_graph method or use storage directly
            # self.save_graph(graph, graph_name)
            return graph

        except Exception as e:
            logger.error(f"Error building flexible graph: {e}")
            raise

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """Update existing graph with new entities and relations"""
        try:
            if new_entities:
                entity_tasks = []
                for entity in new_entities:
                    if entity.id not in graph.entities:
                        task = asyncio.create_task(self._add_entity_async(graph, entity))
                        entity_tasks.append(task)
                await asyncio.gather(*entity_tasks)

            if new_relations:
                relation_tasks = []
                for relation in new_relations:
                    if (
                        relation.head_entity
                        and relation.head_entity.id in graph.entities
                        and relation.tail_entity
                        and relation.tail_entity.id in graph.entities
                        and relation.id not in graph.relations
                    ):
                        task = asyncio.create_task(self._add_relation_async(graph, relation))
                        relation_tasks.append(task)
                await asyncio.gather(*relation_tasks)

            return graph

        except Exception as e:
            logger.error(f"Error updating graph: {e}")
            raise

    async def _process_text_async(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """Process a single text asynchronously"""
        text_entities = await asyncio.get_event_loop().run_in_executor(
            None, self.text_entity_extractor.extract_from_text, text
        )
        text_relations = await asyncio.get_event_loop().run_in_executor(
            None, self.text_relation_extractor.extract_from_text, text, text_entities
        )
        return text_entities, text_relations

    async def _add_entity_async(self, graph: KnowledgeGraph, entity: Entity) -> None:
        """Add entity to graph asynchronously"""
        await asyncio.get_event_loop().run_in_executor(None, graph.add_entity, entity)

    async def _add_relation_async(self, graph: KnowledgeGraph, relation: Relation) -> None:
        """Add relation to graph asynchronously"""
        await asyncio.get_event_loop().run_in_executor(None, graph.add_relation, relation)


class ComprehensiveGraphBuilder(
    FullFeaturedGraphBuilder, GraphMergerMixin, GraphValidatorMixin, GraphExporterMixin, GraphStatisticsMixin
):
    """
    Comprehensive graph builder with all features

    Only implement this interface when you actually need ALL the functionality.
    Most clients should use more focused interfaces like MinimalGraphBuilder
    or FlexibleGraphBuilder.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        db_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
        db_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        super().__init__()
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.db_entity_extractor = db_entity_extractor or DatabaseEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()
        self.db_relation_extractor = db_relation_extractor or DatabaseRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "comprehensive_graph",
    ) -> KnowledgeGraph:
        """Build comprehensive graph with full processing"""
        # Reuse FlexibleGraphBuilder logic
        flexible_builder = FlexibleGraphBuilder(
            self.text_entity_extractor,
            self.db_entity_extractor,
            self.text_relation_extractor,
            self.db_relation_extractor,
        )
        return await flexible_builder.build_graph(texts, database_schema, graph_name)

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """Update graph with validation"""
        # Update first
        flexible_builder = FlexibleGraphBuilder()
        updated_graph = await flexible_builder.update_graph(graph, new_entities, new_relations)

        # Then validate
        validation_result = await self.validate_graph(updated_graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return updated_graph


class StreamingBuilder(StreamingGraphBuilder, IncrementalBuilderMixin, GraphStatisticsMixin):
    """
    Streaming graph builder for incremental updates

    Perfect for real-time applications that need to process documents
    as they arrive, without requiring merging or validation capabilities.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        super().__init__()
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "streaming_graph",
    ) -> KnowledgeGraph:
        """Build initial graph for streaming"""
        if not texts:
            return KnowledgeGraph(name=graph_name)

        # Use minimal builder for initial construction
        minimal_builder = MinimalGraphBuilder(self.text_entity_extractor, self.text_relation_extractor)

        graph = await minimal_builder.build_graph(texts, database_schema, graph_name)
        self._current_graph = graph
        return graph


class BatchBuilder(BatchGraphBuilder, GraphMergerMixin):
    """
    Batch graph builder for processing multiple sources

    Optimized for scenarios where you need to process multiple data sources
    and merge them, but don't need incremental updates or validation.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        db_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
        db_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        super().__init__()
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.db_entity_extractor = db_entity_extractor or DatabaseEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()
        self.db_relation_extractor = db_relation_extractor or DatabaseRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "batch_graph",
    ) -> KnowledgeGraph:
        """Build graph optimized for batch processing"""
        # Use flexible builder logic
        flexible_builder = FlexibleGraphBuilder(
            self.text_entity_extractor,
            self.db_entity_extractor,
            self.text_relation_extractor,
            self.db_relation_extractor,
        )
        return await flexible_builder.build_graph(texts, database_schema, graph_name)

    async def build_from_multiple_sources(
        self, sources: List[Dict[str, Any]], graph_name: str = "multi_source_batch_graph"
    ) -> KnowledgeGraph:
        """Build graph from multiple heterogeneous sources"""
        tasks = []

        for i, source in enumerate(sources):
            source_type = source.get("type")
            source_data = source.get("data")
            source_name = f"{graph_name}_source_{i}"

            if source_type == "text":
                texts = source_data if isinstance(source_data, list) else [source_data]
                task = self.build_graph(texts=texts, graph_name=source_name)
            elif source_type == "database":
                task = self.build_graph(database_schema=source_data, graph_name=source_name)
            elif source_type == "mixed":
                if source_data is not None:
                    texts = source_data.get("texts", [])
                    db_schema = source_data.get("database_schema")
                    task = self.build_graph(texts=texts, database_schema=db_schema, graph_name=source_name)
                else:
                    logger.warning(f"Mixed source data is None for {source_name}")
                    continue
            else:
                logger.warning(f"Unknown source type: {source_type}")
                continue

            tasks.append(task)

        if not tasks:
            return KnowledgeGraph(name=graph_name)

        # Execute all tasks concurrently
        sub_graphs = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_graphs: List[KnowledgeGraph] = [
            graph for graph in sub_graphs if isinstance(graph, KnowledgeGraph) and not isinstance(graph, Exception)
        ]

        if not valid_graphs:
            return KnowledgeGraph(name=graph_name)

        # Merge all sub-graphs
        merged_graph = await self.merge_graphs(valid_graphs)
        merged_graph.name = graph_name
        return merged_graph
