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
    """Minimal graph builder implementing only core building functionality.

    This class provides a lightweight implementation for basic knowledge graph
    construction from text sources. It's designed for clients that only need
    essential graph building capabilities without additional features like
    merging, validation, or incremental updates.

    Attributes:
        text_entity_extractor: Extractor for identifying entities from text.
        text_relation_extractor: Extractor for identifying relations from text.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        """Initialize the minimal graph builder.

        Args:
            text_entity_extractor: Custom entity extractor for text processing.
                If None, uses default TextEntityExtractor.
            text_relation_extractor: Custom relation extractor for text processing.
                If None, uses default TextRelationExtractor.
        """
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "minimal_graph",
    ) -> KnowledgeGraph:
        """Build a knowledge graph from text sources.

        This method processes a list of text documents to extract entities and
        their relationships, then constructs a knowledge graph. The process is
        performed asynchronously for better performance.

        Args:
            texts: List of text documents to process. If None or empty,
                returns an empty graph.
            database_schema: Database schema for entity extraction. Currently
                not used in minimal builder but kept for interface compatibility.
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Constructed graph containing extracted entities
                and relations from the input texts.

        Raises:
            Exception: Re-raises any exception that occurs during graph building
                after logging the error.
        """
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
    """Flexible graph builder supporting building and updating operations.

    This builder extends the basic functionality with update capabilities and
    graph merging through mixins. It can process both text and database sources
    to construct comprehensive knowledge graphs. Uses composition pattern to
    add merging functionality only when needed.

    Attributes:
        text_entity_extractor: Extractor for identifying entities from text.
        db_entity_extractor: Extractor for identifying entities from database schema.
        text_relation_extractor: Extractor for identifying relations from text.
        db_relation_extractor: Extractor for identifying relations from database.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        db_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
        db_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        """Initialize the flexible graph builder.

        Args:
            text_entity_extractor: Custom entity extractor for text processing.
                If None, uses default TextEntityExtractor.
            db_entity_extractor: Custom entity extractor for database processing.
                If None, uses default DatabaseEntityExtractor.
            text_relation_extractor: Custom relation extractor for text processing.
                If None, uses default TextRelationExtractor.
            db_relation_extractor: Custom relation extractor for database processing.
                If None, uses default DatabaseRelationExtractor.
        """
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
        """Build a knowledge graph from multiple data sources.

        This method can process both text documents and database schemas to
        create a comprehensive knowledge graph. It performs entity deduplication
        and ensures all operations are executed asynchronously for optimal performance.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema dictionary for extracting structured data.
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Constructed graph containing entities and relations
                from all provided sources.

        Raises:
            Exception: Re-raises any exception that occurs during graph building
                after logging the error.
        """
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
        """Update an existing knowledge graph with new entities and relations.

        This method adds new entities and relations to an existing graph,
        ensuring no duplicates are added and maintaining graph integrity.

        Args:
            graph: Existing knowledge graph to update.
            new_entities: List of new entities to add to the graph.
            new_relations: List of new relations to add to the graph.

        Returns:
            KnowledgeGraph: Updated graph with new entities and relations.

        Raises:
            Exception: Re-raises any exception that occurs during graph updating
                after logging the error.
        """
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
        """Process a single text document asynchronously to extract entities and relations.

        Args:
            text: Text document to process.

        Returns:
            Tuple containing lists of extracted entities and relations.
        """
        text_entities = await asyncio.get_event_loop().run_in_executor(
            None, self.text_entity_extractor.extract_from_text, text
        )
        text_relations = await asyncio.get_event_loop().run_in_executor(
            None, self.text_relation_extractor.extract_from_text, text, text_entities
        )
        return text_entities, text_relations

    async def _add_entity_async(self, graph: KnowledgeGraph, entity: Entity) -> None:
        """Add an entity to the graph asynchronously.

        Args:
            graph: Knowledge graph to add entity to.
            entity: Entity to add to the graph.
        """
        await asyncio.get_event_loop().run_in_executor(None, graph.add_entity, entity)

    async def _add_relation_async(self, graph: KnowledgeGraph, relation: Relation) -> None:
        """Add a relation to the graph asynchronously.

        Args:
            graph: Knowledge graph to add relation to.
            relation: Relation to add to the graph.
        """
        await asyncio.get_event_loop().run_in_executor(None, graph.add_relation, relation)


class ComprehensiveGraphBuilder(
    FullFeaturedGraphBuilder, GraphMergerMixin, GraphValidatorMixin, GraphExporterMixin, GraphStatisticsMixin
):
    """Comprehensive graph builder with all available features.

    This is the most feature-complete builder that combines all available
    functionality through multiple mixins. It provides building, updating,
    merging, validation, exporting, and statistics capabilities.

    Only use this builder when you actually need ALL the functionality.
    For most use cases, consider using more focused builders like
    MinimalGraphBuilder or FlexibleGraphBuilder for better performance.

    Attributes:
        text_entity_extractor: Extractor for identifying entities from text.
        db_entity_extractor: Extractor for identifying entities from database schema.
        text_relation_extractor: Extractor for identifying relations from text.
        db_relation_extractor: Extractor for identifying relations from database.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        db_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
        db_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        """Initialize the comprehensive graph builder.

        Args:
            text_entity_extractor: Custom entity extractor for text processing.
                If None, uses default TextEntityExtractor.
            db_entity_extractor: Custom entity extractor for database processing.
                If None, uses default DatabaseEntityExtractor.
            text_relation_extractor: Custom relation extractor for text processing.
                If None, uses default TextRelationExtractor.
            db_relation_extractor: Custom relation extractor for database processing.
                If None, uses default DatabaseRelationExtractor.
        """
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
        """Build a comprehensive knowledge graph with full processing capabilities.

        This method leverages the FlexibleGraphBuilder's logic to construct
        a knowledge graph from multiple sources while providing access to
        all additional features through mixins.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema dictionary for extracting structured data.
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Comprehensive graph with all processing features available.
        """
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
        """Update the knowledge graph with validation.

        This method first updates the graph with new entities and relations,
        then performs validation to ensure graph integrity.

        Args:
            graph: Existing knowledge graph to update.
            new_entities: List of new entities to add to the graph.
            new_relations: List of new relations to add to the graph.

        Returns:
            KnowledgeGraph: Updated and validated graph.
        """
        # Update first
        flexible_builder = FlexibleGraphBuilder()
        updated_graph = await flexible_builder.update_graph(graph, new_entities, new_relations)

        # Then validate
        validation_result = await self.validate_graph(updated_graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return updated_graph


class StreamingBuilder(StreamingGraphBuilder, IncrementalBuilderMixin, GraphStatisticsMixin):
    """Streaming graph builder for real-time incremental updates.

    This builder is optimized for real-time applications that need to process
    documents as they arrive. It provides incremental update capabilities
    and statistics tracking without requiring merging or validation features
    that might slow down streaming operations.

    Attributes:
        text_entity_extractor: Extractor for identifying entities from text.
        text_relation_extractor: Extractor for identifying relations from text.
        _current_graph: Internal reference to the current graph state.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        """Initialize the streaming graph builder.

        Args:
            text_entity_extractor: Custom entity extractor for text processing.
                If None, uses default TextEntityExtractor.
            text_relation_extractor: Custom relation extractor for text processing.
                If None, uses default TextRelationExtractor.
        """
        super().__init__()
        self.text_entity_extractor = text_entity_extractor or TextEntityExtractor()
        self.text_relation_extractor = text_relation_extractor or TextRelationExtractor()

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "streaming_graph",
    ) -> KnowledgeGraph:
        """Build the initial knowledge graph for streaming operations.

        This method creates the base graph that will be incrementally updated
        as new documents arrive in the stream.

        Args:
            texts: Initial set of text documents to process.
            database_schema: Database schema for initial extraction. Currently
                not used in streaming builder but kept for interface compatibility.
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Initial graph ready for streaming updates.
        """
        if not texts:
            return KnowledgeGraph(name=graph_name)

        # Use minimal builder for initial construction
        minimal_builder = MinimalGraphBuilder(self.text_entity_extractor, self.text_relation_extractor)

        graph = await minimal_builder.build_graph(texts, database_schema, graph_name)
        self._current_graph = graph
        return graph


class BatchBuilder(BatchGraphBuilder, GraphMergerMixin):
    """Batch graph builder for processing multiple heterogeneous sources.

    This builder is optimized for scenarios where you need to process multiple
    data sources efficiently and merge them into a single knowledge graph.
    It doesn't include incremental updates or validation features to maintain
    optimal performance for batch operations.

    Attributes:
        text_entity_extractor: Extractor for identifying entities from text.
        db_entity_extractor: Extractor for identifying entities from database schema.
        text_relation_extractor: Extractor for identifying relations from text.
        db_relation_extractor: Extractor for identifying relations from database.
    """

    def __init__(
        self,
        text_entity_extractor: Optional[BaseEntityExtractor] = None,
        db_entity_extractor: Optional[BaseEntityExtractor] = None,
        text_relation_extractor: Optional[BaseRelationExtractor] = None,
        db_relation_extractor: Optional[BaseRelationExtractor] = None,
    ):
        """Initialize the batch graph builder.

        Args:
            text_entity_extractor: Custom entity extractor for text processing.
                If None, uses default TextEntityExtractor.
            db_entity_extractor: Custom entity extractor for database processing.
                If None, uses default DatabaseEntityExtractor.
            text_relation_extractor: Custom relation extractor for text processing.
                If None, uses default TextRelationExtractor.
            db_relation_extractor: Custom relation extractor for database processing.
                If None, uses default DatabaseRelationExtractor.
        """
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
        """Build a knowledge graph optimized for batch processing.

        This method leverages the FlexibleGraphBuilder's logic to efficiently
        process multiple data sources in batch mode.

        Args:
            texts: List of text documents to process for entity and relation extraction.
            database_schema: Database schema dictionary for extracting structured data.
            graph_name: Name identifier for the generated graph.

        Returns:
            KnowledgeGraph: Constructed graph optimized for batch operations.
        """
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
        """Build a knowledge graph from multiple heterogeneous data sources.

        This method processes various types of data sources concurrently and
        merges them into a single comprehensive knowledge graph. It handles
        text, database, and mixed sources efficiently.

        Args:
            sources: List of source dictionaries, each containing 'type' and 'data' keys.
                Supported types: 'text', 'database', 'mixed'.
            graph_name: Name identifier for the final merged graph.

        Returns:
            KnowledgeGraph: Merged graph containing data from all valid sources.
        """
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
