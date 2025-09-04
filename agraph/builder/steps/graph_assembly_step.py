"""
Graph assembly step implementation.
"""

from typing import Any, Union

from ...base.graphs.optimized import KnowledgeGraph
from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation
from ...base.models.text import TextChunk
from ...config import BuildSteps
from ..handler.graph_assembler import GraphAssembler
from .base import BuildStep, StepResult
from .context import BuildContext


class GraphAssemblyStep(BuildStep):
    """Step for assembling the final knowledge graph."""
    
    def __init__(self, graph_assembler: GraphAssembler, cache_manager):
        """
        Initialize graph assembly step.
        
        Args:
            graph_assembler: Handler for graph assembly operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.GRAPH_ASSEMBLY, cache_manager)
        self.graph_assembler = graph_assembler
    
    async def _execute_step(self, context: BuildContext) -> StepResult[KnowledgeGraph]:
        """
        Execute graph assembly logic.
        
        Args:
            context: Build context containing all components for graph assembly
            
        Returns:
            StepResult containing the assembled knowledge graph
        """
        try:
            # Get required components from context
            entities = context.entities or []
            relations = context.relations or []
            clusters = context.clusters or []
            chunks = context.chunks or []
            
            # Validate components (some can be empty for partial graphs)
            if not chunks:
                return StepResult.failure_result("No chunks available for graph assembly")
            
            # Validate input types
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, TextChunk):
                    return StepResult.failure_result(
                        f"Invalid chunk at index {i}: expected TextChunk, got {type(chunk)}"
                    )
            
            for i, entity in enumerate(entities):
                if not isinstance(entity, Entity):
                    return StepResult.failure_result(
                        f"Invalid entity at index {i}: expected Entity, got {type(entity)}"
                    )
            
            for i, relation in enumerate(relations):
                if not isinstance(relation, Relation):
                    return StepResult.failure_result(
                        f"Invalid relation at index {i}: expected Relation, got {type(relation)}"
                    )
            
            for i, cluster in enumerate(clusters):
                if not isinstance(cluster, Cluster):
                    return StepResult.failure_result(
                        f"Invalid cluster at index {i}: expected Cluster, got {type(cluster)}"
                    )
            
            # Execute graph assembly (synchronous operation)
            knowledge_graph = self.graph_assembler.assemble_knowledge_graph(
                entities,
                relations,
                clusters,
                chunks,
                context.graph_name,
                context.graph_description,
                context.use_cache
            )
            
            # Validate the result
            if not isinstance(knowledge_graph, KnowledgeGraph):
                return StepResult.failure_result(
                    f"Graph assembly returned invalid type: expected OptimizedKnowledgeGraph, "
                    f"got {type(knowledge_graph)}"
                )
            
            # Calculate graph metrics
            graph_metrics = {
                "graph_name": knowledge_graph.name,
                "graph_description": knowledge_graph.description,
                "total_entities": len(knowledge_graph.entities),
                "total_relations": len(knowledge_graph.relations),
                "total_clusters": len(knowledge_graph.clusters) if hasattr(knowledge_graph, 'clusters') else 0,
                "total_text_chunks": len(knowledge_graph.text_chunks) if hasattr(knowledge_graph, 'text_chunks') else 0
            }
            
            # Calculate additional metrics if it's KnowledgeGraph
            if isinstance(knowledge_graph, KnowledgeGraph):
                # Get performance metrics if available
                if hasattr(knowledge_graph, 'get_performance_metrics'):
                    try:
                        perf_metrics = knowledge_graph.get_performance_metrics()
                        graph_metrics.update({
                            "performance_metrics": perf_metrics
                        })
                    except Exception:
                        # Performance metrics are optional
                        pass
                
                # Index statistics
                if hasattr(knowledge_graph, 'index_manager'):
                    try:
                        index_stats = knowledge_graph.index_manager.get_index_statistics()
                        graph_metrics.update({
                            "index_statistics": index_stats
                        })
                    except Exception:
                        # Index stats are optional
                        pass
            
            # Entity type distribution
            entity_types = {}
            for entity in knowledge_graph.entities.values():
                entity_type = str(entity.entity_type)
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Relation type distribution
            relation_types = {}
            for relation in knowledge_graph.relations.values():
                relation_type = str(relation.relation_type)
                relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
            
            graph_metrics.update({
                "entity_type_distribution": entity_types,
                "relation_type_distribution": relation_types,
                "graph_density": len(knowledge_graph.relations) / max(len(knowledge_graph.entities), 1),
                "graph_type": type(knowledge_graph).__name__
            })
            
            return StepResult.success_result(
                knowledge_graph,
                metadata=graph_metrics
            )
            
        except Exception as e:
            return StepResult.failure_result(f"Graph assembly failed: {str(e)}")
    
    def _get_cache_input_data(self, context: BuildContext) -> Any:
        """Get input data for cache key generation."""
        # Use all components for cache key
        return (
            context.entities,
            context.relations,
            context.clusters,
            context.chunks,
            context.graph_name,
            context.graph_description
        )
    
    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""
        return KnowledgeGraph  # Default to optimized version