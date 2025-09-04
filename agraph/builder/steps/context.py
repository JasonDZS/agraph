"""
Build context for managing state and data throughout the build pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from ...base.graphs.legacy import KnowledgeGraph
from ...base.graphs.optimized import OptimizedKnowledgeGraph
from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation
from ...base.models.text import TextChunk
from ...config import BuildSteps


@dataclass
class BuildContext:
    """
    Build context that carries state and data throughout the build pipeline.
    
    This context object centralizes:
    - Input parameters
    - Intermediate results from each step
    - Build state and configuration
    - Step execution control logic
    """
    
    # Input parameters
    texts: List[str]
    graph_name: str = ""
    graph_description: str = ""
    use_cache: bool = True
    from_step: Optional[str] = None
    enable_knowledge_graph: bool = True
    
    # Additional configuration
    documents: Optional[List[Union[str, Any]]] = None  # For build_from_documents
    
    # Intermediate results (populated by steps)
    chunks: Optional[List[TextChunk]] = None
    entities: Optional[List[Entity]] = None
    relations: Optional[List[Relation]] = None
    clusters: Optional[List[Cluster]] = None
    knowledge_graph: Optional[Union[KnowledgeGraph, OptimizedKnowledgeGraph]] = None
    
    # Step execution state
    current_step: Optional[str] = None
    completed_steps: Set[str] = field(default_factory=set)
    skipped_steps: Set[str] = field(default_factory=set)
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Error tracking
    errors: List[Exception] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_start_time: Optional[float] = None
    step_start_times: Dict[str, float] = field(default_factory=dict)
    step_execution_times: Dict[str, float] = field(default_factory=dict)
    
    def initialize(self, input_data: Any) -> None:
        """Initialize context with input data."""
        if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            self.texts = input_data
        else:
            # Handle other input types if needed
            self.texts = input_data if isinstance(input_data, list) else [str(input_data)]
    
    def should_execute_step(self, step_name: str) -> bool:
        """
        Determine if a step should be executed based on from_step parameter.
        
        Args:
            step_name: Name of the step to check
            
        Returns:
            True if step should be executed
        """
        if self.from_step is None:
            return True
        
        try:
            from_step_index = BuildSteps.get_step_index(self.from_step)
            current_step_index = BuildSteps.get_step_index(step_name)
            return current_step_index >= from_step_index
        except (AttributeError, ValueError):
            # If step index lookup fails, default to executing the step
            return True
    
    def should_skip_step(self, step_name: str) -> bool:
        """
        Determine if a step should be skipped based on configuration.
        
        Args:
            step_name: Name of the step to check
            
        Returns:
            True if step should be skipped
        """
        # Skip knowledge graph related steps if disabled
        if not self.enable_knowledge_graph:
            kg_steps = {
                BuildSteps.ENTITY_EXTRACTION,
                BuildSteps.RELATION_EXTRACTION,
                BuildSteps.CLUSTER_FORMATION
            }
            if step_name in kg_steps:
                return True
        
        # Skip if step is already completed and we're not forcing re-execution
        if step_name in self.completed_steps and not self.should_execute_step(step_name):
            return True
        
        return False
    
    def mark_step_started(self, step_name: str, start_time: float) -> None:
        """Mark step as started."""
        self.current_step = step_name
        self.step_start_times[step_name] = start_time
    
    def mark_step_completed(self, step_name: str, end_time: float, result: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as completed."""
        self.completed_steps.add(step_name)
        
        if step_name in self.step_start_times:
            execution_time = end_time - self.step_start_times[step_name]
            self.step_execution_times[step_name] = execution_time
        
        if result is not None:
            self.step_results[step_name] = result
            
        if metadata:
            self.step_metadata[step_name] = metadata
        
        # Update intermediate results based on step
        self._update_intermediate_results(step_name, result)
        
        # Clear current step
        if self.current_step == step_name:
            self.current_step = None
    
    def mark_step_skipped(self, step_name: str, reason: str = "") -> None:
        """Mark step as skipped."""
        self.skipped_steps.add(step_name)
        if reason:
            self.warnings.append(f"Step {step_name} skipped: {reason}")
    
    def add_error(self, error: Exception, step_name: Optional[str] = None) -> None:
        """Add error to context."""
        self.errors.append(error)
        if step_name:
            self.step_metadata.setdefault(step_name, {})["error"] = str(error)
    
    def add_warning(self, warning: str) -> None:
        """Add warning to context."""
        self.warnings.append(warning)
    
    def get_result_for_step(self, step_name: str) -> Any:
        """Get result for a specific step."""
        return self.step_results.get(step_name)
    
    def get_metadata_for_step(self, step_name: str) -> Dict[str, Any]:
        """Get metadata for a specific step."""
        return self.step_metadata.get(step_name, {})
    
    def is_step_completed(self, step_name: str) -> bool:
        """Check if step is completed."""
        return step_name in self.completed_steps
    
    def is_step_skipped(self, step_name: str) -> bool:
        """Check if step was skipped."""
        return step_name in self.skipped_steps
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total_time = sum(self.step_execution_times.values())
        
        return {
            "completed_steps": list(self.completed_steps),
            "skipped_steps": list(self.skipped_steps),
            "total_execution_time": total_time,
            "step_execution_times": dict(self.step_execution_times),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "final_results": {
                "chunks_count": len(self.chunks) if self.chunks else 0,
                "entities_count": len(self.entities) if self.entities else 0,
                "relations_count": len(self.relations) if self.relations else 0,
                "clusters_count": len(self.clusters) if self.clusters else 0,
                "has_knowledge_graph": self.knowledge_graph is not None
            }
        }
    
    def _update_intermediate_results(self, step_name: str, result: Any) -> None:
        """Update intermediate results based on step completion."""
        if step_name == BuildSteps.TEXT_CHUNKING and isinstance(result, list):
            self.chunks = result
        elif step_name == BuildSteps.ENTITY_EXTRACTION and isinstance(result, list):
            self.entities = result
        elif step_name == BuildSteps.RELATION_EXTRACTION and isinstance(result, list):
            self.relations = result
        elif step_name == BuildSteps.CLUSTER_FORMATION and isinstance(result, list):
            self.clusters = result
        elif step_name == BuildSteps.GRAPH_ASSEMBLY:
            if isinstance(result, (KnowledgeGraph, OptimizedKnowledgeGraph)):
                self.knowledge_graph = result
    
    def validate_state(self) -> List[str]:
        """
        Validate the current state and return any validation errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required inputs
        if not self.texts:
            errors.append("No input texts provided")
        
        # Check step dependencies
        if self.is_step_completed(BuildSteps.ENTITY_EXTRACTION) and not self.chunks:
            errors.append("Entity extraction completed but no chunks available")
        
        if self.is_step_completed(BuildSteps.RELATION_EXTRACTION) and not self.entities:
            errors.append("Relation extraction completed but no entities available")
        
        if self.is_step_completed(BuildSteps.CLUSTER_FORMATION) and (not self.entities or not self.relations):
            errors.append("Cluster formation completed but entities/relations missing")
        
        if self.is_step_completed(BuildSteps.GRAPH_ASSEMBLY) and not self.knowledge_graph:
            errors.append("Graph assembly completed but no knowledge graph created")
        
        return errors
    
    def reset_from_step(self, from_step: str) -> None:
        """Reset context state from a specific step onwards."""
        try:
            from_step_index = BuildSteps.get_step_index(from_step)
            
            # Clear results for steps at or after the from_step
            steps_to_clear = []
            for step_name in self.completed_steps:
                try:
                    step_index = BuildSteps.get_step_index(step_name)
                    if step_index >= from_step_index:
                        steps_to_clear.append(step_name)
                except (AttributeError, ValueError):
                    continue
            
            # Clear the identified steps
            for step_name in steps_to_clear:
                self.completed_steps.discard(step_name)
                self.step_results.pop(step_name, None)
                self.step_metadata.pop(step_name, None)
                self.step_execution_times.pop(step_name, None)
            
            # Reset intermediate results based on from_step
            if from_step_index <= BuildSteps.get_step_index(BuildSteps.TEXT_CHUNKING):
                self.chunks = None
            if from_step_index <= BuildSteps.get_step_index(BuildSteps.ENTITY_EXTRACTION):
                self.entities = None
            if from_step_index <= BuildSteps.get_step_index(BuildSteps.RELATION_EXTRACTION):
                self.relations = None
            if from_step_index <= BuildSteps.get_step_index(BuildSteps.CLUSTER_FORMATION):
                self.clusters = None
            if from_step_index <= BuildSteps.get_step_index(BuildSteps.GRAPH_ASSEMBLY):
                self.knowledge_graph = None
                
        except (AttributeError, ValueError):
            # If step index lookup fails, clear everything to be safe
            self.completed_steps.clear()
            self.step_results.clear()
            self.step_metadata.clear() 
            self.step_execution_times.clear()
            self.chunks = None
            self.entities = None
            self.relations = None
            self.clusters = None
            self.knowledge_graph = None