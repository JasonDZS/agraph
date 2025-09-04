"""
Build pipeline for orchestrating knowledge graph construction steps.
"""

import time
from typing import Any, List, Union

from ..base.graphs.optimized import KnowledgeGraph
from ..logger import logger
from .cache import CacheManager
from .steps.base import BuildStep, StepResult


class BuildPipeline:
    """
    Pipeline for orchestrating knowledge graph build steps.
    
    This pipeline provides:
    - Sequential step execution with error handling
    - Step skipping based on configuration
    - Comprehensive logging and metrics
    - State management through BuildContext
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize build pipeline.
        
        Args:
            cache_manager: Cache manager for step coordination
        """
        self.cache_manager = cache_manager
        self.steps: List[BuildStep] = []
        self._execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "step_metrics": {}
        }
    
    def add_step(self, step: BuildStep) -> "BuildPipeline":
        """
        Add a step to the pipeline.
        
        Args:
            step: Build step to add
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        logger.debug(f"Added step '{step.name}' to pipeline")
        return self
    
    def insert_step(self, index: int, step: BuildStep) -> "BuildPipeline":
        """
        Insert a step at specific position in the pipeline.
        
        Args:
            index: Position to insert the step
            step: Build step to insert
            
        Returns:
            Self for method chaining
        """
        self.steps.insert(index, step)
        logger.debug(f"Inserted step '{step.name}' at position {index}")
        return self
    
    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step from the pipeline.
        
        Args:
            step_name: Name of the step to remove
            
        Returns:
            True if step was found and removed
        """
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                removed_step = self.steps.pop(i)
                logger.debug(f"Removed step '{removed_step.name}' from pipeline")
                return True
        return False
    
    async def execute(self, context: "BuildContext") -> KnowledgeGraph:
        """
        Execute the complete build pipeline.
        
        Args:
            context: Build context containing input data and configuration
            
        Returns:
            Assembled knowledge graph
            
        Raises:
            Exception: If pipeline execution fails
        """
        start_time = time.time()
        self._execution_metrics["total_executions"] += 1
        
        try:
            logger.info(f"Starting build pipeline with {len(self.steps)} steps")
            logger.info(f"Input: {len(context.texts)} texts, Graph: '{context.graph_name}'")
            
            # Initialize context timing
            context.total_start_time = start_time
            
            # Execute each step in sequence
            for step in self.steps:
                step_start_time = time.time()
                
                # Check if step should be skipped
                if context.should_skip_step(step.name):
                    logger.info(f"Skipping step '{step.name}' (disabled by configuration)")
                    context.mark_step_skipped(step.name, "disabled by configuration")
                    continue
                
                # Check if step should be executed based on from_step
                if not context.should_execute_step(step.name):
                    logger.info(f"Skipping step '{step.name}' (before from_step)")
                    context.mark_step_skipped(step.name, "before from_step")
                    continue
                
                # Mark step as started
                context.mark_step_started(step.name, step_start_time)
                
                # Execute the step
                logger.info(f"Executing step: {step.name}")
                result = await step.execute(context)
                
                step_end_time = time.time()
                
                # Handle step result
                if result.is_success():
                    # Mark step as completed
                    context.mark_step_completed(
                        step.name,
                        step_end_time,
                        result.data,
                        result.metadata
                    )
                    
                    # Update pipeline metrics
                    self._update_step_metrics(step.name, result.execution_time, True)
                    
                    logger.info(f"Step '{step.name}' completed successfully in {result.execution_time:.2f}s")
                    
                else:
                    # Step failed
                    error_msg = result.error.message if result.error else "Unknown error"
                    logger.error(f"Step '{step.name}' failed: {error_msg}")
                    
                    # Add error to context
                    if result.error and result.error.cause:
                        context.add_error(result.error.cause, step.name)
                    else:
                        context.add_error(Exception(error_msg), step.name)
                    
                    # Update pipeline metrics
                    self._update_step_metrics(step.name, result.execution_time, False)
                    
                    # Stop execution on failure
                    raise result.error.to_exception() if result.error else Exception(f"Step {step.name} failed")
            
            # Get the final knowledge graph from context
            knowledge_graph = context.knowledge_graph
            if not knowledge_graph:
                raise Exception("Pipeline completed but no knowledge graph was created")
            
            # Calculate total execution time
            total_time = time.time() - start_time
            self._execution_metrics["total_execution_time"] += total_time
            self._execution_metrics["successful_executions"] += 1
            
            # Log completion
            logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"Final graph: '{knowledge_graph.name}' with {len(knowledge_graph.entities)} entities, "
                       f"{len(knowledge_graph.relations)} relations")
            
            return knowledge_graph
            
        except Exception as e:
            # Update failure metrics
            total_time = time.time() - start_time
            self._execution_metrics["total_execution_time"] += total_time
            self._execution_metrics["failed_executions"] += 1
            
            logger.error(f"Pipeline failed after {total_time:.2f}s: {str(e)}")
            
            # Update cache manager with error
            self.cache_manager.update_build_status(error_message=str(e))
            
            raise
    
    def get_step_names(self) -> List[str]:
        """Get list of step names in execution order."""
        return [step.name for step in self.steps]
    
    def get_step_by_name(self, step_name: str) -> BuildStep:
        """
        Get step by name.
        
        Args:
            step_name: Name of the step to find
            
        Returns:
            The build step
            
        Raises:
            ValueError: If step not found
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        raise ValueError(f"Step '{step_name}' not found in pipeline")
    
    def has_step(self, step_name: str) -> bool:
        """Check if pipeline contains a step with given name."""
        return any(step.name == step_name for step in self.steps)
    
    def get_pipeline_metrics(self) -> dict:
        """Get pipeline execution metrics."""
        success_rate = 0.0
        if self._execution_metrics["total_executions"] > 0:
            success_rate = (self._execution_metrics["successful_executions"] / 
                          self._execution_metrics["total_executions"]) * 100
        
        avg_execution_time = 0.0
        if self._execution_metrics["total_executions"] > 0:
            avg_execution_time = (self._execution_metrics["total_execution_time"] /
                                self._execution_metrics["total_executions"])
        
        return {
            "pipeline_info": {
                "total_steps": len(self.steps),
                "step_names": self.get_step_names()
            },
            "execution_metrics": {
                "total_executions": self._execution_metrics["total_executions"],
                "successful_executions": self._execution_metrics["successful_executions"],
                "failed_executions": self._execution_metrics["failed_executions"],
                "success_rate_percent": success_rate,
                "total_execution_time": self._execution_metrics["total_execution_time"],
                "average_execution_time": avg_execution_time
            },
            "step_metrics": dict(self._execution_metrics["step_metrics"])
        }
    
    def _update_step_metrics(self, step_name: str, execution_time: float, success: bool) -> None:
        """Update metrics for a specific step."""
        if step_name not in self._execution_metrics["step_metrics"]:
            self._execution_metrics["step_metrics"][step_name] = {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }
        
        step_metrics = self._execution_metrics["step_metrics"][step_name]
        step_metrics["executions"] += 1
        step_metrics["total_time"] += execution_time
        
        if success:
            step_metrics["successes"] += 1
        else:
            step_metrics["failures"] += 1
        
        # Update average time
        step_metrics["average_time"] = step_metrics["total_time"] / step_metrics["executions"]
    
    def reset_metrics(self) -> None:
        """Reset pipeline metrics."""
        self._execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "step_metrics": {}
        }
        logger.debug("Pipeline metrics reset")
    
    def __len__(self) -> int:
        """Get number of steps in pipeline."""
        return len(self.steps)
    
    def __repr__(self) -> str:
        """String representation of pipeline."""
        return f"BuildPipeline(steps={len(self.steps)}, names={self.get_step_names()})"