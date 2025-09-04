"""
Concurrent pipeline scheduler for parallel step execution.
"""

import asyncio
import time
from typing import Any, Dict, List, Set, Union
from dataclasses import dataclass, field

from ..base.graphs.optimized import KnowledgeGraph
from ..logger import logger
from .cache import CacheManager
from .concurrency_config import get_concurrency_manager
from .pipeline import BuildPipeline
from .steps.base import BuildStep, StepResult
from .steps.context import BuildContext


@dataclass
class StepDependency:
    """Represents a dependency relationship between pipeline steps."""
    step_name: str
    depends_on: List[str] = field(default_factory=list)
    can_run_in_parallel: bool = True
    priority: int = 0  # Higher priority runs first


class ConcurrentPipeline(BuildPipeline):
    """
    Enhanced pipeline that supports concurrent execution of independent steps.
    
    Features:
    - Automatic dependency resolution
    - Parallel execution of independent steps
    - Resource-aware scheduling
    - Step priority management
    - Dynamic load balancing
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize concurrent pipeline.
        
        Args:
            cache_manager: Cache manager for step coordination
        """
        super().__init__(cache_manager)
        self.concurrency_manager = get_concurrency_manager()
        self._step_dependencies: Dict[str, StepDependency] = {}
        self._parallel_groups: List[List[str]] = []
        self._step_completion_events: Dict[str, asyncio.Event] = {}
    
    def add_step(self, step: BuildStep, depends_on: List[str] = None, can_run_in_parallel: bool = True, priority: int = 0) -> "ConcurrentPipeline":
        """
        Add a step to the pipeline with dependency information.
        
        Args:
            step: Build step to add
            depends_on: List of step names this step depends on
            can_run_in_parallel: Whether this step can run in parallel with others
            priority: Execution priority (higher runs first)
            
        Returns:
            Self for method chaining
        """
        # Add step to base pipeline
        super().add_step(step)
        
        # Register dependency information
        self._step_dependencies[step.name] = StepDependency(
            step_name=step.name,
            depends_on=depends_on or [],
            can_run_in_parallel=can_run_in_parallel,
            priority=priority
        )
        
        # Create completion event for this step
        self._step_completion_events[step.name] = asyncio.Event()
        
        # Rebuild parallel execution groups
        self._build_parallel_groups()
        
        logger.debug(
            f"Added step '{step.name}' with dependencies: {depends_on or []}, "
            f"parallel: {can_run_in_parallel}, priority: {priority}"
        )
        
        return self
    
    def _build_parallel_groups(self):
        """Build groups of steps that can be executed in parallel."""
        self._parallel_groups = []
        
        # Get all step names in dependency order
        ordered_steps = self._topological_sort()
        processed_steps = set()
        
        while processed_steps != set(ordered_steps):
            # Find next group of steps that can run in parallel
            parallel_group = []
            
            for step_name in ordered_steps:
                if step_name in processed_steps:
                    continue
                
                dependency = self._step_dependencies[step_name]
                
                # Check if all dependencies are satisfied
                if all(dep in processed_steps for dep in dependency.depends_on):
                    # Check if step can run in parallel
                    if dependency.can_run_in_parallel:
                        parallel_group.append(step_name)
                    else:
                        # Non-parallel step goes in its own group
                        if not parallel_group:
                            parallel_group.append(step_name)
                        break
            
            if parallel_group:
                # Sort group by priority
                parallel_group.sort(key=lambda name: self._step_dependencies[name].priority, reverse=True)
                self._parallel_groups.append(parallel_group)
                processed_steps.update(parallel_group)
            else:
                # Deadlock or circular dependency detection
                remaining = set(ordered_steps) - processed_steps
                logger.error(f"Cannot resolve dependencies for steps: {remaining}")
                break
        
        logger.info(f"Built {len(self._parallel_groups)} parallel execution groups: {self._parallel_groups}")
    
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort of steps based on dependencies.
        
        Returns:
            List of step names in dependency order
        """
        # Kahn's algorithm for topological sorting
        in_degree = {name: 0 for name in self._step_dependencies.keys()}
        adjacency = {name: [] for name in self._step_dependencies.keys()}
        
        # Build adjacency list and calculate in-degrees
        for step_name, dependency in self._step_dependencies.items():
            for dep in dependency.depends_on:
                if dep in adjacency:
                    adjacency[dep].append(step_name)
                    in_degree[step_name] += 1
        
        # Find all nodes with no incoming edges
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority before processing
            queue.sort(key=lambda name: self._step_dependencies[name].priority, reverse=True)
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent steps
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(self._step_dependencies):
            remaining = set(self._step_dependencies.keys()) - set(result)
            logger.error(f"Circular dependencies detected for steps: {remaining}")
        
        return result
    
    async def execute(self, context: BuildContext) -> KnowledgeGraph:
        """
        Execute the pipeline with concurrent step processing.
        
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
            logger.info(f"Starting concurrent pipeline with {len(self.steps)} steps in {len(self._parallel_groups)} groups")
            logger.info(f"Input: {len(context.texts)} texts, Graph: '{context.graph_name}'")
            
            # Initialize context timing
            context.total_start_time = start_time
            
            # Reset completion events
            for event in self._step_completion_events.values():
                event.clear()
            
            # Execute each parallel group
            for group_idx, parallel_group in enumerate(self._parallel_groups):
                logger.info(f"Executing parallel group {group_idx + 1}: {parallel_group}")
                
                # Filter steps that should actually be executed
                executable_steps = []
                for step_name in parallel_group:
                    step = self.get_step_by_name(step_name)
                    
                    if context.should_skip_step(step_name):
                        logger.info(f"Skipping step '{step_name}' (disabled by configuration)")
                        context.mark_step_skipped(step_name, "disabled by configuration")
                        self._step_completion_events[step_name].set()
                        continue
                    
                    if not context.should_execute_step(step_name):
                        logger.info(f"Skipping step '{step_name}' (before from_step)")
                        context.mark_step_skipped(step_name, "before from_step")
                        self._step_completion_events[step_name].set()
                        continue
                    
                    executable_steps.append(step)
                
                if not executable_steps:
                    logger.info(f"No executable steps in group {group_idx + 1}")
                    continue
                
                # Execute steps in parallel
                if len(executable_steps) == 1:
                    # Single step execution
                    await self._execute_single_step(executable_steps[0], context)
                else:
                    # Parallel execution
                    await self._execute_parallel_group(executable_steps, context)
            
            # Get the final knowledge graph from context
            knowledge_graph = context.knowledge_graph
            if not knowledge_graph:
                raise Exception("Pipeline completed but no knowledge graph was created")
            
            # Calculate total execution time
            total_time = time.time() - start_time
            self._execution_metrics["total_execution_time"] += total_time
            self._execution_metrics["successful_executions"] += 1
            
            # Log completion with concurrency stats
            resource_stats = self.concurrency_manager.get_resource_stats()
            logger.info(f"Concurrent pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"Final graph: '{knowledge_graph.name}' with {len(knowledge_graph.entities)} entities, "
                       f"{len(knowledge_graph.relations)} relations")
            logger.debug(f"Resource utilization: {resource_stats}")
            
            return knowledge_graph
            
        except Exception as e:
            # Update failure metrics
            total_time = time.time() - start_time
            self._execution_metrics["total_execution_time"] += total_time
            self._execution_metrics["failed_executions"] += 1
            
            logger.error(f"Concurrent pipeline failed after {total_time:.2f}s: {str(e)}")
            
            # Update cache manager with error
            self.cache_manager.update_build_status(error_message=str(e))
            
            raise
    
    async def _execute_single_step(self, step: BuildStep, context: BuildContext):
        """Execute a single step."""
        step_start_time = time.time()
        context.mark_step_started(step.name, step_start_time)
        
        logger.info(f"Executing step: {step.name}")
        result = await step.execute(context)
        
        step_end_time = time.time()
        
        if result.is_success():
            context.mark_step_completed(
                step.name,
                step_end_time,
                result.data,
                result.metadata
            )
            self._update_step_metrics(step.name, result.execution_time, True)
            logger.info(f"Step '{step.name}' completed successfully in {result.execution_time:.2f}s")
        else:
            error_msg = result.error.message if result.error else "Unknown error"
            logger.error(f"Step '{step.name}' failed: {error_msg}")
            
            if result.error and result.error.cause:
                context.add_error(result.error.cause, step.name)
            else:
                context.add_error(Exception(error_msg), step.name)
            
            self._update_step_metrics(step.name, result.execution_time, False)
            raise result.error.to_exception() if result.error else Exception(f"Step {step.name} failed")
        
        # Mark step as completed
        self._step_completion_events[step.name].set()
    
    async def _execute_parallel_group(self, steps: List[BuildStep], context: BuildContext):
        """Execute a group of steps in parallel."""
        logger.info(f"Executing {len(steps)} steps in parallel: {[s.name for s in steps]}")
        
        async def execute_step_with_context(step: BuildStep):
            """Execute step with proper context and event management."""
            try:
                await self._execute_single_step(step, context)
            except Exception as e:
                logger.error(f"Parallel step '{step.name}' failed: {str(e)}")
                # Mark as completed even on failure for dependency resolution
                self._step_completion_events[step.name].set()
                raise
        
        # Execute all steps concurrently
        results = await asyncio.gather(
            *[execute_step_with_context(step) for step in steps],
            return_exceptions=True
        )
        
        # Check for failures
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                logger.error(f"Parallel execution failed for step '{step.name}': {result}")
                raise result
    
    def get_execution_plan(self) -> Dict[str, Any]:
        """
        Get detailed execution plan showing parallel groups and dependencies.
        
        Returns:
            Dictionary containing execution plan details
        """
        plan = {
            "total_steps": len(self.steps),
            "parallel_groups": len(self._parallel_groups),
            "execution_groups": [],
            "dependencies": {}
        }
        
        for i, group in enumerate(self._parallel_groups):
            group_info = {
                "group_id": i + 1,
                "steps": group,
                "can_run_parallel": len(group) > 1,
                "estimated_concurrency": min(len(group), max(1, len([s for s in group if self._step_dependencies[s].can_run_in_parallel])))
            }
            plan["execution_groups"].append(group_info)
        
        # Add dependency information
        for step_name, dependency in self._step_dependencies.items():
            plan["dependencies"][step_name] = {
                "depends_on": dependency.depends_on,
                "can_run_in_parallel": dependency.can_run_in_parallel,
                "priority": dependency.priority
            }
        
        return plan