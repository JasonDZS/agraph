"""
Concurrency configuration and management for pipeline execution.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional
from ..config import get_settings


@dataclass
class ConcurrencyConfig:
    """Configuration for pipeline concurrency settings."""
    
    # Step-level concurrency limits
    text_chunking_workers: int = 4
    entity_extraction_workers: int = 8  
    relation_extraction_workers: int = 6
    cluster_formation_workers: int = 4
    graph_assembly_workers: int = 2
    
    # Batch processing settings
    entity_batch_size: int = 50
    relation_batch_size: int = 100
    chunk_batch_size: int = 20
    
    # Resource limits
    max_concurrent_llm_calls: int = 10
    max_concurrent_documents: int = 5
    max_memory_usage_mb: int = 2048
    
    # Timeouts (seconds)
    step_timeout: int = 300
    batch_timeout: int = 60
    llm_call_timeout: int = 30
    
    @classmethod
    def from_settings(cls) -> "ConcurrencyConfig":
        """Create configuration from global settings."""
        settings = get_settings()
        
        return cls(
            # Scale based on system capabilities
            text_chunking_workers=min(4, settings.system.max_workers or 4),
            entity_extraction_workers=min(8, (settings.system.max_workers or 4) * 2),
            relation_extraction_workers=min(6, settings.system.max_workers or 4),
            cluster_formation_workers=min(4, settings.system.max_workers or 4),
            graph_assembly_workers=2,  # Keep low to avoid memory issues
            
            # Batch sizes based on model capacity
            entity_batch_size=getattr(settings.llm, 'batch_size', 50),
            relation_batch_size=getattr(settings.llm, 'batch_size', 100) * 2,
            
            # LLM concurrency limits
            max_concurrent_llm_calls=getattr(settings.llm, 'max_concurrent_requests', 10),
        )


class ConcurrencyManager:
    """Manages concurrency resources and semaphores for pipeline execution."""
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """
        Initialize concurrency manager.
        
        Args:
            config: Concurrency configuration, defaults to settings-based config
        """
        self.config = config or ConcurrencyConfig.from_settings()
        
        # Create semaphores for different operation types
        self._semaphores = {
            "text_chunking": asyncio.Semaphore(self.config.text_chunking_workers),
            "entity_extraction": asyncio.Semaphore(self.config.entity_extraction_workers),
            "relation_extraction": asyncio.Semaphore(self.config.relation_extraction_workers),
            "cluster_formation": asyncio.Semaphore(self.config.cluster_formation_workers),
            "graph_assembly": asyncio.Semaphore(self.config.graph_assembly_workers),
            "llm_calls": asyncio.Semaphore(self.config.max_concurrent_llm_calls),
            "documents": asyncio.Semaphore(self.config.max_concurrent_documents),
        }
        
        # Track active tasks and resource usage
        self._active_tasks: Dict[str, set] = {}
        self._resource_usage = {
            "memory_mb": 0,
            "active_llm_calls": 0,
            "active_documents": 0
        }
    
    def get_semaphore(self, resource_type: str) -> asyncio.Semaphore:
        """Get semaphore for specific resource type."""
        return self._semaphores.get(resource_type, asyncio.Semaphore(1))
    
    async def acquire_resources(self, step_name: str, resource_types: list) -> dict:
        """
        Acquire multiple resources for a step execution.
        
        Args:
            step_name: Name of the step acquiring resources
            resource_types: List of resource types to acquire
            
        Returns:
            Dictionary of acquired semaphores
        """
        acquired = {}
        
        try:
            for resource_type in resource_types:
                semaphore = self.get_semaphore(resource_type)
                await semaphore.acquire()
                acquired[resource_type] = semaphore
                
            # Track active tasks
            if step_name not in self._active_tasks:
                self._active_tasks[step_name] = set()
            self._active_tasks[step_name].update(resource_types)
            
            return acquired
            
        except Exception as e:
            # Release any acquired resources on failure
            for resource_type, semaphore in acquired.items():
                semaphore.release()
            raise e
    
    def release_resources(self, step_name: str, acquired_resources: dict):
        """Release acquired resources."""
        for resource_type, semaphore in acquired_resources.items():
            semaphore.release()
        
        # Update tracking
        if step_name in self._active_tasks:
            self._active_tasks[step_name] -= set(acquired_resources.keys())
            if not self._active_tasks[step_name]:
                del self._active_tasks[step_name]
    
    def get_resource_stats(self) -> dict:
        """Get current resource utilization statistics."""
        return {
            "active_tasks": dict(self._active_tasks),
            "resource_usage": self._resource_usage.copy(),
            "semaphore_availability": {
                name: sem._value for name, sem in self._semaphores.items()
            }
        }
    
    async def batch_process(self, items, processor_func, batch_size: int, resource_type: str):
        """
        Process items in concurrent batches.
        
        Args:
            items: Items to process
            processor_func: Async function to process each batch
            batch_size: Size of each batch
            resource_type: Type of resource to use for concurrency control
            
        Returns:
            List of processed results
        """
        semaphore = self.get_semaphore(resource_type)
        
        # Split items into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        async def process_batch(batch):
            async with semaphore:
                return await asyncio.wait_for(
                    processor_func(batch), 
                    timeout=self.config.batch_timeout
                )
        
        # Process all batches concurrently
        results = await asyncio.gather(
            *[process_batch(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Flatten successful results
        flattened_results = []
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue with other batches
                print(f"Batch processing error: {result}")
            elif isinstance(result, list):
                flattened_results.extend(result)
            elif result is not None:
                flattened_results.append(result)
        
        return flattened_results


# Global concurrency manager instance
_concurrency_manager = None


def get_concurrency_manager() -> ConcurrencyManager:
    """Get global concurrency manager instance."""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ConcurrencyManager()
    return _concurrency_manager


def set_concurrency_manager(manager: ConcurrencyManager):
    """Set global concurrency manager instance."""
    global _concurrency_manager
    _concurrency_manager = manager