"""
Concurrent-enabled base classes for pipeline steps.
"""

import asyncio
import time
from abc import abstractmethod
from typing import Any, Callable, Generic, List, Optional, TypeVar

from ...logger import logger
from ..cache import CacheManager
from ..concurrency_config import get_concurrency_manager
from .base import BuildStep, StepError, StepResult
from .context import BuildContext

T = TypeVar("T")


class ConcurrentBuildStep(BuildStep, Generic[T]):
    """
    Base class for concurrent-enabled build steps.

    Provides built-in support for:
    - Batch processing with configurable batch sizes
    - Resource-aware concurrency control
    - Automatic timeout handling
    - Memory usage monitoring
    """

    def __init__(self, name: str, cache_manager: CacheManager):
        """
        Initialize concurrent build step.

        Args:
            name: Step name
            cache_manager: Cache manager instance
        """
        super().__init__(name, cache_manager)
        self.concurrency_manager = get_concurrency_manager()
        self._required_resources = self._get_required_resources()

    @abstractmethod
    def _get_required_resources(self) -> List[str]:
        """Return list of required resource types for this step."""

    @abstractmethod
    def _get_batch_size(self) -> int:
        """Return optimal batch size for this step."""

    @abstractmethod
    async def _process_batch(self, batch: List[Any], context: BuildContext) -> List[T]:
        """
        Process a single batch of items.

        Args:
            batch: Batch of items to process
            context: Build context

        Returns:
            List of processed results
        """

    @abstractmethod
    def _prepare_items_for_batching(self, context: BuildContext) -> List[Any]:
        """
        Prepare items from context for batch processing.

        Args:
            context: Build context

        Returns:
            List of items ready for batching
        """

    @abstractmethod
    def _validate_batch_results(self, results: List[T]) -> bool:
        """
        Validate batch processing results.

        Args:
            results: Results from batch processing

        Returns:
            True if results are valid
        """

    async def _execute_step(self, context: BuildContext) -> StepResult[List[T]]:
        """
        Execute step with concurrent batch processing.

        Args:
            context: Build context

        Returns:
            StepResult containing processed results
        """
        try:
            # Prepare items for processing
            items = self._prepare_items_for_batching(context)
            if not items:
                return StepResult.success_result([], metadata={"message": f"No items to process for {self.name}"})

            logger.info(f"Step {self.name}: Processing {len(items)} items with concurrency")

            # Acquire necessary resources
            start_time = time.time()
            acquired_resources = await self.concurrency_manager.acquire_resources(self.name, self._required_resources)

            try:
                # Process items in concurrent batches
                batch_size = self._get_batch_size()
                results = await self.concurrency_manager.batch_process(
                    items=items,
                    processor_func=lambda batch: self._process_batch(batch, context),
                    batch_size=batch_size,
                    resource_type=self.name.lower(),
                )

                # Validate results
                if not self._validate_batch_results(results):
                    return StepResult.failure_result(f"Step {self.name}: Batch processing validation failed")

                execution_time = time.time() - start_time

                return StepResult.success_result(
                    results,
                    metadata={
                        "items_processed": len(items),
                        "results_count": len(results),
                        "batch_size": batch_size,
                        "execution_time": execution_time,
                        "batches_processed": len(items) // batch_size + (1 if len(items) % batch_size else 0),
                    },
                )

            finally:
                # Always release resources
                self.concurrency_manager.release_resources(self.name, acquired_resources)

        except asyncio.TimeoutError:
            return StepResult.failure_result(
                StepError(code="TIMEOUT_ERROR", message=f"Step {self.name} exceeded timeout limit"),
            )
        except Exception as e:
            logger.error(f"Step {self.name}: Execution failed with error: {str(e)}")
            return StepResult.failure_result(StepError.from_exception(e, self.name))


class ParallelProcessingMixin:
    """
    Mixin providing parallel processing utilities for steps.
    """

    async def parallel_map(
        self,
        items: List[Any],
        processor_func: Callable,
        max_concurrency: int = 10,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Apply processor function to items in parallel.

        Args:
            items: Items to process
            processor_func: Function to apply to each item
            max_concurrency: Maximum concurrent operations
            timeout: Timeout per operation

        Returns:
            List of processed results
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_item(item: Any) -> Any:
            async with semaphore:
                if timeout:
                    return await asyncio.wait_for(processor_func(item), timeout=timeout)
                return await processor_func(item)

        return await asyncio.gather(*[process_item(item) for item in items], return_exceptions=True)

    async def parallel_batch_map(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int,
        max_concurrent_batches: int = 5,
        timeout_per_batch: Optional[float] = None,
    ) -> List[Any]:
        """
        Process items in parallel batches.

        Args:
            items: Items to process
            processor_func: Function to apply to each batch
            batch_size: Size of each batch
            max_concurrent_batches: Maximum concurrent batches
            timeout_per_batch: Timeout per batch

        Returns:
            Flattened list of processed results
        """
        # Split into batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        # Process batches in parallel
        batch_results = await self.parallel_map(
            batches,
            processor_func,
            max_concurrency=max_concurrent_batches,
            timeout=timeout_per_batch,
        )

        # Flatten results
        flattened = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.warning(f"Batch processing error: {batch_result}")
                continue
            if isinstance(batch_result, list):
                flattened.extend(batch_result)
            elif batch_result is not None:
                flattened.append(batch_result)

        return flattened


class ResourceAwareStep(ConcurrentBuildStep[T]):
    """
    Step that monitors and adapts to resource availability.
    """

    def __init__(self, name: str, cache_manager: CacheManager):
        super().__init__(name, cache_manager)
        self._adaptive_batch_size = self._get_batch_size()

    def _adapt_batch_size(self) -> int:
        """
        Adapt batch size based on current resource availability.

        Returns:
            Adapted batch size
        """
        stats = self.concurrency_manager.get_resource_stats()

        # Get available resources for this step
        resource_type = self.name.lower()
        available = stats["semaphore_availability"].get(resource_type, 1)

        # Adjust batch size based on availability
        base_batch_size = self._get_batch_size()

        if available <= 1:
            # Low resources, reduce batch size
            self._adaptive_batch_size = max(1, base_batch_size // 2)
        elif available >= 3:
            # High resources, increase batch size
            self._adaptive_batch_size = min(base_batch_size * 2, base_batch_size * available)
        else:
            # Normal resources, use base batch size
            self._adaptive_batch_size = base_batch_size

        logger.debug(
            f"Step {self.name}: Adapted batch size to {self._adaptive_batch_size} "
            f"(available resources: {available})"
        )

        return self._adaptive_batch_size

    async def _execute_step(self, context: BuildContext) -> StepResult[List[T]]:
        """Execute step with adaptive batch sizing."""
        # Adapt batch size before execution
        self._adapt_batch_size()
        return await super()._execute_step(context)
