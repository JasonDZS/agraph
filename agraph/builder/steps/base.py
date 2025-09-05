"""
Base classes for build steps.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from ...logger import logger
from ..cache import CacheManager
from .context import BuildContext

T = TypeVar("T")


@dataclass
@dataclass
class StepError:
    """Represents an error that occurred during step execution."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    cause: Optional[Exception] = None

    @classmethod
    def from_exception(cls, exc: Exception, code: str = "STEP_ERROR") -> "StepError":
        """Create StepError from exception."""
        return cls(code=code, message=str(exc), cause=exc)

    def to_exception(self) -> Exception:
        """Convert to exception."""
        if self.cause:
            return self.cause
        return RuntimeError(f"{self.code}: {self.message}")


@dataclass
class StepResult(Generic[T]):
    """Result of step execution."""

    success: bool
    data: Optional[T] = None
    error: Optional[StepError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    @classmethod
    def success_result(cls, data: T, metadata: Optional[Dict[str, Any]] = None) -> "StepResult[T]":
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def failure_result(cls, error: Union[StepError, str, Exception]) -> "StepResult[T]":
        """Create failure result."""
        if isinstance(error, str):
            error = StepError(code="STEP_FAILURE", message=error)
        elif isinstance(error, Exception):
            error = StepError.from_exception(error)

        return cls(success=False, error=error)

    def is_success(self) -> bool:
        """Check if result is successful."""
        return self.success

    def is_failure(self) -> bool:
        """Check if result is failure."""
        return not self.success


class BuildStep(ABC):
    """Abstract base class for all build steps."""

    def __init__(self, name: str, cache_manager: CacheManager):
        """
        Initialize build step.

        Args:
            name: Step name (should match BuildSteps constants)
            cache_manager: Cache manager instance
        """
        self.name = name
        self.cache_manager = cache_manager
        self._execution_count = 0
        self._total_execution_time = 0.0

    async def execute(self, context: "BuildContext") -> StepResult[Any]:
        """
        Execute the step with caching and state management.

        Args:
            context: Build context containing input data and state

        Returns:
            StepResult containing the step output or error
        """
        start_time = time.time()
        self._execution_count += 1

        try:
            # Log step start
            logger.info(f"Step {self.name}: Starting execution")

            # Update build status
            self.cache_manager.update_build_status(current_step=self.name)

            # Check if we should use cached result
            if self._should_use_cache(context):
                logger.info(f"Step {self.name}: Using cached result")
                cached_result = self._get_cached_result(context)
                if cached_result is not None:
                    # Update execution time and return cached result
                    execution_time = time.time() - start_time
                    self._total_execution_time += execution_time

                    result = StepResult.success_result(
                        cached_result,
                        metadata={
                            "step": self.name,
                            "cached": True,
                            "execution_time": execution_time,
                        },
                    )
                    result.execution_time = execution_time
                    return result

            # Execute the actual step logic
            logger.info(f"Step {self.name}: Executing step logic")
            result = await self._execute_step(context)

            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            self._total_execution_time += execution_time

            if result.is_success():
                # Save result to cache if caching is enabled
                if context.use_cache:
                    self._save_to_cache(context, result.data)

                # Update build status
                self.cache_manager.update_build_status(completed_step=self.name)

                # Add execution metadata
                result.metadata.update(
                    {
                        "step": self.name,
                        "cached": False,
                        "execution_time": execution_time,
                        "execution_count": self._execution_count,
                    }
                )

                logger.info(f"Step {self.name}: Completed successfully in {execution_time:.2f}s")
            else:
                logger.error(
                    f"Step {self.name}: Failed with error: {result.error.message if result.error else 'Unknown error'}"
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time

            logger.error(f"Step {self.name}: Exception occurred: {str(e)}")

            # Update build status with error
            self.cache_manager.update_build_status(error_message=str(e))

            result = StepResult.failure_result(StepError.from_exception(e, f"{self.name.upper()}_ERROR"))
            result.execution_time = execution_time
            return result

    @abstractmethod
    async def _execute_step(self, context: "BuildContext") -> StepResult[Any]:
        """
        Execute the actual step logic.

        This method should be implemented by concrete step classes.

        Args:
            context: Build context containing input data and state

        Returns:
            StepResult containing the step output or error
        """

    def _should_use_cache(self, context: "BuildContext") -> bool:
        """
        Check if cached result should be used.

        Args:
            context: Build context

        Returns:
            True if cached result should be used
        """
        return context.use_cache and not context.should_execute_step(self.name) and self._has_cached_result(context)

    def _has_cached_result(self, context: "BuildContext") -> bool:
        """Check if cached result exists for this step."""
        cache_key = self._get_cache_key(context)
        return self.cache_manager.backend.has(cache_key)

    def _get_cached_result(self, context: "BuildContext") -> Any:
        """Get cached result for this step."""
        input_data = self._get_cache_input_data(context)
        expected_type = self._get_expected_result_type()

        return self.cache_manager.get_step_result(self.name, input_data, expected_type)

    def _save_to_cache(self, context: "BuildContext", result_data: Any) -> None:
        """Save result to cache."""
        input_data = self._get_cache_input_data(context)
        self.cache_manager.save_step_result(self.name, input_data, result_data)

    def _get_cache_key(self, context: "BuildContext") -> str:
        """Get cache key for this step."""
        input_data = self._get_cache_input_data(context)
        return f"{self.name}_{hash(str(input_data))}"

    @abstractmethod
    def _get_cache_input_data(self, context: "BuildContext") -> Any:
        """Get input data for cache key generation."""

    @abstractmethod
    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""

    def get_metrics(self) -> Dict[str, Any]:
        """Get step execution metrics."""
        avg_time = self._total_execution_time / max(self._execution_count, 1)
        return {
            "step": self.name,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time,
        }
