"""
Result response type system for consistent API responses.

This module provides a unified Result type for all manager operations,
ensuring consistent error handling and response format across the AGraph system.
"""

import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class ErrorCode(Enum):
    """Standard error codes for AGraph operations."""

    SUCCESS = 0
    NOT_FOUND = 404
    INVALID_INPUT = 400
    DUPLICATE_ENTRY = 409
    INTERNAL_ERROR = 500
    CONCURRENT_MODIFICATION = 409
    PERMISSION_DENIED = 403
    TIMEOUT = 408
    VALIDATION_ERROR = 422
    DEPENDENCY_ERROR = 424
    INVALID_OPERATION = 400


@dataclass
class ErrorDetail:
    """Detailed error information."""

    field_name: Optional[str] = None
    message: str = ""
    code: Optional[str] = None

    def __post_init__(self) -> None:
        if not hasattr(self, "context"):
            self.context: Dict[str, Any] = {}


@dataclass
class Result(Generic[T]):
    """
    Generic result type for consistent API responses.

    This class provides a unified way to handle both successful results
    and errors across all manager operations, enabling better error handling
    and functional programming patterns.

    Attributes:
        success: Whether the operation was successful
        data: The result data (if successful)
        error_code: Standard error code (if failed)
        error_message: Human-readable error message (if failed)
        error_details: Detailed error information (if failed)
        metadata: Additional metadata about the operation
        timestamp: When the result was created
    """

    success: bool
    data: Optional[T] = None
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if not hasattr(self, "error_details"):
            self.error_details: List[ErrorDetail] = []
        if not hasattr(self, "metadata"):
            self.metadata: Dict[str, Any] = {}
        if not hasattr(self, "timestamp"):
            self.timestamp: datetime = datetime.now()

    @classmethod
    def ok(cls, data: T, metadata: Optional[Dict[str, Any]] = None) -> "Result[T]":
        """Create a successful result."""
        result = cls(success=True, data=data, error_code=ErrorCode.SUCCESS)
        result.metadata = metadata or {}
        return result

    @classmethod
    def fail(
        cls,
        error_code: ErrorCode,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Result[T]":
        """Create a failed result."""
        result = cls(success=False, error_code=error_code, error_message=message)
        result.error_details = details or []
        result.metadata = metadata or {}
        return result

    @classmethod
    def not_found(cls, resource: str, resource_id: str) -> "Result[T]":
        """Create a not found result."""
        return cls.fail(ErrorCode.NOT_FOUND, f"{resource} with ID '{resource_id}' not found")

    @classmethod
    def invalid_input(cls, message: str, field: Optional[str] = None) -> "Result[T]":
        """Create an invalid input result."""
        details = []
        if field:
            details.append(ErrorDetail(field_name=field, message=message))
        return cls.fail(ErrorCode.INVALID_INPUT, message, details=details)

    @classmethod
    def internal_error(cls, exception: Exception) -> "Result[T]":
        """Create an internal error result from exception."""
        return cls.fail(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(exception)}",
            metadata={"traceback": traceback.format_exc()},
        )

    def is_ok(self) -> bool:
        """Check if the result is successful."""
        return self.success

    def is_error(self) -> bool:
        """Check if the result is an error."""
        return not self.success

    def unwrap(self) -> T:
        """
        Unwrap the result data, raising an exception if failed.

        Returns:
            The result data

        Raises:
            ValueError: If the result is not successful
        """
        if not self.success:
            raise ValueError(f"Cannot unwrap failed result: {self.error_message}")
        if self.data is None:
            raise ValueError("Cannot unwrap None data")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """
        Unwrap the result data or return a default value.

        Args:
            default: Default value to return if result failed

        Returns:
            The result data or default value
        """
        return self.data if self.success and self.data is not None else default

    def map(self, func: Callable[[T], U]) -> "Result[U]":
        """
        Apply a function to the result data if successful.

        Args:
            func: Function to apply to the data

        Returns:
            New Result with transformed data or propagated error
        """
        if self.success and self.data is not None:
            try:
                new_data = func(self.data)
                return Result.ok(new_data, self.metadata)
            except Exception as e:
                return Result.internal_error(e)
        else:
            result: Result[U] = Result(
                success=False, error_code=self.error_code, error_message=self.error_message
            )
            result.error_details = self.error_details
            result.metadata = self.metadata
            return result

    def flat_map(self, func: Callable[[T], "Result[U]"]) -> "Result[U]":
        """
        Apply a function that returns a Result to the result data if successful.

        Args:
            func: Function that takes data and returns a Result

        Returns:
            New Result from the function or propagated error
        """
        if self.success and self.data is not None:
            try:
                return func(self.data)
            except Exception as e:
                return Result.internal_error(e)
        else:
            result: Result[U] = Result(
                success=False, error_code=self.error_code, error_message=self.error_message
            )
            result.error_details = self.error_details
            result.metadata = self.metadata
            return result

    def filter(self, predicate: Callable[[T], bool]) -> "Result[T]":
        """
        Filter the result based on a predicate.

        Args:
            predicate: Function to test the data

        Returns:
            Original result if predicate passes, error result otherwise
        """
        if self.success and self.data is not None:
            try:
                if predicate(self.data):
                    return self

                return Result.fail(
                    ErrorCode.VALIDATION_ERROR, "Result data does not satisfy the predicate"
                )
            except Exception as e:
                return Result.internal_error(e)
        else:
            return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        result_dict = {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

        if self.success:
            result_dict["data"] = self.data
        else:
            result_dict.update(
                {
                    "error_code": self.error_code.name if self.error_code else None,
                    "error_message": self.error_message,
                    "error_details": [
                        {
                            "field": detail.field_name,
                            "message": detail.message,
                            "code": detail.code,
                            "context": detail.context,
                        }
                        for detail in self.error_details
                    ],
                }
            )

        return result_dict


class ResultUtils:
    """Utility functions for working with Result objects."""

    @staticmethod
    def combine(*results: Result) -> Result[List[Any]]:
        """
        Combine multiple results into a single result.

        If all results are successful, returns a list of all data.
        If any result fails, returns the first failure.

        Args:
            *results: Results to combine

        Returns:
            Combined result
        """
        data_list = []
        for result in results:
            if not result.is_ok():
                failed_result: Result[List[Any]] = Result(
                    success=False, error_code=result.error_code, error_message=result.error_message
                )
                failed_result.error_details = result.error_details
                failed_result.metadata = result.metadata
                return failed_result
            data_list.append(result.data)

        return Result.ok(data_list)

    @staticmethod
    def sequence(results: List[Result[T]]) -> Result[List[T]]:
        """
        Convert a list of Results into a Result of list.

        Args:
            results: List of results to sequence

        Returns:
            Result containing list of all data or first error
        """
        return ResultUtils.combine(*results)

    @staticmethod
    def traverse(items: List[T], func: Callable[[T], Result[U]]) -> Result[List[U]]:
        """
        Apply a function to each item and collect the results.

        Args:
            items: Items to process
            func: Function to apply to each item

        Returns:
            Result containing list of all processed data or first error
        """
        results = [func(item) for item in items]
        return ResultUtils.sequence(results)


def result_safe(func: Callable[..., T]) -> Callable[..., Result[T]]:
    """
    Decorator to wrap a function to return Result instead of raising exceptions.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that returns Result
    """

    def wrapper(*args: Any, **kwargs: Any) -> Result[T]:
        try:
            result = func(*args, **kwargs)
            return Result.ok(result)
        except Exception as e:
            return Result.internal_error(e)

    return wrapper
