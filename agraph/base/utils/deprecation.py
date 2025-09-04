"""
Deprecation utilities for AGraph.

This module provides utilities for managing the deprecation of legacy components
while maintaining backward compatibility.
"""

import warnings
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class DeprecationLevel:
    """Deprecation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def deprecated(
    reason: str,
    version: str = "0.2.0",
    removal_version: str = "1.0.0",
    alternative: Optional[str] = None,
    level: str = DeprecationLevel.WARNING,
) -> Callable[[F], F]:
    """
    Decorator to mark functions/classes as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version in which deprecation started
        removal_version: Version in which component will be removed
        alternative: Recommended alternative
        level: Deprecation level (info/warning/error)

    Returns:
        Decorated function/class
    """

    def decorator(func_or_class: F) -> F:
        @wraps(func_or_class)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build deprecation message
            name = getattr(func_or_class, "__name__", str(func_or_class))
            message_parts = [
                f"{name} is deprecated as of version {version}",
                f"and will be removed in version {removal_version}.",
            ]

            if reason:
                message_parts.append(f"Reason: {reason}")

            if alternative:
                message_parts.append(f"Use {alternative} instead.")

            message = " ".join(message_parts)

            # Issue appropriate warning
            if level == DeprecationLevel.INFO:
                warnings.warn(message, FutureWarning, stacklevel=2)
            elif level == DeprecationLevel.WARNING:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
            elif level == DeprecationLevel.ERROR:
                raise DeprecationWarning(message)

            return func_or_class(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def deprecation_warning(
    component_name: str,
    reason: str,
    alternative: Optional[str] = None,
    version: str = "0.2.0",
    removal_version: str = "1.0.0",
) -> None:
    """
    Issue a deprecation warning for a component.

    Args:
        component_name: Name of deprecated component
        reason: Reason for deprecation
        alternative: Recommended alternative
        version: Version in which deprecation started
        removal_version: Version in which component will be removed
    """
    message_parts = [
        f"{component_name} is deprecated as of version {version}",
        f"and will be removed in version {removal_version}.",
    ]

    if reason:
        message_parts.append(f"Reason: {reason}")

    if alternative:
        message_parts.append(f"Use {alternative} instead.")

    message = " ".join(message_parts)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


class DeprecationManager:
    """Centralized deprecation management."""

    def __init__(self) -> None:
        self._deprecation_config = {
            # Phase 1: Soft warnings (v0.2.0)
            "KnowledgeGraph": {
                "level": DeprecationLevel.INFO,
                "reason": "Replaced by KnowledgeGraph with 10-100x performance improvement",
                "alternative": "KnowledgeGraph",
                "removal_version": "1.0.0",
            },
            "EntityManager": {
                "level": DeprecationLevel.INFO,
                "reason": "Replaced by unified architecture with enhanced error handling",
                "alternative": "Unified Manager interfaces",
                "removal_version": "1.0.0",
            },
        }

    def check_deprecation(self, component_name: str) -> None:
        """Check if component is deprecated and issue warning."""
        if component_name in self._deprecation_config:
            config = self._deprecation_config[component_name]
            deprecation_warning(
                component_name,
                config["reason"],
                config.get("alternative"),
                removal_version=config["removal_version"],
            )


# Global deprecation manager instance
_deprecation_manager = DeprecationManager()


def check_component_deprecation(component_name: str) -> None:
    """Check if a component is deprecated."""
    _deprecation_manager.check_deprecation(component_name)
