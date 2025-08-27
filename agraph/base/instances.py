"""Instance management for AGraph components.

This module provides functions to manage and reset cached instances
without creating circular import dependencies.
"""

from typing import Callable, Dict, Optional, Set

# Global instance reset callbacks registry
_reset_callbacks: Dict[str, Callable[[Optional[str]], None]] = {}


def register_reset_callback(name: str, callback: Callable[[Optional[str]], None]) -> None:
    """Register a callback function to be called when instances are reset.

    Args:
        name: Unique name for the callback
        callback: Function that takes an optional project_name and resets instances
    """
    _reset_callbacks[name] = callback


def unregister_reset_callback(name: str) -> None:
    """Unregister a reset callback.

    Args:
        name: Name of the callback to remove
    """
    _reset_callbacks.pop(name, None)


def reset_instances(project_name: Optional[str] = None) -> None:
    """Reset instances for a specific project or all instances.

    Args:
        project_name: If provided, only reset instances for this project.
                     If None, reset all instances.
    """
    for callback in _reset_callbacks.values():
        try:
            callback(project_name)
        except Exception:
            # Ignore errors in individual callbacks to prevent one
            # failing callback from affecting others
            pass


def get_registered_callbacks() -> Set[str]:
    """Get the names of all registered reset callbacks.

    Returns:
        Set of callback names currently registered
    """
    return set(_reset_callbacks.keys())
