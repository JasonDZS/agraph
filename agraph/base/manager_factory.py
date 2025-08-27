"""
Manager factory implementation for creating different manager instances.

This module provides concrete implementations of the ManagerFactory interface,
allowing for dependency injection and configuration-based manager creation.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .batch import BatchOperationContext
from .dao import DataAccessLayer, MemoryDataAccessLayer
from .interfaces import (
    BatchOperationManager,
    ClusterManager,
    EntityManager,
    ManagerFactory,
    RelationManager,
    TextChunkManager,
)
from .result import Result
from .unified_managers import (
    UnifiedClusterManager,
    UnifiedEntityManager,
    UnifiedRelationManager,
    UnifiedTextChunkManager,
)

if TYPE_CHECKING:
    from .batch import BatchContext


class DefaultManagerFactory(ManagerFactory):
    """
    Default implementation of ManagerFactory.

    Creates unified manager instances backed by a configurable data access layer.
    Supports different DAO implementations and manager configurations.
    """

    def __init__(self, dao: Optional[DataAccessLayer] = None):
        """
        Initialize the manager factory.

        Args:
            dao: Data access layer to use. If None, creates a default MemoryDataAccessLayer
        """
        self.dao = dao or MemoryDataAccessLayer()

    def create_entity_manager(self, config: Optional[Dict[str, Any]] = None) -> EntityManager:
        """
        Create an entity manager instance.

        Args:
            config: Optional configuration parameters (currently unused)

        Returns:
            EntityManager instance
        """
        return UnifiedEntityManager(self.dao)

    def create_relation_manager(self, config: Optional[Dict[str, Any]] = None) -> RelationManager:
        """
        Create a relation manager instance.

        Args:
            config: Optional configuration parameters (currently unused)

        Returns:
            RelationManager instance
        """
        return UnifiedRelationManager(self.dao)

    def create_cluster_manager(self, config: Optional[Dict[str, Any]] = None) -> ClusterManager:
        """
        Create a cluster manager instance.

        Args:
            config: Optional configuration parameters (currently unused)

        Returns:
            ClusterManager instance
        """
        return UnifiedClusterManager(self.dao)

    def create_text_chunk_manager(
        self, config: Optional[Dict[str, Any]] = None
    ) -> TextChunkManager:
        """
        Create a text chunk manager instance.

        Args:
            config: Optional configuration parameters (currently unused)

        Returns:
            TextChunkManager instance
        """
        return UnifiedTextChunkManager(self.dao)

    def create_batch_operation_manager(
        self, config: Optional[Dict[str, Any]] = None
    ) -> BatchOperationManager:
        """
        Create a batch operation manager instance.

        Args:
            config: Optional configuration parameters (currently unused)

        Returns:
            BatchOperationManager instance
        """
        return DefaultBatchOperationManager(self.dao)


class OptimizedManagerFactory(ManagerFactory):
    """
    Factory for creating optimized manager instances.

    This factory creates managers with enhanced caching, indexing, and performance
    optimizations suitable for production workloads.
    """

    def __init__(self, dao: Optional[DataAccessLayer] = None, cache_size: int = 10000):
        """
        Initialize the optimized manager factory.

        Args:
            dao: Data access layer to use
            cache_size: Maximum cache size for optimized operations
        """
        self.dao = dao or MemoryDataAccessLayer()
        self.cache_size = cache_size

    def create_entity_manager(self, config: Optional[Dict[str, Any]] = None) -> EntityManager:
        """Create an optimized entity manager instance."""
        manager = UnifiedEntityManager(self.dao)
        # Could add caching decorators or other optimizations here
        return manager

    def create_relation_manager(self, config: Optional[Dict[str, Any]] = None) -> RelationManager:
        """Create an optimized relation manager instance."""
        manager = UnifiedRelationManager(self.dao)
        # Could add caching decorators or other optimizations here
        return manager

    def create_cluster_manager(self, config: Optional[Dict[str, Any]] = None) -> ClusterManager:
        """Create an optimized cluster manager instance."""
        manager = UnifiedClusterManager(self.dao)
        # Could add caching decorators or other optimizations here
        return manager

    def create_text_chunk_manager(
        self, config: Optional[Dict[str, Any]] = None
    ) -> TextChunkManager:
        """Create an optimized text chunk manager instance."""
        manager = UnifiedTextChunkManager(self.dao)
        # Could add caching decorators or other optimizations here
        return manager

    def create_batch_operation_manager(
        self, config: Optional[Dict[str, Any]] = None
    ) -> BatchOperationManager:
        """Create an optimized batch operation manager instance."""
        return DefaultBatchOperationManager(self.dao)


class DefaultBatchOperationManager(BatchOperationManager):
    """
    Default implementation of BatchOperationManager.

    Provides transactional operations across multiple managers using the DAO layer's
    transaction support.
    """

    def __init__(self, dao: DataAccessLayer):
        """
        Initialize the batch operation manager.

        Args:
            dao: Data access layer to use for transactions
        """
        self.dao = dao

    def begin_batch(self) -> "Result[BatchContext]":
        """
        Begin a batch operation context.

        Returns:
            Result containing batch context
        """
        try:
            context = BatchOperationContext(self.dao)
            return Result.ok(context)
        except Exception as e:
            return Result.internal_error(e)

    def commit_batch(self, context: "BatchContext") -> "Result[Dict[str, Any]]":
        """
        Commit all operations in a batch.

        Args:
            context: Batch context

        Returns:
            Result containing operation summary
        """
        try:
            if not isinstance(context, BatchOperationContext):
                return Result.invalid_input("Invalid batch context")

            summary = context.commit()
            return Result.ok(summary)
        except Exception as e:
            return Result.internal_error(e)

    def rollback_batch(self, context: "BatchContext") -> "Result[bool]":
        """
        Rollback all operations in a batch.

        Args:
            context: Batch context

        Returns:
            Result indicating success/failure
        """
        try:
            if not isinstance(context, BatchOperationContext):
                return Result.invalid_input("Invalid batch context")

            success = context.rollback()
            return Result.ok(success)
        except Exception as e:
            return Result.internal_error(e)


class ManagerFactoryRegistry:
    """
    Registry for managing multiple factory instances.

    Allows registration of different factory types and retrieval by name,
    enabling flexible manager creation patterns.
    """

    def __init__(self) -> None:
        """Initialize the factory registry."""
        self._factories: Dict[str, ManagerFactory] = {}

        # Register default factories
        self.register_factory("default", DefaultManagerFactory())
        self.register_factory("optimized", OptimizedManagerFactory())

    def register_factory(self, name: str, factory: ManagerFactory) -> None:
        """
        Register a factory instance.

        Args:
            name: Factory name
            factory: Factory instance
        """
        self._factories[name] = factory

    def get_factory(self, name: str) -> Optional[ManagerFactory]:
        """
        Get a factory by name.

        Args:
            name: Factory name

        Returns:
            Factory instance or None if not found
        """
        return self._factories.get(name)

    def list_factories(self) -> list[str]:
        """
        List all registered factory names.

        Returns:
            List of factory names
        """
        return list(self._factories.keys())

    def create_managers(self, factory_name: str = "default") -> Dict[str, Any]:
        """
        Create a complete set of managers using the specified factory.

        Args:
            factory_name: Name of the factory to use

        Returns:
            Dictionary containing all manager instances

        Raises:
            ValueError: If factory name is not found
        """
        factory = self.get_factory(factory_name)
        if not factory:
            raise ValueError(f"Factory '{factory_name}' not found")

        return {
            "entity_manager": factory.create_entity_manager(),
            "relation_manager": factory.create_relation_manager(),
            "cluster_manager": factory.create_cluster_manager(),
            "text_chunk_manager": factory.create_text_chunk_manager(),
            "batch_operation_manager": factory.create_batch_operation_manager(),
        }


# Global factory registry instance
factory_registry = ManagerFactoryRegistry()


def get_manager_factory(name: str = "default") -> ManagerFactory:
    """
    Convenience function to get a manager factory.

    Args:
        name: Factory name

    Returns:
        Factory instance

    Raises:
        ValueError: If factory name is not found
    """
    factory = factory_registry.get_factory(name)
    if not factory:
        raise ValueError(f"Factory '{name}' not found")
    return factory


def create_managers(
    factory_name: str = "default", dao: Optional[DataAccessLayer] = None
) -> Dict[str, Any]:
    """
    Convenience function to create a complete set of managers.

    Args:
        factory_name: Name of the factory to use
        dao: Optional DAO instance to use

    Returns:
        Dictionary containing all manager instances
    """
    if dao:
        # Create factory with custom DAO
        factory: Union[OptimizedManagerFactory, DefaultManagerFactory]
        if factory_name == "optimized":
            factory = OptimizedManagerFactory(dao)
        else:
            factory = DefaultManagerFactory(dao)
        return {
            "entity_manager": factory.create_entity_manager(),
            "relation_manager": factory.create_relation_manager(),
            "cluster_manager": factory.create_cluster_manager(),
            "text_chunk_manager": factory.create_text_chunk_manager(),
            "batch_operation_manager": factory.create_batch_operation_manager(),
        }
    return factory_registry.create_managers(factory_name)
