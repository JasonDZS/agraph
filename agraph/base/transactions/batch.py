"""
Batch operation context and management for transactional operations.

This module provides the BatchContext implementation and related utilities
for managing batch operations across multiple managers with transaction support.
Enhanced with integration to the new ACID transaction system.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from ..core.result import ErrorCode, Result
from ..infrastructure.dao import DataAccessLayer
from ..models.clusters import Cluster
from ..models.entities import Entity
from ..models.relations import Relation
from ..models.text import TextChunk

if TYPE_CHECKING:
    from .transaction import IsolationLevel, Transaction, TransactionManager


class BatchOperationError(Exception):
    """Exception raised when batch operations fail."""


class BatchOperationType(Enum):
    """Types of batch operations."""

    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"
    CUSTOM = "custom"


@dataclass
class BatchOperation:
    """
    Represents a single operation in a batch.

    Attributes:
        operation_type: Type of operation
        operation_name: Name/description of the operation
        target_type: Type of target object (entity, relation, etc.)
        target_id: ID of the target object
        data: Operation data
        rollback_data: Data needed for rollback
        executed: Whether the operation has been executed
        success: Whether the operation succeeded
        error: Error message if operation failed
    """

    operation_type: BatchOperationType
    operation_name: str
    target_type: str
    target_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    rollback_data: Optional[Dict[str, Any]] = None
    executed: bool = False
    success: bool = False
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BatchContext:
    """
    Abstract base class for batch operation contexts.

    This class defines the interface that all batch contexts should implement.
    """

    def add_operation(self, operation: BatchOperation) -> None:
        """Add an operation to the batch."""
        raise NotImplementedError

    def execute_operation(self, operation: BatchOperation) -> bool:
        """Execute a single operation."""
        raise NotImplementedError

    def rollback_operation(self, operation: BatchOperation) -> bool:
        """Rollback a single operation."""
        raise NotImplementedError

    def commit(self) -> Dict[str, Any]:
        """Commit all operations in the batch."""
        raise NotImplementedError

    def rollback(self) -> bool:
        """Rollback all executed operations."""
        raise NotImplementedError

    def get_operations(self) -> List[BatchOperation]:
        """Get all operations in the batch."""
        raise NotImplementedError


class BatchOperationContext(BatchContext):
    """
    Concrete implementation of batch operation context.

    Provides transactional batch operations with commit/rollback support
    using the DAO layer's transaction capabilities.
    """

    def __init__(self, dao: "DataAccessLayer"):
        """
        Initialize the batch operation context.

        Args:
            dao: Data access layer for transaction support
        """
        self.dao = dao
        self.operations: List[BatchOperation] = []
        self.committed = False
        self.rolled_back = False
        self.start_time = time.time()
        self._transaction_active = False

    def add_operation(self, operation: BatchOperation) -> None:
        """
        Add an operation to the batch.

        Args:
            operation: Operation to add

        Raises:
            ValueError: If batch is already committed or rolled back
        """
        if self.committed or self.rolled_back:
            raise ValueError("Cannot add operations to committed or rolled back batch")

        self.operations.append(operation)

    def execute_operation(self, operation: BatchOperation) -> bool:
        """
        Execute a single operation.

        Args:
            operation: Operation to execute

        Returns:
            True if successful, False otherwise
        """
        try:
            if operation.operation_type == BatchOperationType.ADD:
                return self._execute_add_operation(operation)
            if operation.operation_type == BatchOperationType.REMOVE:
                return self._execute_remove_operation(operation)
            if operation.operation_type == BatchOperationType.UPDATE:
                return self._execute_update_operation(operation)
            if operation.operation_type == BatchOperationType.CUSTOM:
                return self._execute_custom_operation(operation)

            operation.error = f"Unknown operation type: {operation.operation_type}"
            return False

        except Exception as e:
            operation.error = str(e)
            return False

    def _execute_add_operation(self, operation: BatchOperation) -> bool:
        """Execute an add operation."""
        if operation.target_type == "entity" and operation.data:
            # Store rollback data
            operation.rollback_data = {"action": "remove", "id": operation.target_id}

            # Execute add

            entity = Entity.from_dict(operation.data)
            self.dao.save_entity(entity)
            return True

        if operation.target_type == "relation" and operation.data:
            operation.rollback_data = {"action": "remove", "id": operation.target_id}

            relation = Relation.from_dict(operation.data)
            self.dao.save_relation(relation)
            return True

        if operation.target_type == "cluster" and operation.data:
            operation.rollback_data = {"action": "remove", "id": operation.target_id}

            cluster = Cluster.from_dict(operation.data)
            self.dao.save_cluster(cluster)
            return True

        if operation.target_type == "text_chunk" and operation.data:
            operation.rollback_data = {"action": "remove", "id": operation.target_id}

            chunk = TextChunk.from_dict(operation.data)
            self.dao.save_text_chunk(chunk)
            return True

        return False

    def _execute_remove_operation(self, operation: BatchOperation) -> bool:
        """Execute a remove operation."""
        if not operation.target_id:
            operation.error = "Target ID is required for remove operations"
            return False

        # Store rollback data (the original object)
        if operation.target_type == "entity":
            entity = self.dao.get_entity_by_id(operation.target_id)
            if entity:
                operation.rollback_data = {"action": "add", "data": entity.to_dict()}
                return self.dao.delete_entity(operation.target_id)

        if operation.target_type == "relation":
            relation = self.dao.get_relation_by_id(operation.target_id)
            if relation:
                operation.rollback_data = {"action": "add", "data": relation.to_dict()}
                return self.dao.delete_relation(operation.target_id)

        if operation.target_type == "cluster":
            cluster = self.dao.get_cluster_by_id(operation.target_id)
            if cluster:
                operation.rollback_data = {"action": "add", "data": cluster.to_dict()}
                return self.dao.delete_cluster(operation.target_id)

        if operation.target_type == "text_chunk":
            chunk = self.dao.get_text_chunk_by_id(operation.target_id)
            if chunk:
                operation.rollback_data = {"action": "add", "data": chunk.to_dict()}
                return self.dao.delete_text_chunk(operation.target_id)

        return False

    def _execute_update_operation(self, operation: BatchOperation) -> bool:
        """Execute an update operation."""
        # Update operations would require more sophisticated DAO methods
        # For now, implement as remove + add
        if not operation.target_id or not operation.data:
            operation.error = "Target ID and data are required for update operations"
            return False

        # Store original for rollback
        original_data = None
        if operation.target_type == "entity":
            entity = self.dao.get_entity_by_id(operation.target_id)
            if entity:
                original_data = entity.to_dict()
        # Similar for other types...

        if original_data:
            operation.rollback_data = {"action": "update", "data": original_data}

            # Execute update (simplified as remove + add)
            if self._execute_remove_operation(
                BatchOperation(
                    BatchOperationType.REMOVE,
                    "temp_remove",
                    operation.target_type,
                    operation.target_id,
                )
            ):
                return self._execute_add_operation(
                    BatchOperation(
                        BatchOperationType.ADD,
                        "temp_add",
                        operation.target_type,
                        operation.target_id,
                        operation.data,
                    )
                )

        return False

    def _execute_custom_operation(self, operation: BatchOperation) -> bool:
        """Execute a custom operation."""
        # Custom operations would be defined by the client
        # For now, just mark as successful
        operation.error = "Custom operations not yet implemented"
        return False

    def rollback_operation(self, operation: BatchOperation) -> bool:
        """
        Rollback a single operation.

        Args:
            operation: Operation to rollback

        Returns:
            True if successful, False otherwise
        """
        # pylint: disable=too-many-return-statements
        if not operation.executed or not operation.success or not operation.rollback_data:
            return True  # Nothing to rollback

        try:
            rollback_action = operation.rollback_data.get("action")

            if rollback_action == "remove" and operation.rollback_data.get("id"):
                # Rollback an add operation by removing
                if operation.target_type == "entity":
                    return self.dao.delete_entity(operation.rollback_data["id"])
                if operation.target_type == "relation":
                    return self.dao.delete_relation(operation.rollback_data["id"])
                if operation.target_type == "cluster":
                    return self.dao.delete_cluster(operation.rollback_data["id"])
                if operation.target_type == "text_chunk":
                    return self.dao.delete_text_chunk(operation.rollback_data["id"])

            if rollback_action == "add" and operation.rollback_data.get("data"):
                # Rollback a remove operation by adding back
                data = operation.rollback_data["data"]
                if operation.target_type == "entity":
                    entity = Entity.from_dict(data)
                    self.dao.save_entity(entity)
                    return True
                if operation.target_type == "relation":
                    relation = Relation.from_dict(data)
                    self.dao.save_relation(relation)
                    return True
                if operation.target_type == "cluster":
                    cluster = Cluster.from_dict(data)
                    self.dao.save_cluster(cluster)
                    return True
                if operation.target_type == "text_chunk":
                    chunk = TextChunk.from_dict(data)
                    self.dao.save_text_chunk(chunk)
                    return True

            if rollback_action == "update" and operation.rollback_data.get("data"):
                # Rollback an update by restoring original
                return self._execute_update_operation(
                    BatchOperation(
                        BatchOperationType.UPDATE,
                        "rollback_update",
                        operation.target_type,
                        operation.target_id,
                        operation.rollback_data["data"],
                    )
                )

        except Exception as e:
            operation.error = f"Rollback failed: {str(e)}"
            return False

        return False

    def commit(self) -> Dict[str, Any]:
        """
        Commit all operations in the batch.

        Returns:
            Dictionary containing commit summary

        Raises:
            ValueError: If batch is already committed or rolled back
        """
        if self.committed:
            raise ValueError("Batch is already committed")
        if self.rolled_back:
            raise ValueError("Cannot commit rolled back batch")

        start_time = time.time()
        successful_operations = 0
        failed_operations = 0

        try:
            # Use DAO transaction if available
            with self.dao.transaction() as tx_context:
                self._transaction_active = True

                for operation in self.operations:
                    operation.executed = True
                    operation.success = self.execute_operation(operation)

                    if operation.success:
                        successful_operations += 1
                    else:
                        failed_operations += 1
                        # If any operation fails, raise exception to rollback transaction
                        raise BatchOperationError(f"Operation failed: {operation.error}")

                # Explicit commit for transaction context
                commit_result = tx_context.commit()
                if not commit_result.is_ok():
                    raise BatchOperationError(f"Transaction commit failed: {commit_result.error_message}")

                self.committed = True
                self._transaction_active = False

        except Exception as e:
            self._transaction_active = False
            # Transaction will be automatically rolled back
            raise BatchOperationError(f"Batch commit failed: {str(e)}") from e

        execution_time = time.time() - start_time

        return {
            "committed": self.committed,
            "total_operations": len(self.operations),
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "execution_time": execution_time,
            "batch_start_time": self.start_time,
            "operations": [
                {
                    "type": op.operation_type.value,
                    "name": op.operation_name,
                    "target_type": op.target_type,
                    "target_id": op.target_id,
                    "success": op.success,
                    "error": op.error,
                }
                for op in self.operations
            ],
        }

    def rollback(self) -> bool:
        """
        Rollback all executed operations.

        Returns:
            True if all rollbacks successful, False otherwise
        """
        if self.rolled_back:
            return True
        if self.committed:
            # Need to manually rollback committed operations
            success = True
            # Rollback in reverse order
            for operation in reversed(self.operations):
                if not self.rollback_operation(operation):
                    success = False

            self.rolled_back = True
            return success

        # Not yet committed, just mark as rolled back
        self.rolled_back = True
        return True

    def get_operations(self) -> List[BatchOperation]:
        """
        Get all operations in the batch.

        Returns:
            List of all operations
        """
        return self.operations.copy()

    @contextmanager
    def operation_context(self) -> Generator["BatchOperationContext", None, None]:
        """
        Context manager for batch operations.

        Automatically commits on successful completion or rolls back on exception.
        """
        try:
            yield self
            if not self.committed and not self.rolled_back:
                self.commit()
        except Exception as e:
            if not self.rolled_back:
                self.rollback()
            raise e


# Utility functions for common batch operations


def create_entity_batch_operation(operation_type: BatchOperationType, entity_data: Dict[str, Any]) -> BatchOperation:
    """Create a batch operation for entity management."""
    return BatchOperation(
        operation_type=operation_type,
        operation_name=f"{operation_type.value}_entity",
        target_type="entity",
        target_id=entity_data.get("id"),
        data=entity_data,
    )


def create_relation_batch_operation(
    operation_type: BatchOperationType, relation_data: Dict[str, Any]
) -> BatchOperation:
    """Create a batch operation for relation management."""
    return BatchOperation(
        operation_type=operation_type,
        operation_name=f"{operation_type.value}_relation",
        target_type="relation",
        target_id=relation_data.get("id"),
        data=relation_data,
    )


def create_cluster_batch_operation(operation_type: BatchOperationType, cluster_data: Dict[str, Any]) -> BatchOperation:
    """Create a batch operation for cluster management."""
    return BatchOperation(
        operation_type=operation_type,
        operation_name=f"{operation_type.value}_cluster",
        target_type="cluster",
        target_id=cluster_data.get("id"),
        data=cluster_data,
    )


def create_text_chunk_batch_operation(operation_type: BatchOperationType, chunk_data: Dict[str, Any]) -> BatchOperation:
    """Create a batch operation for text chunk management."""
    return BatchOperation(
        operation_type=operation_type,
        operation_name=f"{operation_type.value}_text_chunk",
        target_type="text_chunk",
        target_id=chunk_data.get("id"),
        data=chunk_data,
    )


class TransactionAwareBatchContext(BatchOperationContext):
    """
    Enhanced batch context with transaction integration.

    Provides additional features:
    - Integration with ACID transactions
    - Savepoint support
    - Enhanced error reporting
    - Performance monitoring
    """

    def __init__(self, dao: DataAccessLayer, transaction_manager: Optional["TransactionManager"] = None):
        super().__init__(dao)
        self.transaction_manager = transaction_manager
        self.current_transaction: Optional["Transaction"] = None
        self.savepoints: Dict[str, int] = {}

        # Enhanced metrics
        self.metrics = {
            "operations_per_second": 0.0,
            "average_operation_time": 0.0,
            "cache_hit_ratio": 0.0,
            "lock_wait_time": 0.0,
        }

    def begin_transaction(
        self, isolation_level: Optional["IsolationLevel"] = None, timeout_seconds: float = 300.0
    ) -> Result["Transaction"]:
        """Begin a new transaction for this batch context."""
        if not self.transaction_manager:
            return Result.fail(ErrorCode.DEPENDENCY_ERROR, "TransactionManager not available")

        if self.current_transaction:
            return Result.fail(ErrorCode.INVALID_OPERATION, "Transaction already active for this batch context")

        # Import here to avoid circular imports
        # pylint: disable=import-outside-toplevel
        from .transaction import IsolationLevel

        if isolation_level is None:
            isolation_level = IsolationLevel.READ_COMMITTED

        tx_result = self.transaction_manager.begin_transaction(
            isolation_level=isolation_level, timeout_seconds=timeout_seconds
        )

        if tx_result.is_ok():
            self.current_transaction = tx_result.data

        return tx_result

    def create_savepoint(self, savepoint_name: str) -> Result[str]:
        """Create a savepoint within the current transaction."""
        if not self.current_transaction:
            return Result.fail(ErrorCode.INVALID_OPERATION, "No active transaction")

        result = self.current_transaction.create_savepoint(savepoint_name)
        if result.is_ok():
            self.savepoints[savepoint_name] = len(self.operations)

        return result

    def rollback_to_savepoint(self, savepoint_name: str) -> Result[bool]:
        """Rollback to a specific savepoint."""
        if not self.current_transaction:
            return Result.fail(ErrorCode.INVALID_OPERATION, "No active transaction")

        if savepoint_name not in self.savepoints:
            return Result.fail(ErrorCode.NOT_FOUND, f"Savepoint '{savepoint_name}' not found in batch context")

        # Rollback in transaction
        tx_result = self.current_transaction.rollback_to_savepoint(savepoint_name)

        if tx_result.is_ok():
            # Also rollback operations in batch context
            savepoint_index = self.savepoints[savepoint_name]
            self.operations = self.operations[:savepoint_index]

            # Remove later savepoints
            self.savepoints = {name: index for name, index in self.savepoints.items() if index <= savepoint_index}

        return tx_result

    def add_operation_with_transaction(self, operation: BatchOperation) -> Result[BatchOperation]:
        """Add an operation that will be executed within the current transaction."""
        # pylint: disable=too-many-return-statements
        if self.committed or self.rolled_back:
            return Result.fail(
                ErrorCode.INVALID_OPERATION,
                "Cannot add operations to committed or rolled back batch",
            )

        # Add to batch operations
        self.operations.append(operation)

        # If we have an active transaction, add to transaction as well
        if self.current_transaction:
            if operation.operation_type == BatchOperationType.ADD:
                if operation.target_type == "entity" and operation.data:
                    entity = Entity.from_dict(operation.data)
                    result = self.current_transaction.add_entity(entity)
                    return (
                        Result.ok(operation)
                        if result.is_ok()
                        else Result.fail(
                            result.error_code or ErrorCode.INTERNAL_ERROR,
                            result.error_message or "Failed to add entity",
                        )
                    )
                if operation.target_type == "relation" and operation.data:
                    relation = Relation.from_dict(operation.data)
                    relation_result = self.current_transaction.add_relation(relation)
                    return (
                        Result.ok(operation)
                        if relation_result.is_ok()
                        else Result.fail(
                            relation_result.error_code or ErrorCode.INTERNAL_ERROR,
                            relation_result.error_message or "Failed to add relation",
                        )
                    )

            elif operation.operation_type == BatchOperationType.REMOVE:
                if operation.target_type == "entity" and operation.target_id:
                    remove_entity_result = self.current_transaction.remove_entity(operation.target_id)
                    return (
                        Result.ok(operation)
                        if remove_entity_result.is_ok()
                        else Result.fail(
                            remove_entity_result.error_code or ErrorCode.INTERNAL_ERROR,
                            remove_entity_result.error_message or "Failed to remove entity",
                        )
                    )
                if operation.target_type == "relation" and operation.target_id:
                    remove_relation_result = self.current_transaction.remove_relation(operation.target_id)
                    return (
                        Result.ok(operation)
                        if remove_relation_result.is_ok()
                        else Result.fail(
                            remove_relation_result.error_code or ErrorCode.INTERNAL_ERROR,
                            remove_relation_result.error_message or "Failed to remove relation",
                        )
                    )

            elif operation.operation_type == BatchOperationType.UPDATE:
                if operation.target_type == "entity" and operation.data:
                    entity = Entity.from_dict(operation.data)
                    result = self.current_transaction.update_entity(entity)
                    return (
                        Result.ok(operation)
                        if result.is_ok()
                        else Result.fail(
                            result.error_code or ErrorCode.INTERNAL_ERROR,
                            result.error_message or "Failed to update entity",
                        )
                    )

        return Result.ok(operation)

    def commit_with_transaction(self) -> Result[Dict[str, Any]]:
        """Commit both the batch operations and the transaction."""
        if self.committed:
            return Result.fail(ErrorCode.INVALID_OPERATION, "Batch is already committed")

        if self.rolled_back:
            return Result.fail(ErrorCode.INVALID_OPERATION, "Cannot commit rolled back batch")

        start_time = time.time()

        try:
            # If we have a transaction, commit it (this will execute all operations)
            if self.current_transaction and self.transaction_manager:
                tx_commit_result = self.transaction_manager.commit_transaction(self.current_transaction.id)

                if not tx_commit_result.is_ok():
                    return Result.fail(
                        ErrorCode.INTERNAL_ERROR,
                        f"Transaction commit failed: {tx_commit_result.error_message}",
                        metadata=tx_commit_result.metadata,
                    )

                # Mark all operations as successful (they were executed by the transaction)
                successful_operations = len(self.operations)
                for operation in self.operations:
                    operation.executed = True
                    operation.success = True

                self.committed = True
                execution_time = time.time() - start_time

                # Update metrics
                if successful_operations > 0:
                    self.metrics["operations_per_second"] = successful_operations / execution_time
                    self.metrics["average_operation_time"] = execution_time / successful_operations

                return Result.ok(
                    {
                        "committed": True,
                        "total_operations": len(self.operations),
                        "successful_operations": successful_operations,
                        "failed_operations": 0,
                        "execution_time": execution_time,
                        "batch_start_time": self.start_time,
                        "transaction_id": self.current_transaction.id,
                        "metrics": self.metrics.copy(),
                        "operations": [
                            {
                                "type": op.operation_type.value,
                                "name": op.operation_name,
                                "target_type": op.target_type,
                                "target_id": op.target_id,
                                "success": op.success,
                                "error": op.error,
                            }
                            for op in self.operations
                        ],
                    }
                )
            # Fall back to regular batch commit
            regular_result = self.commit()
            return Result.ok(regular_result)

        except Exception as e:
            # Rollback transaction if something goes wrong
            if (
                self.current_transaction
                and self.transaction_manager
                and not self.current_transaction.status.value.endswith("ed")
            ):
                self.transaction_manager.rollback_transaction(self.current_transaction.id)

            return Result.internal_error(e)

    def rollback_with_transaction(self) -> Result[bool]:
        """Rollback both the batch operations and the transaction."""
        try:
            # Rollback transaction first
            if self.current_transaction and self.transaction_manager:
                tx_rollback_result = self.transaction_manager.rollback_transaction(self.current_transaction.id)

                if not tx_rollback_result.is_ok():
                    return Result.fail(
                        ErrorCode.INTERNAL_ERROR,
                        f"Transaction rollback failed: {tx_rollback_result.error_message}",
                        metadata=tx_rollback_result.metadata,
                    )

            # Then rollback batch operations
            batch_rollback_result = self.rollback()

            return Result.ok(batch_rollback_result)

        except Exception as e:
            return Result.internal_error(e)

    @contextmanager
    def transactional_batch(
        self, isolation_level: Optional["IsolationLevel"] = None, timeout_seconds: float = 300.0
    ) -> Generator["TransactionAwareBatchContext", None, None]:
        """
        Context manager for transactional batch operations.

        Automatically begins a transaction, commits on success, or rolls back on exception.
        """
        # Import here to avoid circular imports
        # pylint: disable=import-outside-toplevel
        from .transaction import TransactionException

        tx_result = self.begin_transaction(isolation_level, timeout_seconds)
        if not tx_result.is_ok():
            raise TransactionException(f"Failed to begin transaction: {tx_result.error_message}")

        try:
            yield self

            # Auto-commit if not already committed or rolled back
            if not self.committed and not self.rolled_back:
                commit_result = self.commit_with_transaction()
                if not commit_result.is_ok():
                    raise TransactionException(f"Failed to commit batch: {commit_result.error_message}")

        except Exception as e:
            # Auto-rollback on exception
            if not self.rolled_back:
                rollback_result = self.rollback_with_transaction()
                if not rollback_result.is_ok():
                    # Log the rollback failure but don't mask the original exception
                    pass
            raise e
        finally:
            self.current_transaction = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced batch operation metrics."""
        base_metrics: Dict[str, Any] = {
            "total_operations": len(self.operations),
            "committed": self.committed,
            "rolled_back": self.rolled_back,
            "batch_duration": (time.time() - self.start_time if not (self.committed or self.rolled_back) else 0),
            "transaction_active": self.current_transaction is not None,
        }

        if self.current_transaction:
            tx_info = self.current_transaction.get_info()
            base_metrics.update(
                {
                    "transaction_id": str(tx_info.transaction_id),
                    "transaction_status": str(tx_info.status.value),
                    "isolation_level": str(tx_info.isolation_level.value),
                    "locks_held": len(tx_info.locks_held),
                }
            )

        base_metrics.update(self.metrics)
        return base_metrics


# Factory functions for creating transaction-aware batch operations


def create_transactional_batch_context(
    dao: DataAccessLayer, transaction_manager: Optional["TransactionManager"] = None
) -> TransactionAwareBatchContext:
    """Create a transaction-aware batch context."""
    return TransactionAwareBatchContext(dao, transaction_manager)


@contextmanager
def atomic_batch_operations(
    dao: DataAccessLayer,
    transaction_manager: "TransactionManager",
    isolation_level: Optional["IsolationLevel"] = None,
    timeout_seconds: float = 300.0,
) -> Generator[TransactionAwareBatchContext, None, None]:
    """Context manager for atomic batch operations with transaction support."""
    batch_context = create_transactional_batch_context(dao, transaction_manager)

    with batch_context.transactional_batch(isolation_level, timeout_seconds) as ctx:
        yield ctx
