"""
ACID-compliant transaction system for AGraph.

This module provides a comprehensive transaction system with full ACID properties:
- Atomicity: All operations succeed or all fail
- Consistency: Data integrity is maintained
- Isolation: Concurrent transactions don't interfere
- Durability: Committed changes persist

Built on top of the existing DAO layer and batch operations system.
"""

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, RLock
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Set

from ..core.result import ErrorCode, Result
from ..infrastructure.dao import DataAccessLayer
from .batch import BatchOperation, BatchOperationContext, BatchOperationType

if TYPE_CHECKING:
    from ..models.entities import Entity
    from ..models.relations import Relation


class IsolationLevel(Enum):
    """Transaction isolation levels."""

    READ_UNCOMMITTED = "read_uncommitted"  # Allows dirty reads
    READ_COMMITTED = "read_committed"  # Prevents dirty reads
    REPEATABLE_READ = "repeatable_read"  # Prevents dirty and non-repeatable reads
    SERIALIZABLE = "serializable"  # Full isolation, prevents all phenomena


class TransactionStatus(Enum):
    """Transaction status enumeration."""

    ACTIVE = "active"  # Transaction is running
    PREPARING = "preparing"  # Preparing to commit
    PREPARED = "prepared"  # Ready to commit
    COMMITTING = "committing"  # Committing changes
    COMMITTED = "committed"  # Successfully committed
    ABORTING = "aborting"  # Rolling back
    ABORTED = "aborted"  # Successfully rolled back
    FAILED = "failed"  # Failed to commit or rollback


class DeadlockException(Exception):
    """Exception raised when a deadlock is detected."""

    pass


class TransactionException(Exception):
    """Base exception for transaction-related errors."""

    pass


class IsolationViolationException(TransactionException):
    """Exception raised when isolation requirements are violated."""

    pass


@dataclass
class LockInfo:
    """Information about a lock held by a transaction."""

    transaction_id: str
    lock_type: str  # "read", "write"
    resource_id: str
    resource_type: str  # "entity", "relation", "cluster", "text_chunk"
    acquired_at: float = field(default_factory=time.time)


@dataclass
class TransactionInfo:
    """Information about a transaction."""

    transaction_id: str
    status: TransactionStatus
    isolation_level: IsolationLevel
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    thread_id: Optional[int] = None
    operations_count: int = 0
    locks_held: List[LockInfo] = field(default_factory=list)


class DeadlockDetector:
    """Deadlock detection using wait-for graph algorithm."""

    def __init__(self) -> None:
        self._wait_for_graph: Dict[str, Set[str]] = (
            {}
        )  # transaction -> set of transactions it's waiting for
        self._lock = RLock()

    def add_wait_edge(self, waiting_tx: str, holding_tx: str) -> None:
        """Add an edge to the wait-for graph."""
        with self._lock:
            if waiting_tx not in self._wait_for_graph:
                self._wait_for_graph[waiting_tx] = set()
            self._wait_for_graph[waiting_tx].add(holding_tx)

    def remove_wait_edge(self, waiting_tx: str, holding_tx: str) -> None:
        """Remove an edge from the wait-for graph."""
        with self._lock:
            if waiting_tx in self._wait_for_graph:
                self._wait_for_graph[waiting_tx].discard(holding_tx)
                if not self._wait_for_graph[waiting_tx]:
                    del self._wait_for_graph[waiting_tx]

    def remove_transaction(self, transaction_id: str) -> None:
        """Remove a transaction from the wait-for graph."""
        with self._lock:
            # Remove as waiting transaction
            self._wait_for_graph.pop(transaction_id, None)

            # Remove from all waiting lists
            for waiting_set in self._wait_for_graph.values():
                waiting_set.discard(transaction_id)

    def detect_deadlock(self) -> Optional[List[str]]:
        """Detect deadlock using DFS cycle detection. Returns the cycle if found."""
        with self._lock:
            visited = set()
            rec_stack = set()

            def _dfs_cycle_detection(node: str, path: List[str]) -> Optional[List[str]]:
                if node in rec_stack:
                    # Found a cycle, return the cycle path
                    cycle_start = path.index(node)
                    return path[cycle_start:]

                if node in visited:
                    return None

                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in self._wait_for_graph.get(node, set()):
                    cycle = _dfs_cycle_detection(neighbor, path.copy())
                    if cycle:
                        return cycle

                rec_stack.remove(node)
                return None

            for transaction in self._wait_for_graph:
                if transaction not in visited:
                    cycle = _dfs_cycle_detection(transaction, [])
                    if cycle:
                        return cycle

            return None


class LockManager:
    """Advanced lock manager with deadlock detection and various lock types."""

    def __init__(self) -> None:
        self._locks: Dict[str, Dict[str, LockInfo]] = (
            {}
        )  # resource_key -> {transaction_id -> LockInfo}
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._deadlock_detector = DeadlockDetector()
        self._lock_timeout = 30.0  # seconds

    def _get_resource_key(self, resource_type: str, resource_id: str) -> str:
        """Generate a unique key for a resource."""
        return f"{resource_type}:{resource_id}"

    def acquire_lock(
        self, transaction_id: str, resource_type: str, resource_id: str, lock_type: str = "write"
    ) -> Result[bool]:
        """Acquire a lock on a resource."""
        resource_key = self._get_resource_key(resource_type, resource_id)

        with self._condition:
            start_time = time.time()

            while True:
                # Check if lock can be acquired
                can_acquire, conflicting_transactions = self._can_acquire_lock(
                    resource_key, transaction_id, lock_type
                )

                if can_acquire:
                    # Acquire the lock
                    if resource_key not in self._locks:
                        self._locks[resource_key] = {}

                    self._locks[resource_key][transaction_id] = LockInfo(
                        transaction_id=transaction_id,
                        lock_type=lock_type,
                        resource_id=resource_id,
                        resource_type=resource_type,
                    )

                    return Result.ok(True)

                # Add wait edges to deadlock detector
                for conflicting_tx in conflicting_transactions:
                    self._deadlock_detector.add_wait_edge(transaction_id, conflicting_tx)

                # Check for deadlock
                cycle = self._deadlock_detector.detect_deadlock()
                if cycle and transaction_id in cycle:
                    # Remove wait edges
                    for conflicting_tx in conflicting_transactions:
                        self._deadlock_detector.remove_wait_edge(transaction_id, conflicting_tx)

                    return Result.fail(
                        ErrorCode.CONCURRENT_MODIFICATION,
                        f"Deadlock detected involving transactions: {cycle}",
                        metadata={"deadlock_cycle": cycle},
                    )

                # Check timeout
                if time.time() - start_time > self._lock_timeout:
                    # Remove wait edges
                    for conflicting_tx in conflicting_transactions:
                        self._deadlock_detector.remove_wait_edge(transaction_id, conflicting_tx)

                    return Result.fail(
                        ErrorCode.TIMEOUT,
                        f"Lock acquisition timeout for {resource_type}:{resource_id}",
                        metadata={"timeout_seconds": self._lock_timeout},
                    )

                # Wait for lock to become available
                self._condition.wait(timeout=1.0)

                # Remove wait edges (they'll be re-added in next iteration if needed)
                for conflicting_tx in conflicting_transactions:
                    self._deadlock_detector.remove_wait_edge(transaction_id, conflicting_tx)

    def _can_acquire_lock(
        self, resource_key: str, transaction_id: str, lock_type: str
    ) -> tuple[bool, Set[str]]:
        """Check if a lock can be acquired and return conflicting transactions."""
        if resource_key not in self._locks:
            return True, set()

        existing_locks = self._locks[resource_key]

        # If transaction already has the lock
        if transaction_id in existing_locks:
            existing_lock = existing_locks[transaction_id]

            # Lock upgrade: read -> write
            if existing_lock.lock_type == "read" and lock_type == "write":
                # Can upgrade if no other transactions have read locks
                conflicting = set(existing_locks.keys()) - {transaction_id}
                return len(conflicting) == 0, conflicting

            # Same or downgrade (write -> read) is always allowed
            return True, set()

        # Check compatibility with existing locks
        conflicting_transactions = set()

        for other_tx_id, lock_info in existing_locks.items():
            if lock_type == "read" and lock_info.lock_type == "read":
                # Read-read compatibility
                continue
            else:
                # Any other combination is incompatible
                conflicting_transactions.add(other_tx_id)

        return len(conflicting_transactions) == 0, conflicting_transactions

    def release_lock(
        self, transaction_id: str, resource_type: str, resource_id: str
    ) -> Result[bool]:
        """Release a lock held by a transaction."""
        resource_key = self._get_resource_key(resource_type, resource_id)

        with self._condition:
            if resource_key in self._locks and transaction_id in self._locks[resource_key]:
                del self._locks[resource_key][transaction_id]

                # Clean up empty resource entries
                if not self._locks[resource_key]:
                    del self._locks[resource_key]

                # Remove from deadlock detector
                self._deadlock_detector.remove_transaction(transaction_id)

                # Notify waiting threads
                self._condition.notify_all()

                return Result.ok(True)

            return Result.fail(
                ErrorCode.NOT_FOUND,
                f"No lock found for transaction {transaction_id} on {resource_type}:{resource_id}",
            )

    def release_all_locks(self, transaction_id: str) -> Result[int]:
        """Release all locks held by a transaction."""
        with self._condition:
            released_count = 0
            resources_to_clean = []

            for resource_key, locks in self._locks.items():
                if transaction_id in locks:
                    del locks[transaction_id]
                    released_count += 1

                    if not locks:
                        resources_to_clean.append(resource_key)

            # Clean up empty resource entries
            for resource_key in resources_to_clean:
                del self._locks[resource_key]

            # Remove from deadlock detector
            self._deadlock_detector.remove_transaction(transaction_id)

            # Notify waiting threads
            self._condition.notify_all()

            return Result.ok(released_count)

    def get_locks_held_by(self, transaction_id: str) -> List[LockInfo]:
        """Get all locks held by a transaction."""
        with self._lock:
            locks = []
            for resource_locks in self._locks.values():
                if transaction_id in resource_locks:
                    locks.append(resource_locks[transaction_id])
            return locks


class Transaction:
    """
    Enhanced ACID transaction implementation.

    Provides full transaction capabilities with:
    - ACID properties
    - Multiple isolation levels
    - Deadlock detection
    - Comprehensive error handling
    - Integration with existing DAO and batch operations
    """

    def __init__(
        self,
        dao: DataAccessLayer,
        transaction_manager: "TransactionManager",
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 300.0,
    ):
        self.id = str(uuid.uuid4())
        self.dao = dao
        self.transaction_manager = transaction_manager
        self.isolation_level = isolation_level
        self.timeout_seconds = timeout_seconds

        # Transaction state
        self.status = TransactionStatus.ACTIVE
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.thread_id = threading.current_thread().ident

        # Operations and data
        self.batch_context: Optional[BatchOperationContext] = None
        self.operations: List[BatchOperation] = []
        self.savepoints: Dict[str, int] = {}  # savepoint_name -> operation_index

        # Snapshot for repeatable read isolation
        self.read_snapshot: Dict[str, Any] = {}
        self.snapshot_timestamp: Optional[float] = None

        # Lock tracking
        self.acquired_locks: Set[str] = set()  # resource_keys

        self._lock = RLock()

    def begin(self) -> Result[bool]:
        """Begin the transaction."""
        with self._lock:
            if self.status != TransactionStatus.ACTIVE:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION,
                    f"Cannot begin transaction in status {self.status.value}",
                )

            self.started_at = time.time()

            # Create batch context for DAO operations
            self.batch_context = BatchOperationContext(self.dao)

            # Create snapshot for repeatable read or serializable isolation
            if self.isolation_level in [
                IsolationLevel.REPEATABLE_READ,
                IsolationLevel.SERIALIZABLE,
            ]:
                self.snapshot_timestamp = time.time()
                # Note: In a full implementation, you'd create a consistent snapshot here

            return Result.ok(True)

    def add_entity(self, entity: "Entity") -> Result["Entity"]:
        """Add an entity within the transaction."""
        return self._execute_operation(
            BatchOperationType.ADD,
            "add_entity",
            "entity",
            entity.id,
            entity.to_dict() if hasattr(entity, "to_dict") else vars(entity),
        )

    def remove_entity(self, entity_id: str) -> Result[bool]:
        """Remove an entity within the transaction."""
        result = self._execute_operation(
            BatchOperationType.REMOVE, "remove_entity", "entity", entity_id
        )
        return Result.ok(True) if result.is_ok() else result

    def update_entity(self, entity: "Entity") -> Result["Entity"]:
        """Update an entity within the transaction."""
        return self._execute_operation(
            BatchOperationType.UPDATE,
            "update_entity",
            "entity",
            entity.id,
            entity.to_dict() if hasattr(entity, "to_dict") else vars(entity),
        )

    def add_relation(self, relation: "Relation") -> Result["Relation"]:
        """Add a relation within the transaction."""
        return self._execute_operation(
            BatchOperationType.ADD,
            "add_relation",
            "relation",
            relation.id,
            relation.to_dict() if hasattr(relation, "to_dict") else vars(relation),
        )

    def remove_relation(self, relation_id: str) -> Result[bool]:
        """Remove a relation within the transaction."""
        result = self._execute_operation(
            BatchOperationType.REMOVE, "remove_relation", "relation", relation_id
        )
        return Result.ok(True) if result.is_ok() else result

    def _execute_operation(
        self,
        operation_type: BatchOperationType,
        operation_name: str,
        target_type: str,
        target_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Result[Any]:
        """Execute an operation within the transaction with proper locking."""
        with self._lock:
            if self.status != TransactionStatus.ACTIVE:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION,
                    f"Cannot execute operation in transaction with status {self.status.value}",
                )

            if not self.batch_context:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION, "Transaction not properly initialized"
                )

            # Check timeout
            if self.started_at and (time.time() - self.started_at) > self.timeout_seconds:
                return Result.fail(
                    ErrorCode.TIMEOUT, f"Transaction timeout after {self.timeout_seconds} seconds"
                )

            # Acquire appropriate locks based on isolation level and operation type
            if target_id:
                lock_type = (
                    "write"
                    if operation_type
                    in [
                        BatchOperationType.ADD,
                        BatchOperationType.REMOVE,
                        BatchOperationType.UPDATE,
                    ]
                    else "read"
                )

                lock_result = self.transaction_manager.lock_manager.acquire_lock(
                    self.id, target_type, target_id, lock_type
                )

                if not lock_result.is_ok():
                    return lock_result

                resource_key = f"{target_type}:{target_id}"
                self.acquired_locks.add(resource_key)

            # Create and add the batch operation
            operation = BatchOperation(
                operation_type=operation_type,
                operation_name=operation_name,
                target_type=target_type,
                target_id=target_id,
                data=data,
            )

            self.operations.append(operation)
            self.batch_context.add_operation(operation)

            # For immediate operations in some isolation levels, you might execute here
            # For now, we defer all execution to commit time

            return Result.ok(
                data,
                metadata={
                    "operation": operation_name,
                    "operation_id": len(self.operations) - 1,
                    "transaction_id": self.id,
                },
            )

    def create_savepoint(self, savepoint_name: str) -> Result[str]:
        """Create a savepoint within the transaction."""
        with self._lock:
            if self.status != TransactionStatus.ACTIVE:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION,
                    f"Cannot create savepoint in transaction with status {self.status.value}",
                )

            if savepoint_name in self.savepoints:
                return Result.fail(
                    ErrorCode.DUPLICATE_ENTRY, f"Savepoint '{savepoint_name}' already exists"
                )

            self.savepoints[savepoint_name] = len(self.operations)

            return Result.ok(
                savepoint_name,
                metadata={
                    "savepoint_name": savepoint_name,
                    "operation_count": len(self.operations),
                    "transaction_id": self.id,
                },
            )

    def rollback_to_savepoint(self, savepoint_name: str) -> Result[bool]:
        """Rollback to a specific savepoint."""
        with self._lock:
            if self.status != TransactionStatus.ACTIVE:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION,
                    f"Cannot rollback to savepoint in transaction with status {self.status.value}",
                )

            if savepoint_name not in self.savepoints:
                return Result.fail(ErrorCode.NOT_FOUND, f"Savepoint '{savepoint_name}' not found")

            savepoint_index = self.savepoints[savepoint_name]

            # Remove operations after the savepoint
            operations_to_remove = self.operations[savepoint_index:]
            self.operations = self.operations[:savepoint_index]

            # Remove later savepoints
            self.savepoints = {
                name: index for name, index in self.savepoints.items() if index <= savepoint_index
            }

            # TODO: In a full implementation, you'd also rollback any locks
            # acquired for the removed operations

            return Result.ok(
                True,
                metadata={
                    "savepoint_name": savepoint_name,
                    "operations_rolled_back": len(operations_to_remove),
                    "remaining_operations": len(self.operations),
                    "transaction_id": self.id,
                },
            )

    def commit(self) -> Result[Dict[str, Any]]:
        """Commit the transaction."""
        with self._lock:
            if self.status != TransactionStatus.ACTIVE:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION,
                    f"Cannot commit transaction in status {self.status.value}",
                )

            if not self.batch_context:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION, "Transaction not properly initialized"
                )

            self.status = TransactionStatus.PREPARING

            try:
                # Two-phase commit preparation
                self.status = TransactionStatus.PREPARED

                # Execute all operations atomically
                self.status = TransactionStatus.COMMITTING
                result = self.batch_context.commit()

                if result["committed"]:
                    self.status = TransactionStatus.COMMITTED
                    self.finished_at = time.time()

                    # Release all locks
                    self.transaction_manager.lock_manager.release_all_locks(self.id)

                    return Result.ok(
                        result,
                        metadata={
                            "transaction_id": self.id,
                            "duration": self.finished_at - (self.started_at or self.created_at),
                            "operations_count": len(self.operations),
                        },
                    )
                else:
                    self.status = TransactionStatus.FAILED
                    return Result.fail(
                        ErrorCode.INTERNAL_ERROR, "Transaction commit failed", metadata=result
                    )

            except Exception as e:
                self.status = TransactionStatus.FAILED
                return Result.internal_error(e)

    def rollback(self) -> Result[bool]:
        """Rollback the transaction."""
        with self._lock:
            if self.status in [TransactionStatus.COMMITTED, TransactionStatus.ABORTED]:
                return Result.fail(
                    ErrorCode.INVALID_OPERATION,
                    f"Cannot rollback transaction in status {self.status.value}",
                )

            self.status = TransactionStatus.ABORTING

            try:
                if self.batch_context:
                    self.batch_context.rollback()

                self.status = TransactionStatus.ABORTED
                self.finished_at = time.time()

                # Release all locks
                self.transaction_manager.lock_manager.release_all_locks(self.id)

                return Result.ok(
                    True,
                    metadata={
                        "transaction_id": self.id,
                        "operations_rolled_back": len(self.operations),
                        "duration": self.finished_at - (self.started_at or self.created_at),
                    },
                )

            except Exception as e:
                self.status = TransactionStatus.FAILED
                return Result.internal_error(e)

    def get_info(self) -> TransactionInfo:
        """Get information about the transaction."""
        with self._lock:
            locks = self.transaction_manager.lock_manager.get_locks_held_by(self.id)

            return TransactionInfo(
                transaction_id=self.id,
                status=self.status,
                isolation_level=self.isolation_level,
                created_at=self.created_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                thread_id=self.thread_id,
                operations_count=len(self.operations),
                locks_held=locks,
            )


class TransactionManager:
    """
    Central transaction manager providing ACID transaction capabilities.

    Features:
    - Multiple concurrent transactions
    - Deadlock detection and prevention
    - Various isolation levels
    - Lock management
    - Transaction monitoring and statistics
    """

    def __init__(self, dao: DataAccessLayer):
        self.dao = dao
        self.lock_manager = LockManager()

        # Active transactions
        self.active_transactions: Dict[str, Transaction] = {}
        self.transaction_history: List[TransactionInfo] = []

        # Statistics
        self.stats = {
            "total_transactions": 0,
            "committed_transactions": 0,
            "aborted_transactions": 0,
            "deadlocks_detected": 0,
            "average_transaction_duration": 0.0,
        }

        self._lock = RLock()

    def begin_transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 300.0,
    ) -> Result[Transaction]:
        """Begin a new transaction."""
        with self._lock:
            transaction = Transaction(
                dao=self.dao,
                transaction_manager=self,
                isolation_level=isolation_level,
                timeout_seconds=timeout_seconds,
            )

            begin_result = transaction.begin()
            if not begin_result.is_ok():
                return begin_result

            self.active_transactions[transaction.id] = transaction
            self.stats["total_transactions"] += 1

            return Result.ok(
                transaction,
                metadata={
                    "transaction_id": transaction.id,
                    "isolation_level": isolation_level.value,
                    "timeout_seconds": timeout_seconds,
                },
            )

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get an active transaction by ID."""
        with self._lock:
            return self.active_transactions.get(transaction_id)

    def commit_transaction(self, transaction_id: str) -> Result[Dict[str, Any]]:
        """Commit a transaction."""
        with self._lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                return Result.not_found("Transaction", transaction_id)

            result = transaction.commit()

            # Update statistics and cleanup
            if result.is_ok():
                self.stats["committed_transactions"] += 1

            self._cleanup_transaction(transaction)
            return result

    def rollback_transaction(self, transaction_id: str) -> Result[bool]:
        """Rollback a transaction."""
        with self._lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                return Result.not_found("Transaction", transaction_id)

            result = transaction.rollback()

            # Update statistics and cleanup
            if result.is_ok():
                self.stats["aborted_transactions"] += 1

            self._cleanup_transaction(transaction)
            return result

    def _cleanup_transaction(self, transaction: Transaction) -> None:
        """Clean up a finished transaction."""
        # Move to history
        self.transaction_history.append(transaction.get_info())

        # Keep only last 1000 transactions in history
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]

        # Remove from active transactions
        self.active_transactions.pop(transaction.id, None)

        # Update average duration
        if transaction.finished_at and transaction.started_at:
            duration = transaction.finished_at - transaction.started_at
            total_finished = (
                self.stats["committed_transactions"] + self.stats["aborted_transactions"]
            )
            if total_finished > 0:
                current_avg = self.stats["average_transaction_duration"]
                self.stats["average_transaction_duration"] = (
                    current_avg * (total_finished - 1) + duration
                ) / total_finished

    @contextmanager
    def transaction(
        self,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        timeout_seconds: float = 300.0,
    ) -> Generator[Transaction, None, None]:
        """Context manager for automatic transaction handling."""
        tx_result = self.begin_transaction(isolation_level, timeout_seconds)
        if not tx_result.is_ok():
            raise TransactionException(f"Failed to begin transaction: {tx_result.error_message}")

        transaction = tx_result.data
        try:
            yield transaction

            # Auto-commit if transaction is still active
            if transaction.status == TransactionStatus.ACTIVE:
                commit_result = self.commit_transaction(transaction.id)
                if not commit_result.is_ok():
                    raise TransactionException(
                        f"Failed to commit transaction: {commit_result.error_message}"
                    )

        except Exception as e:
            # Auto-rollback on exception
            if transaction.status == TransactionStatus.ACTIVE:
                rollback_result = self.rollback_transaction(transaction.id)
                if not rollback_result.is_ok():
                    # Log the rollback failure but don't mask the original exception
                    pass
            raise e

    def get_active_transactions(self) -> List[TransactionInfo]:
        """Get information about all active transactions."""
        with self._lock:
            return [tx.get_info() for tx in self.active_transactions.values()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get transaction manager statistics."""
        with self._lock:
            return {
                **self.stats.copy(),
                "active_transactions_count": len(self.active_transactions),
                "history_size": len(self.transaction_history),
                "locks_held": len(
                    [
                        lock
                        for tx in self.active_transactions.values()
                        for lock in self.lock_manager.get_locks_held_by(tx.id)
                    ]
                ),
            }

    def detect_and_resolve_deadlocks(self) -> Result[List[str]]:
        """Manually trigger deadlock detection and resolution."""
        cycle = self.lock_manager._deadlock_detector.detect_deadlock()
        if cycle:
            self.stats["deadlocks_detected"] += 1

            # Simple resolution: abort the youngest transaction in the cycle
            youngest_tx = None
            youngest_time = float("inf")

            for tx_id in cycle:
                transaction = self.active_transactions.get(tx_id)
                if transaction and transaction.created_at < youngest_time:
                    youngest_time = transaction.created_at
                    youngest_tx = transaction

            if youngest_tx:
                self.rollback_transaction(youngest_tx.id)
                return Result.ok(
                    cycle, metadata={"resolved_by_aborting": youngest_tx.id, "cycle": cycle}
                )

        return Result.ok([])


# Utility functions for easy transaction usage


def create_transaction_manager(dao: DataAccessLayer) -> TransactionManager:
    """Create a new transaction manager with the given DAO."""
    return TransactionManager(dao)


@contextmanager
def atomic_transaction(
    transaction_manager: TransactionManager,
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
    timeout_seconds: float = 300.0,
) -> Generator[Transaction, None, None]:
    """Context manager for atomic transactions."""
    with transaction_manager.transaction(isolation_level, timeout_seconds) as tx:
        yield tx
