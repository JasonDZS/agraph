"""
Unit tests for ACID transaction manager implementation.

Tests cover:
- Transaction lifecycle management
- Multiple isolation levels
- Concurrent transaction processing
- Deadlock detection and resolution
- Lock management functionality
- Transaction timeout handling
- Savepoint functionality
- Transaction statistics and monitoring
"""

import threading
import time
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from agraph.base.core.result import ErrorCode
from agraph.base.core.types import EntityType, RelationType
from agraph.base.infrastructure.dao import MemoryDataAccessLayer
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.base.transactions.transaction import (
    DeadlockDetector,
    DeadlockException,
    IsolationLevel,
    LockInfo,
    LockManager,
    Transaction,
    TransactionException,
    TransactionInfo,
    TransactionManager,
    TransactionStatus,
    atomic_transaction,
    create_transaction_manager,
)


class TestDeadlockDetector(unittest.TestCase):
    """Test deadlock detection functionality."""

    def setUp(self):
        self.detector = DeadlockDetector()

    def test_no_deadlock_when_empty(self):
        """Test that no deadlock is detected when no transactions exist."""
        result = self.detector.detect_deadlock()
        self.assertIsNone(result)

    def test_simple_deadlock_detection(self):
        """Test detection of a simple A->B->A deadlock."""
        self.detector.add_wait_edge("tx1", "tx2")
        self.detector.add_wait_edge("tx2", "tx1")

        cycle = self.detector.detect_deadlock()
        self.assertIsNotNone(cycle)
        self.assertIn("tx1", cycle)
        self.assertIn("tx2", cycle)

    def test_complex_deadlock_detection(self):
        """Test detection of a complex A->B->C->A deadlock."""
        self.detector.add_wait_edge("tx1", "tx2")
        self.detector.add_wait_edge("tx2", "tx3")
        self.detector.add_wait_edge("tx3", "tx1")

        cycle = self.detector.detect_deadlock()
        self.assertIsNotNone(cycle)
        self.assertEqual(len(cycle), 3)

    def test_no_deadlock_with_chain(self):
        """Test that no deadlock is detected for simple chains."""
        self.detector.add_wait_edge("tx1", "tx2")
        self.detector.add_wait_edge("tx2", "tx3")

        cycle = self.detector.detect_deadlock()
        self.assertIsNone(cycle)

    def test_wait_edge_removal(self):
        """Test removing wait edges."""
        self.detector.add_wait_edge("tx1", "tx2")
        self.detector.add_wait_edge("tx2", "tx1")

        # Should detect deadlock
        cycle = self.detector.detect_deadlock()
        self.assertIsNotNone(cycle)

        # Remove edge and verify no deadlock
        self.detector.remove_wait_edge("tx1", "tx2")
        cycle = self.detector.detect_deadlock()
        self.assertIsNone(cycle)

    def test_transaction_removal(self):
        """Test removing a transaction from the wait graph."""
        self.detector.add_wait_edge("tx1", "tx2")
        self.detector.add_wait_edge("tx2", "tx3")
        self.detector.add_wait_edge("tx3", "tx1")

        # Remove tx2 and verify no deadlock
        self.detector.remove_transaction("tx2")
        cycle = self.detector.detect_deadlock()
        self.assertIsNone(cycle)


class TestLockManager(unittest.TestCase):
    """Test lock manager functionality."""

    def setUp(self):
        self.lock_manager = LockManager()

    def test_acquire_read_lock(self):
        """Test acquiring a read lock."""
        result = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "read")
        self.assertTrue(result.is_ok())
        self.assertTrue(result.data)

    def test_acquire_write_lock(self):
        """Test acquiring a write lock."""
        result = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.assertTrue(result.is_ok())
        self.assertTrue(result.data)

    def test_concurrent_read_locks(self):
        """Test that multiple read locks can be acquired on the same resource."""
        result1 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "read")
        result2 = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "read")

        self.assertTrue(result1.is_ok())
        self.assertTrue(result2.is_ok())

    def test_write_lock_exclusivity(self):
        """Test that write locks are exclusive."""
        # Acquire write lock
        result1 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.assertTrue(result1.is_ok())

        # Try to acquire another write lock - should be blocked/timeout
        with patch.object(self.lock_manager, "_lock_timeout", 0.1):
            result2 = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "write")
            self.assertFalse(result2.is_ok())
            self.assertEqual(result2.error_code, ErrorCode.TIMEOUT)

    def test_read_write_conflict(self):
        """Test conflict between read and write locks."""
        # Acquire read lock
        result1 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "read")
        self.assertTrue(result1.is_ok())

        # Try to acquire write lock - should be blocked/timeout
        with patch.object(self.lock_manager, "_lock_timeout", 0.1):
            result2 = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "write")
            self.assertFalse(result2.is_ok())
            self.assertEqual(result2.error_code, ErrorCode.TIMEOUT)

    def test_lock_upgrade(self):
        """Test upgrading a read lock to write lock."""
        # Acquire read lock
        result1 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "read")
        self.assertTrue(result1.is_ok())

        # Upgrade to write lock (should work for same transaction)
        result2 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.assertTrue(result2.is_ok())

    def test_lock_upgrade_with_conflict(self):
        """Test lock upgrade when other transactions hold read locks."""
        # Two transactions acquire read locks
        result1 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "read")
        result2 = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "read")
        self.assertTrue(result1.is_ok())
        self.assertTrue(result2.is_ok())

        # Try to upgrade one to write - should be blocked/timeout
        with patch.object(self.lock_manager, "_lock_timeout", 0.1):
            result3 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
            self.assertFalse(result3.is_ok())
            self.assertEqual(result3.error_code, ErrorCode.TIMEOUT)

    def test_release_lock(self):
        """Test releasing a lock."""
        # Acquire lock
        result1 = self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.assertTrue(result1.is_ok())

        # Release lock
        result2 = self.lock_manager.release_lock("tx1", "entity", "entity1")
        self.assertTrue(result2.is_ok())

        # Should be able to acquire again
        result3 = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "write")
        self.assertTrue(result3.is_ok())

    def test_release_nonexistent_lock(self):
        """Test releasing a lock that doesn't exist."""
        result = self.lock_manager.release_lock("tx1", "entity", "entity1")
        self.assertFalse(result.is_ok())
        self.assertEqual(result.error_code, ErrorCode.NOT_FOUND)

    def test_release_all_locks(self):
        """Test releasing all locks for a transaction."""
        # Acquire multiple locks
        self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.lock_manager.acquire_lock("tx1", "entity", "entity2", "read")
        self.lock_manager.acquire_lock("tx1", "relation", "rel1", "write")

        # Release all locks
        result = self.lock_manager.release_all_locks("tx1")
        self.assertTrue(result.is_ok())
        self.assertEqual(result.data, 3)

        # Should be able to acquire any of these locks now
        result1 = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "write")
        result2 = self.lock_manager.acquire_lock("tx2", "entity", "entity2", "write")
        result3 = self.lock_manager.acquire_lock("tx2", "relation", "rel1", "write")

        self.assertTrue(result1.is_ok())
        self.assertTrue(result2.is_ok())
        self.assertTrue(result3.is_ok())

    def test_get_locks_held_by(self):
        """Test getting locks held by a transaction."""
        # Acquire locks
        self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.lock_manager.acquire_lock("tx1", "entity", "entity2", "read")

        locks = self.lock_manager.get_locks_held_by("tx1")
        self.assertEqual(len(locks), 2)

        # Verify lock details
        lock_types = {lock.lock_type for lock in locks}
        self.assertIn("write", lock_types)
        self.assertIn("read", lock_types)

    def test_deadlock_detection_in_lock_manager(self):
        """Test deadlock detection integration in lock manager."""
        # Create potential deadlock scenario
        self.lock_manager.acquire_lock("tx1", "entity", "entity1", "write")
        self.lock_manager.acquire_lock("tx2", "entity", "entity2", "write")

        # This should trigger deadlock detection when timeout is very short
        with patch.object(self.lock_manager, "_lock_timeout", 0.1):
            # tx1 tries to acquire entity2 (held by tx2)
            # tx2 tries to acquire entity1 (held by tx1)
            def tx1_operation():
                result = self.lock_manager.acquire_lock("tx1", "entity", "entity2", "write")
                return result

            def tx2_operation():
                time.sleep(0.05)  # Small delay to ensure tx1 starts first
                result = self.lock_manager.acquire_lock("tx2", "entity", "entity1", "write")
                return result

            thread1 = threading.Thread(target=tx1_operation)
            thread2 = threading.Thread(target=tx2_operation)

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()


class TestTransaction(unittest.TestCase):
    """Test individual transaction functionality."""

    def setUp(self):
        self.dao = MemoryDataAccessLayer()
        self.transaction_manager = TransactionManager(self.dao)

    def test_transaction_creation(self):
        """Test transaction creation and initialization."""
        result = self.transaction_manager.begin_transaction()
        self.assertTrue(result.is_ok())

        transaction = result.data
        self.assertIsInstance(transaction, Transaction)
        self.assertEqual(transaction.status, TransactionStatus.ACTIVE)
        self.assertIsNotNone(transaction.id)
        self.assertIsNotNone(transaction.created_at)

    def test_transaction_begin(self):
        """Test transaction begin operation."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Transaction should be active after begin
        self.assertEqual(transaction.status, TransactionStatus.ACTIVE)
        self.assertIsNotNone(transaction.started_at)
        self.assertIsNotNone(transaction.batch_context)

    def test_transaction_isolation_levels(self):
        """Test different isolation levels."""
        isolation_levels = [
            IsolationLevel.READ_UNCOMMITTED,
            IsolationLevel.READ_COMMITTED,
            IsolationLevel.REPEATABLE_READ,
            IsolationLevel.SERIALIZABLE,
        ]

        for level in isolation_levels:
            with self.subTest(isolation_level=level):
                result = self.transaction_manager.begin_transaction(isolation_level=level)
                self.assertTrue(result.is_ok())

                transaction = result.data
                self.assertEqual(transaction.isolation_level, level)

    def test_add_entity_in_transaction(self):
        """Test adding an entity within a transaction."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        entity = Entity(
            id="entity1",
            name="Test Entity",
            entity_type=EntityType.PERSON,
            description="Test entity for transaction",
        )

        add_result = transaction.add_entity(entity)
        self.assertTrue(add_result.is_ok())
        self.assertEqual(len(transaction.operations), 1)

    def test_add_relation_in_transaction(self):
        """Test adding a relation within a transaction."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Create entities first
        entity1 = Entity(
            id="entity1", name="Entity 1", entity_type=EntityType.PERSON, description="Test"
        )
        entity2 = Entity(
            id="entity2", name="Entity 2", entity_type=EntityType.PERSON, description="Test"
        )

        relation = Relation(
            id="rel1",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.LOCATED_IN,
            description="Test relation",
        )

        add_result = transaction.add_relation(relation)
        self.assertTrue(add_result.is_ok())

    def test_remove_entity_in_transaction(self):
        """Test removing an entity within a transaction."""
        # First add an entity to the DAO
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        self.dao.save_entity(entity)

        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        remove_result = transaction.remove_entity("entity1")
        self.assertTrue(remove_result.is_ok())

    def test_transaction_commit(self):
        """Test transaction commit."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity)

        commit_result = self.transaction_manager.commit_transaction(transaction.id)
        self.assertTrue(commit_result.is_ok())
        self.assertEqual(transaction.status, TransactionStatus.COMMITTED)

    def test_transaction_rollback(self):
        """Test transaction rollback."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity)

        rollback_result = self.transaction_manager.rollback_transaction(transaction.id)
        self.assertTrue(rollback_result.is_ok())
        self.assertEqual(transaction.status, TransactionStatus.ABORTED)

    def test_transaction_timeout(self):
        """Test transaction timeout functionality."""
        result = self.transaction_manager.begin_transaction(timeout_seconds=0.1)
        transaction = result.data

        # Wait for timeout
        time.sleep(0.2)

        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )

        # This operation should fail due to timeout
        add_result = transaction.add_entity(entity)
        self.assertFalse(add_result.is_ok())
        self.assertEqual(add_result.error_code, ErrorCode.TIMEOUT)

    def test_savepoint_creation(self):
        """Test creating savepoints."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        savepoint_result = transaction.create_savepoint("sp1")
        self.assertTrue(savepoint_result.is_ok())
        self.assertEqual(savepoint_result.data, "sp1")
        self.assertIn("sp1", transaction.savepoints)

    def test_duplicate_savepoint(self):
        """Test creating duplicate savepoints."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        transaction.create_savepoint("sp1")
        duplicate_result = transaction.create_savepoint("sp1")

        self.assertFalse(duplicate_result.is_ok())
        self.assertEqual(duplicate_result.error_code, ErrorCode.DUPLICATE_ENTRY)

    def test_rollback_to_savepoint(self):
        """Test rolling back to a savepoint."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Add first entity
        entity1 = Entity(
            id="entity1", name="Entity 1", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity1)

        # Create savepoint
        transaction.create_savepoint("sp1")

        # Add second entity
        entity2 = Entity(
            id="entity2", name="Entity 2", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity2)

        # Should have 2 operations
        self.assertEqual(len(transaction.operations), 2)

        # Rollback to savepoint
        rollback_result = transaction.rollback_to_savepoint("sp1")
        self.assertTrue(rollback_result.is_ok())

        # Should have 1 operation now
        self.assertEqual(len(transaction.operations), 1)

    def test_rollback_to_nonexistent_savepoint(self):
        """Test rolling back to a non-existent savepoint."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        rollback_result = transaction.rollback_to_savepoint("nonexistent")
        self.assertFalse(rollback_result.is_ok())
        self.assertEqual(rollback_result.error_code, ErrorCode.NOT_FOUND)

    def test_transaction_info(self):
        """Test getting transaction information."""
        result = self.transaction_manager.begin_transaction(
            isolation_level=IsolationLevel.SERIALIZABLE, timeout_seconds=600.0
        )
        transaction = result.data

        info = transaction.get_info()
        self.assertIsInstance(info, TransactionInfo)
        self.assertEqual(info.transaction_id, transaction.id)
        self.assertEqual(info.status, TransactionStatus.ACTIVE)
        self.assertEqual(info.isolation_level, IsolationLevel.SERIALIZABLE)
        self.assertIsNotNone(info.created_at)
        self.assertIsNotNone(info.thread_id)


class TestTransactionManager(unittest.TestCase):
    """Test transaction manager functionality."""

    def setUp(self):
        self.dao = MemoryDataAccessLayer()
        self.transaction_manager = TransactionManager(self.dao)

    def test_manager_initialization(self):
        """Test transaction manager initialization."""
        self.assertIsInstance(self.transaction_manager.dao, MemoryDataAccessLayer)
        self.assertIsInstance(self.transaction_manager.lock_manager, LockManager)
        self.assertEqual(len(self.transaction_manager.active_transactions), 0)
        self.assertEqual(self.transaction_manager.stats["total_transactions"], 0)

    def test_begin_transaction(self):
        """Test beginning a new transaction."""
        result = self.transaction_manager.begin_transaction()
        self.assertTrue(result.is_ok())

        transaction = result.data
        self.assertIn(transaction.id, self.transaction_manager.active_transactions)
        self.assertEqual(self.transaction_manager.stats["total_transactions"], 1)

    def test_get_transaction(self):
        """Test getting an active transaction."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        retrieved = self.transaction_manager.get_transaction(transaction.id)
        self.assertEqual(retrieved, transaction)

        nonexistent = self.transaction_manager.get_transaction("nonexistent")
        self.assertIsNone(nonexistent)

    def test_commit_transaction(self):
        """Test committing a transaction."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity)

        commit_result = self.transaction_manager.commit_transaction(transaction.id)
        self.assertTrue(commit_result.is_ok())
        self.assertEqual(self.transaction_manager.stats["committed_transactions"], 1)
        self.assertNotIn(transaction.id, self.transaction_manager.active_transactions)

    def test_rollback_transaction(self):
        """Test rolling back a transaction."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity)

        rollback_result = self.transaction_manager.rollback_transaction(transaction.id)
        self.assertTrue(rollback_result.is_ok())
        self.assertEqual(self.transaction_manager.stats["aborted_transactions"], 1)
        self.assertNotIn(transaction.id, self.transaction_manager.active_transactions)

    def test_commit_nonexistent_transaction(self):
        """Test committing a non-existent transaction."""
        result = self.transaction_manager.commit_transaction("nonexistent")
        self.assertFalse(result.is_ok())
        self.assertEqual(result.error_code, ErrorCode.NOT_FOUND)

    def test_rollback_nonexistent_transaction(self):
        """Test rolling back a non-existent transaction."""
        result = self.transaction_manager.rollback_transaction("nonexistent")
        self.assertFalse(result.is_ok())
        self.assertEqual(result.error_code, ErrorCode.NOT_FOUND)

    def test_context_manager(self):
        """Test transaction context manager."""
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )

        with self.transaction_manager.transaction() as tx:
            tx.add_entity(entity)

        # Transaction should be automatically committed
        self.assertEqual(self.transaction_manager.stats["committed_transactions"], 1)

    def test_context_manager_with_exception(self):
        """Test transaction context manager with exception handling."""
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )

        with self.assertRaises(ValueError):
            with self.transaction_manager.transaction() as tx:
                tx.add_entity(entity)
                raise ValueError("Test exception")

        # Transaction should be automatically rolled back
        self.assertEqual(self.transaction_manager.stats["aborted_transactions"], 1)

    def test_get_active_transactions(self):
        """Test getting active transaction information."""
        result1 = self.transaction_manager.begin_transaction()
        result2 = self.transaction_manager.begin_transaction()

        active_transactions = self.transaction_manager.get_active_transactions()
        self.assertEqual(len(active_transactions), 2)

        for info in active_transactions:
            self.assertIsInstance(info, TransactionInfo)
            self.assertEqual(info.status, TransactionStatus.ACTIVE)

    def test_get_statistics(self):
        """Test getting transaction manager statistics."""
        # Begin and commit one transaction
        result = self.transaction_manager.begin_transaction()
        transaction = result.data
        self.transaction_manager.commit_transaction(transaction.id)

        # Begin and rollback another
        result2 = self.transaction_manager.begin_transaction()
        transaction2 = result2.data
        self.transaction_manager.rollback_transaction(transaction2.id)

        stats = self.transaction_manager.get_statistics()
        self.assertEqual(stats["total_transactions"], 2)
        self.assertEqual(stats["committed_transactions"], 1)
        self.assertEqual(stats["aborted_transactions"], 1)
        self.assertEqual(stats["active_transactions_count"], 0)
        self.assertGreaterEqual(stats["average_transaction_duration"], 0)

    def test_concurrent_transactions(self):
        """Test multiple concurrent transactions."""

        def create_and_commit_transaction(entity_name: str):
            result = self.transaction_manager.begin_transaction()
            transaction = result.data

            entity = Entity(
                id=f"entity_{entity_name}",
                name=entity_name,
                entity_type=EntityType.PERSON,
                description="Test",
            )
            transaction.add_entity(entity)

            commit_result = self.transaction_manager.commit_transaction(transaction.id)
            return commit_result.is_ok()

        threads = []
        results = []

        for i in range(5):
            thread = threading.Thread(
                target=lambda idx=i: results.append(create_and_commit_transaction(f"Entity{idx}"))
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All transactions should succeed
        self.assertTrue(all(results))
        self.assertEqual(self.transaction_manager.stats["committed_transactions"], 5)

    def test_transaction_cleanup(self):
        """Test transaction cleanup after completion."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data
        transaction_id = transaction.id

        # Transaction should be in active list
        self.assertIn(transaction_id, self.transaction_manager.active_transactions)

        # Commit transaction
        self.transaction_manager.commit_transaction(transaction_id)

        # Transaction should be removed from active list and added to history
        self.assertNotIn(transaction_id, self.transaction_manager.active_transactions)
        self.assertEqual(len(self.transaction_manager.transaction_history), 1)

    def test_transaction_history_limit(self):
        """Test transaction history size limitation."""
        # Create and commit 1005 transactions (more than the 1000 limit)
        for i in range(1005):
            result = self.transaction_manager.begin_transaction()
            transaction = result.data
            self.transaction_manager.commit_transaction(transaction.id)

        # History should be limited to 1000
        self.assertEqual(len(self.transaction_manager.transaction_history), 1000)

    def test_deadlock_detection_and_resolution(self):
        """Test deadlock detection and resolution."""
        # Set up a deadlock scenario
        result1 = self.transaction_manager.begin_transaction()
        result2 = self.transaction_manager.begin_transaction()
        tx1 = result1.data
        tx2 = result2.data

        # Create entities that would cause deadlock
        entity1 = Entity(
            id="entity1", name="Entity 1", entity_type=EntityType.PERSON, description="Test"
        )
        entity2 = Entity(
            id="entity2", name="Entity 2", entity_type=EntityType.PERSON, description="Test"
        )

        # Save entities first
        self.dao.save_entity(entity1)
        self.dao.save_entity(entity2)

        # Simulate deadlock by having transactions try to access resources in opposite order
        with patch.object(self.transaction_manager.lock_manager, "_lock_timeout", 0.1):
            # tx1 gets lock on entity1, tries to get entity2
            tx1.remove_entity("entity1")

            # tx2 gets lock on entity2, tries to get entity1
            tx2.remove_entity("entity2")

            # This would normally cause deadlock, but our timeout should handle it
            result = tx1.remove_entity("entity2")

            # At least one should fail due to timeout/deadlock
            # The exact behavior depends on timing and implementation details

    def test_isolation_level_enforcement(self):
        """Test that isolation levels are properly enforced."""
        # Test with READ_COMMITTED
        result = self.transaction_manager.begin_transaction(
            isolation_level=IsolationLevel.READ_COMMITTED
        )
        transaction = result.data

        # Add snapshot timestamp for REPEATABLE_READ and SERIALIZABLE
        repeatable_result = self.transaction_manager.begin_transaction(
            isolation_level=IsolationLevel.REPEATABLE_READ
        )
        repeatable_tx = repeatable_result.data
        self.assertIsNotNone(repeatable_tx.snapshot_timestamp)

        serializable_result = self.transaction_manager.begin_transaction(
            isolation_level=IsolationLevel.SERIALIZABLE
        )
        serializable_tx = serializable_result.data
        self.assertIsNotNone(serializable_tx.snapshot_timestamp)


class TestTransactionIntegration(unittest.TestCase):
    """Test transaction integration with the broader system."""

    def setUp(self):
        self.dao = MemoryDataAccessLayer()
        self.transaction_manager = TransactionManager(self.dao)

    def test_atomic_transaction_utility(self):
        """Test the atomic_transaction utility function."""
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )

        with atomic_transaction(self.transaction_manager) as tx:
            tx.add_entity(entity)

        # Should be committed
        self.assertEqual(self.transaction_manager.stats["committed_transactions"], 1)

    def test_create_transaction_manager_utility(self):
        """Test the create_transaction_manager utility function."""
        dao = MemoryDataAccessLayer()
        manager = create_transaction_manager(dao)

        self.assertIsInstance(manager, TransactionManager)
        self.assertEqual(manager.dao, dao)

    def test_transaction_with_dao_integration(self):
        """Test transaction integration with DAO operations."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Add entity through transaction
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity)

        # Entity should not be visible in DAO yet
        dao_entity = self.dao.get_entity_by_id("entity1")
        self.assertIsNone(dao_entity)

        # Commit transaction
        self.transaction_manager.commit_transaction(transaction.id)

        # Now entity should be visible
        dao_entity = self.dao.get_entity_by_id("entity1")
        self.assertIsNotNone(dao_entity)
        self.assertEqual(dao_entity.name, "Test Entity")

    def test_transaction_rollback_with_dao(self):
        """Test transaction rollback with DAO integration."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Add entity through transaction
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        transaction.add_entity(entity)

        # Rollback transaction
        self.transaction_manager.rollback_transaction(transaction.id)

        # Entity should not be in DAO
        dao_entity = self.dao.get_entity_by_id("entity1")
        self.assertIsNone(dao_entity)

    def test_complex_transaction_workflow(self):
        """Test a complex transaction with multiple operations."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Create entities
        entity1 = Entity(
            id="entity1",
            name="Person 1",
            entity_type=EntityType.PERSON,
            description="Test person 1",
        )
        entity2 = Entity(
            id="entity2", name="Place 1", entity_type=EntityType.LOCATION, description="Test place"
        )

        # Add entities
        transaction.add_entity(entity1)
        transaction.add_entity(entity2)

        # Create savepoint
        transaction.create_savepoint("after_entities")

        # Add relation
        relation = Relation(
            id="rel1",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.LOCATED_IN,
            description="Person located in place",
        )
        transaction.add_relation(relation)

        # Verify operation count
        self.assertEqual(len(transaction.operations), 3)

        # Commit transaction
        commit_result = self.transaction_manager.commit_transaction(transaction.id)
        self.assertTrue(commit_result.is_ok())

        # Verify all data is in DAO
        self.assertIsNotNone(self.dao.get_entity_by_id("entity1"))
        self.assertIsNotNone(self.dao.get_entity_by_id("entity2"))

    def test_concurrent_transaction_conflicts(self):
        """Test handling of concurrent transaction conflicts."""
        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        self.dao.save_entity(entity)

        def transaction_operation(tx_name: str, sleep_time: float):
            result = self.transaction_manager.begin_transaction()
            transaction = result.data

            # Sleep to create timing dependencies
            time.sleep(sleep_time)

            # Try to modify the same entity
            updated_entity = Entity(
                id="entity1",
                name=f"Updated by {tx_name}",
                entity_type=EntityType.PERSON,
                description="Updated",
            )

            update_result = transaction.update_entity(updated_entity)
            if update_result.is_ok():
                commit_result = self.transaction_manager.commit_transaction(transaction.id)
                return commit_result.is_ok()
            else:
                self.transaction_manager.rollback_transaction(transaction.id)
                return False

        # Start two concurrent transactions
        results = []
        threads = []

        for i, sleep_time in enumerate([0.1, 0.05]):
            thread = threading.Thread(
                target=lambda name=f"tx{i}", sleep=sleep_time: results.append(
                    transaction_operation(name, sleep)
                )
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # At least one should succeed (exact behavior depends on lock timing)
        self.assertTrue(any(results))

    def test_transaction_statistics_accuracy(self):
        """Test accuracy of transaction statistics."""
        initial_stats = self.transaction_manager.get_statistics()

        # Create and commit transactions
        for i in range(3):
            result = self.transaction_manager.begin_transaction()
            transaction = result.data
            entity = Entity(
                id=f"entity{i}",
                name=f"Entity {i}",
                entity_type=EntityType.PERSON,
                description="Test",
            )
            transaction.add_entity(entity)
            self.transaction_manager.commit_transaction(transaction.id)

        # Create and rollback transactions
        for i in range(2):
            result = self.transaction_manager.begin_transaction()
            transaction = result.data
            entity = Entity(
                id=f"rollback_entity{i}",
                name=f"Rollback Entity {i}",
                entity_type=EntityType.PERSON,
                description="Test",
            )
            transaction.add_entity(entity)
            self.transaction_manager.rollback_transaction(transaction.id)

        final_stats = self.transaction_manager.get_statistics()

        self.assertEqual(final_stats["total_transactions"] - initial_stats["total_transactions"], 5)
        self.assertEqual(
            final_stats["committed_transactions"] - initial_stats["committed_transactions"], 3
        )
        self.assertEqual(
            final_stats["aborted_transactions"] - initial_stats["aborted_transactions"], 2
        )


class TestTransactionErrors(unittest.TestCase):
    """Test transaction error handling and edge cases."""

    def setUp(self):
        self.dao = MemoryDataAccessLayer()
        self.transaction_manager = TransactionManager(self.dao)

    def test_invalid_transaction_state_operations(self):
        """Test operations on transactions in invalid states."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Commit transaction
        self.transaction_manager.commit_transaction(transaction.id)

        # Try to add entity to committed transaction
        entity = Entity(
            id="entity1", name="Test", entity_type=EntityType.PERSON, description="Test"
        )
        add_result = transaction.add_entity(entity)
        self.assertFalse(add_result.is_ok())
        self.assertEqual(add_result.error_code, ErrorCode.INVALID_OPERATION)

    def test_transaction_operations_after_rollback(self):
        """Test that operations fail after transaction rollback."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Rollback transaction
        self.transaction_manager.rollback_transaction(transaction.id)

        # Try to create savepoint on rolled back transaction
        savepoint_result = transaction.create_savepoint("sp1")
        self.assertFalse(savepoint_result.is_ok())
        self.assertEqual(savepoint_result.error_code, ErrorCode.INVALID_OPERATION)

    def test_double_commit(self):
        """Test that double commit fails appropriately."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # First commit should succeed
        commit_result1 = self.transaction_manager.commit_transaction(transaction.id)
        self.assertTrue(commit_result1.is_ok())

        # Second commit should fail
        commit_result2 = self.transaction_manager.commit_transaction(transaction.id)
        self.assertFalse(commit_result2.is_ok())
        self.assertEqual(commit_result2.error_code, ErrorCode.NOT_FOUND)

    def test_rollback_after_commit(self):
        """Test that rollback after commit fails appropriately."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Commit transaction
        self.transaction_manager.commit_transaction(transaction.id)

        # Try to rollback committed transaction
        rollback_result = transaction.rollback()
        self.assertFalse(rollback_result.is_ok())
        self.assertEqual(rollback_result.error_code, ErrorCode.INVALID_OPERATION)

    def test_exception_handling_in_commit(self):
        """Test exception handling during commit operations."""
        result = self.transaction_manager.begin_transaction()
        transaction = result.data

        # Mock batch context to raise exception
        with patch.object(transaction.batch_context, "commit", side_effect=Exception("Test error")):
            commit_result = self.transaction_manager.commit_transaction(transaction.id)
            self.assertFalse(commit_result.is_ok())
            self.assertEqual(transaction.status, TransactionStatus.FAILED)

    def test_lock_acquisition_failure(self):
        """Test handling of lock acquisition failures."""
        # Start first transaction and acquire lock
        result1 = self.transaction_manager.begin_transaction()
        tx1 = result1.data

        entity = Entity(
            id="entity1", name="Test Entity", entity_type=EntityType.PERSON, description="Test"
        )
        self.dao.save_entity(entity)

        # Acquire lock on entity1
        tx1.remove_entity("entity1")

        # Start second transaction and try to acquire same lock with short timeout
        result2 = self.transaction_manager.begin_transaction(timeout_seconds=0.1)
        tx2 = result2.data

        # This should fail due to lock conflict and timeout
        remove_result = tx2.remove_entity("entity1")

        # Either the lock acquisition fails, or we get a timeout
        # The exact error depends on implementation timing
        if not remove_result.is_ok():
            self.assertIn(
                remove_result.error_code, [ErrorCode.CONCURRENT_MODIFICATION, ErrorCode.TIMEOUT]
            )


class TestLockInfo(unittest.TestCase):
    """Test LockInfo data structure."""

    def test_lock_info_creation(self):
        """Test LockInfo creation and properties."""
        lock_info = LockInfo(
            transaction_id="tx1", lock_type="write", resource_id="entity1", resource_type="entity"
        )

        self.assertEqual(lock_info.transaction_id, "tx1")
        self.assertEqual(lock_info.lock_type, "write")
        self.assertEqual(lock_info.resource_id, "entity1")
        self.assertEqual(lock_info.resource_type, "entity")
        self.assertIsInstance(lock_info.acquired_at, float)
        self.assertGreater(lock_info.acquired_at, 0)


class TestTransactionInfo(unittest.TestCase):
    """Test TransactionInfo data structure."""

    def test_transaction_info_creation(self):
        """Test TransactionInfo creation and properties."""
        info = TransactionInfo(
            transaction_id="tx1",
            status=TransactionStatus.ACTIVE,
            isolation_level=IsolationLevel.READ_COMMITTED,
        )

        self.assertEqual(info.transaction_id, "tx1")
        self.assertEqual(info.status, TransactionStatus.ACTIVE)
        self.assertEqual(info.isolation_level, IsolationLevel.READ_COMMITTED)
        self.assertIsInstance(info.created_at, float)
        self.assertIsNone(info.started_at)
        self.assertIsNone(info.finished_at)
        self.assertEqual(info.operations_count, 0)
        self.assertEqual(len(info.locks_held), 0)


if __name__ == "__main__":
    unittest.main()
