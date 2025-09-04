# AGraph ACID Transaction System Guide

## Overview

The AGraph ACID Transaction System provides enterprise-grade transactional capabilities for knowledge graph operations. Built on top of the existing DAO layer and batch operations system, it ensures data integrity through full ACID (Atomicity, Consistency, Isolation, Durability) compliance.

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Isolation Levels](#isolation-levels)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)
10. [Examples](#examples)

## Key Features

### ACID Properties
- **Atomicity**: All operations in a transaction succeed or all fail
- **Consistency**: Data integrity constraints are maintained
- **Isolation**: Concurrent transactions don't interfere with each other
- **Durability**: Committed changes persist permanently

### Advanced Capabilities
- **Multiple Isolation Levels**: READ_COMMITTED, REPEATABLE_READ, SERIALIZABLE
- **Deadlock Detection**: Automatic detection and resolution of deadlocks
- **Savepoints**: Partial rollback within transactions
- **Concurrent Transactions**: Support for multiple simultaneous transactions
- **Lock Management**: Sophisticated read/write lock system
- **Transaction-Aware Batching**: Integration with batch operations

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                Transaction Manager                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │Transaction 1│  │Transaction 2│  │    Lock Manager     │   │
│  │             │  │             │  │                     │   │
│  │ - Operations│  │ - Operations│  │ - Read/Write Locks  │   │
│  │ - Savepoints│  │ - Savepoints│  │ - Deadlock Detection│   │
│  │ - Status    │  │ - Status    │  │ - Resource Tracking │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   DAO Layer                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │   Entities  │  │  Relations  │  │    Batch Context    │   │
│  │             │  │             │  │                     │   │
│  │ - CRUD Ops  │  │ - CRUD Ops  │  │ - Atomic Execution  │   │
│  │ - Validation│  │ - Validation│  │ - Rollback Support  │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Basic Usage

### Creating a Transaction Manager

```python
from agraph.base.infrastructure.dao import MemoryDataAccessLayer
from agraph.base.transactions.transaction import TransactionManager, IsolationLevel

# Create DAO and transaction manager
dao = MemoryDataAccessLayer()
transaction_manager = TransactionManager(dao)
```

### Simple Transaction with Context Manager

```python
from agraph.base.models.entities import Entity
from agraph.types import EntityType

# Using context manager (recommended)
with transaction_manager.transaction() as tx:
    # Add an entity
    entity = Entity(
        id="person_001",
        name="John Doe",
        entity_type=EntityType.PERSON,
        confidence=0.95
    )

    result = tx.add_entity(entity)
    if not result.is_ok():
        raise Exception(f"Failed to add entity: {result.error_message}")

    # Add a relation
    relation = Relation(
        id="works_for_001",
        head_entity=entity,
        tail_entity=company_entity,
        relation_type=RelationType.WORKS_FOR,
        confidence=0.90
    )

    tx.add_relation(relation)

# Transaction is automatically committed here
# If any exception occurs, it's automatically rolled back
```

### Manual Transaction Control

```python
# Begin transaction manually
tx_result = transaction_manager.begin_transaction(
    isolation_level=IsolationLevel.READ_COMMITTED,
    timeout_seconds=300.0
)

if not tx_result.is_ok():
    print(f"Failed to begin transaction: {tx_result.error_message}")
    return

transaction = tx_result.data

try:
    # Perform operations
    entity = Entity(
        id="person_002",
        name="Jane Smith",
        entity_type=EntityType.PERSON
    )

    add_result = transaction.add_entity(entity)
    if not add_result.is_ok():
        print(f"Failed to add entity: {add_result.error_message}")
        # Rollback and exit
        transaction_manager.rollback_transaction(transaction.id)
        return

    # Commit transaction
    commit_result = transaction_manager.commit_transaction(transaction.id)
    if not commit_result.is_ok():
        print(f"Failed to commit: {commit_result.error_message}")
    else:
        print("Transaction committed successfully")
        print(f"Operations executed: {commit_result.data['total_operations']}")

except Exception as e:
    # Rollback on error
    rollback_result = transaction_manager.rollback_transaction(transaction.id)
    print(f"Transaction rolled back due to error: {e}")
```

## Advanced Features

### Savepoints and Partial Rollback

```python
with transaction_manager.transaction() as tx:
    # Add some entities
    entity1 = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
    entity2 = Entity(id="e2", name="Entity 2", entity_type=EntityType.PERSON)

    tx.add_entity(entity1)
    tx.add_entity(entity2)

    # Create a savepoint
    savepoint_result = tx.create_savepoint("after_entities")
    if not savepoint_result.is_ok():
        raise Exception("Failed to create savepoint")

    # Add a relation that might fail
    try:
        risky_relation = Relation(
            id="risky_rel",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.CUSTOM,  # Might not be valid
            confidence=0.5
        )

        result = tx.add_relation(risky_relation)
        if not result.is_ok():
            # Rollback to savepoint instead of entire transaction
            rollback_result = tx.rollback_to_savepoint("after_entities")
            if rollback_result.is_ok():
                print("Rolled back to savepoint, entities preserved")

    except Exception as e:
        # Rollback to savepoint
        tx.rollback_to_savepoint("after_entities")
        print(f"Rolled back to savepoint due to: {e}")

# Transaction commits with entities but without the risky relation
```

### Transaction-Aware Batch Operations

```python
from agraph.base.transactions.batch import atomic_batch_operations, create_entity_batch_operation
from agraph.base.transactions.batch import BatchOperationType

# Using atomic batch operations
with atomic_batch_operations(
    dao,
    transaction_manager,
    isolation_level=IsolationLevel.REPEATABLE_READ,
    timeout_seconds=600.0
) as batch_ctx:

    # Create multiple entities in a batch
    entities_data = [
        {
            "id": f"batch_entity_{i}",
            "name": f"Batch Entity {i}",
            "entity_type": EntityType.PERSON.value,
            "confidence": 0.8 + (i * 0.02)
        }
        for i in range(10)
    ]

    # Add all entities to the batch
    for entity_data in entities_data:
        operation = create_entity_batch_operation(
            BatchOperationType.ADD, entity_data
        )

        result = batch_ctx.add_operation_with_transaction(operation)
        if not result.is_ok():
            print(f"Failed to add operation: {result.error_message}")
            break

    # Create a savepoint before risky operations
    batch_ctx.create_savepoint("before_relations")

    # Add some relations that might fail
    for i in range(5):
        relation_data = {
            "id": f"batch_relation_{i}",
            "head_entity_id": f"batch_entity_{i}",
            "tail_entity_id": f"batch_entity_{i+1}",
            "relation_type": RelationType.KNOWS.value,
            "confidence": 0.7
        }

        try:
            relation_op = create_relation_batch_operation(
                BatchOperationType.ADD, relation_data
            )
            batch_ctx.add_operation_with_transaction(relation_op)

        except Exception as e:
            print(f"Relation {i} failed, rolling back to savepoint")
            batch_ctx.rollback_to_savepoint("before_relations")
            break

# All operations are committed atomically
```

### Concurrent Transaction Monitoring

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def worker_transaction(worker_id: int, transaction_manager: TransactionManager):
    """Worker function that performs transactions."""
    try:
        with transaction_manager.transaction(
            isolation_level=IsolationLevel.READ_COMMITTED,
            timeout_seconds=60.0
        ) as tx:

            # Simulate some work
            for i in range(5):
                entity = Entity(
                    id=f"worker_{worker_id}_entity_{i}",
                    name=f"Worker {worker_id} Entity {i}",
                    entity_type=EntityType.PERSON,
                    confidence=0.8
                )

                result = tx.add_entity(entity)
                if not result.is_ok():
                    print(f"Worker {worker_id}: Failed to add entity {i}")
                    return False

                # Small delay to increase chance of concurrent access
                time.sleep(0.01)

            print(f"Worker {worker_id}: Successfully completed transaction")
            return True

    except Exception as e:
        print(f"Worker {worker_id}: Transaction failed with error: {e}")
        return False

# Monitor transactions while they run
def monitor_transactions(transaction_manager: TransactionManager, duration: int):
    """Monitor transaction statistics."""
    start_time = time.time()

    while time.time() - start_time < duration:
        stats = transaction_manager.get_statistics()
        active_txs = transaction_manager.get_active_transactions()

        print(f"Stats: {stats['total_transactions']} total, "
              f"{stats['committed_transactions']} committed, "
              f"{stats['aborted_transactions']} aborted, "
              f"{len(active_txs)} active")

        if active_txs:
            for tx_info in active_txs:
                print(f"  - TX {tx_info.transaction_id}: {tx_info.status.value}, "
                      f"{tx_info.operations_count} ops, "
                      f"{len(tx_info.locks_held)} locks")

        time.sleep(1.0)

# Run concurrent transactions
with ThreadPoolExecutor(max_workers=5) as executor:
    # Start monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_transactions,
        args=(transaction_manager, 10)
    )
    monitor_thread.start()

    # Submit worker transactions
    futures = [
        executor.submit(worker_transaction, i, transaction_manager)
        for i in range(10)
    ]

    # Wait for completion
    results = [future.result() for future in futures]
    success_count = sum(results)

    print(f"Completed: {success_count}/{len(futures)} transactions succeeded")

    # Wait for monitoring to complete
    monitor_thread.join()
```

## Isolation Levels

### READ_COMMITTED (Default)
Prevents dirty reads but allows non-repeatable reads and phantom reads.

```python
with transaction_manager.transaction(
    isolation_level=IsolationLevel.READ_COMMITTED
) as tx:
    # This transaction will only see committed data
    # But data might change between reads within the transaction
    pass
```

### REPEATABLE_READ
Prevents dirty reads and non-repeatable reads, but allows phantom reads.

```python
with transaction_manager.transaction(
    isolation_level=IsolationLevel.REPEATABLE_READ
) as tx:
    # Creates a snapshot at transaction start
    # Repeated reads of the same data will return the same results
    # but new data might appear (phantoms)
    pass
```

### SERIALIZABLE
Highest isolation level - prevents all phenomena.

```python
with transaction_manager.transaction(
    isolation_level=IsolationLevel.SERIALIZABLE
) as tx:
    # Full isolation - transaction appears to run in complete isolation
    # Most restrictive but safest for critical operations
    pass
```

## Error Handling

### Comprehensive Error Handling

```python
from agraph.base.core.result import ErrorCode
from agraph.base.transactions.transaction import TransactionException, DeadlockException

def robust_transaction_example():
    """Example of robust error handling in transactions."""

    try:
        with transaction_manager.transaction(
            isolation_level=IsolationLevel.READ_COMMITTED,
            timeout_seconds=30.0
        ) as tx:

            # Try to add entities
            entities = [
                Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON),
                Entity(id="e2", name="Entity 2", entity_type=EntityType.ORGANIZATION),
            ]

            for entity in entities:
                result = tx.add_entity(entity)
                if not result.is_ok():
                    if result.error_code == ErrorCode.DUPLICATE_ENTRY:
                        print(f"Entity {entity.id} already exists, skipping")
                        continue
                    elif result.error_code == ErrorCode.VALIDATION_ERROR:
                        print(f"Entity {entity.id} validation failed: {result.error_message}")
                        # You could fix the entity and retry, or skip it
                        continue
                    elif result.error_code == ErrorCode.TIMEOUT:
                        print("Transaction timeout - operation took too long")
                        raise TransactionException("Timeout during entity addition")
                    else:
                        print(f"Unexpected error adding entity: {result.error_message}")
                        raise TransactionException(f"Failed to add entity: {result.error_message}")

            print("All entities processed successfully")

    except DeadlockException as e:
        print(f"Deadlock detected: {e}")
        print("Transaction was automatically rolled back")
        # Could implement retry logic here

    except TransactionException as e:
        print(f"Transaction failed: {e}")
        # Transaction is automatically rolled back

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Transaction is automatically rolled back

# Example with retry logic for deadlocks
def transaction_with_retry(max_retries: int = 3):
    """Transaction with automatic retry on deadlock."""

    for attempt in range(max_retries):
        try:
            with transaction_manager.transaction() as tx:
                # Your transaction logic here
                entity = Entity(
                    id="retry_entity",
                    name="Retry Test Entity",
                    entity_type=EntityType.PERSON
                )

                result = tx.add_entity(entity)
                if not result.is_ok():
                    raise TransactionException(f"Failed to add entity: {result.error_message}")

                print(f"Transaction succeeded on attempt {attempt + 1}")
                return True

        except DeadlockException:
            if attempt < max_retries - 1:
                print(f"Deadlock on attempt {attempt + 1}, retrying...")
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            else:
                print("Max retries exceeded due to deadlocks")
                return False

        except TransactionException as e:
            print(f"Transaction failed permanently: {e}")
            return False

    return False
```

## Performance Considerations

### Lock Contention Minimization

```python
# Bad: Long-running transaction holding locks
with transaction_manager.transaction() as tx:
    # Get entities
    entities = []
    for i in range(100):
        entity = Entity(id=f"e_{i}", name=f"Entity {i}", entity_type=EntityType.PERSON)
        tx.add_entity(entity)

        # Bad: Expensive operation while holding locks
        time.sleep(0.1)  # Simulating expensive computation
        entities.append(entity)

# Good: Minimize time in transaction
# Prepare data outside transaction
entities = []
for i in range(100):
    entity = Entity(id=f"e_{i}", name=f"Entity {i}", entity_type=EntityType.PERSON)
    # Do expensive computation here, outside transaction
    entity.description = generate_complex_description(entity)  # Expensive operation
    entities.append(entity)

# Quick transaction to just store data
with transaction_manager.transaction() as tx:
    for entity in entities:
        tx.add_entity(entity)
```

### Batch Operations for Performance

```python
# Instead of individual transactions
for entity_data in large_entity_dataset:
    with transaction_manager.transaction() as tx:
        entity = Entity.from_dict(entity_data)
        tx.add_entity(entity)

# Use batch operations for better performance
with atomic_batch_operations(
    dao,
    transaction_manager,
    isolation_level=IsolationLevel.READ_COMMITTED
) as batch_ctx:

    for entity_data in large_entity_dataset:
        operation = create_entity_batch_operation(
            BatchOperationType.ADD, entity_data
        )
        batch_ctx.add_operation_with_transaction(operation)

        # Create savepoints periodically for partial recovery
        if len(batch_ctx.operations) % 100 == 0:
            batch_ctx.create_savepoint(f"checkpoint_{len(batch_ctx.operations)}")
```

### Transaction Pool Management

```python
class TransactionPool:
    """Manages a pool of transactions for high-throughput applications."""

    def __init__(self, transaction_manager: TransactionManager, pool_size: int = 10):
        self.transaction_manager = transaction_manager
        self.pool_size = pool_size
        self.active_transactions: Dict[str, Transaction] = {}
        self._lock = threading.Lock()

    def get_transaction(self,
                       isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
                       timeout_seconds: float = 300.0) -> Transaction:
        """Get a transaction from the pool or create a new one."""
        with self._lock:
            if len(self.active_transactions) < self.pool_size:
                tx_result = self.transaction_manager.begin_transaction(
                    isolation_level=isolation_level,
                    timeout_seconds=timeout_seconds
                )

                if tx_result.is_ok():
                    tx = tx_result.data
                    self.active_transactions[tx.id] = tx
                    return tx
                else:
                    raise TransactionException(f"Failed to create transaction: {tx_result.error_message}")
            else:
                raise TransactionException("Transaction pool exhausted")

    def release_transaction(self, transaction_id: str, commit: bool = True):
        """Release a transaction back to the pool."""
        with self._lock:
            if transaction_id in self.active_transactions:
                if commit:
                    self.transaction_manager.commit_transaction(transaction_id)
                else:
                    self.transaction_manager.rollback_transaction(transaction_id)

                del self.active_transactions[transaction_id]

# Usage
tx_pool = TransactionPool(transaction_manager, pool_size=5)

def high_throughput_operation(data_batch):
    """High-throughput operation using transaction pool."""
    tx = tx_pool.get_transaction()

    try:
        for item in data_batch:
            entity = Entity.from_dict(item)
            result = tx.add_entity(entity)

            if not result.is_ok():
                tx_pool.release_transaction(tx.id, commit=False)
                raise Exception(f"Failed to add entity: {result.error_message}")

        tx_pool.release_transaction(tx.id, commit=True)

    except Exception as e:
        tx_pool.release_transaction(tx.id, commit=False)
        raise e
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with transaction_manager.transaction() as tx:
    # Your operations here
    pass

# Avoid manual transaction management unless necessary
```

### 2. Keep Transactions Short

```python
# Good: Short transaction
with transaction_manager.transaction() as tx:
    tx.add_entity(prepared_entity)
    tx.add_relation(prepared_relation)

# Bad: Long transaction with I/O
with transaction_manager.transaction() as tx:
    entity_data = fetch_from_external_api()  # Bad: I/O in transaction
    entity = Entity.from_dict(entity_data)
    tx.add_entity(entity)
```

### 3. Use Appropriate Isolation Levels

```python
# For read-heavy operations where consistency is critical
with transaction_manager.transaction(isolation_level=IsolationLevel.REPEATABLE_READ) as tx:
    # Generate report based on consistent snapshot
    pass

# For high-throughput operations where some inconsistency is acceptable
with transaction_manager.transaction(isolation_level=IsolationLevel.READ_COMMITTED) as tx:
    # Bulk data ingestion
    pass
```

### 4. Handle Errors Gracefully

```python
def safe_transaction_operation():
    """Example of safe transaction handling."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            with transaction_manager.transaction() as tx:
                # Your operations
                return True

        except DeadlockException:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        except TransactionException as e:
            # Log error and don't retry for permanent failures
            logger.error(f"Transaction failed permanently: {e}")
            return False

    return False
```

### 5. Monitor Transaction Performance

```python
def monitored_transaction_operation():
    """Transaction with performance monitoring."""
    start_time = time.time()

    try:
        with transaction_manager.transaction() as tx:
            # Your operations
            operation_result = perform_complex_operations(tx)

            # Log success metrics
            duration = time.time() - start_time
            logger.info(f"Transaction completed in {duration:.3f}s")

            return operation_result

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Transaction failed after {duration:.3f}s: {e}")

        # Report failure metrics
        metrics_client.increment('transaction.failures')
        metrics_client.timing('transaction.duration', duration)

        raise
```

## API Reference

### TransactionManager

#### `begin_transaction(isolation_level, timeout_seconds) -> Result[Transaction]`
Begin a new transaction.

#### `commit_transaction(transaction_id) -> Result[Dict[str, Any]]`
Commit a transaction by ID.

#### `rollback_transaction(transaction_id) -> Result[bool]`
Rollback a transaction by ID.

#### `transaction(isolation_level, timeout_seconds) -> ContextManager[Transaction]`
Context manager for automatic transaction handling.

#### `get_active_transactions() -> List[TransactionInfo]`
Get information about all active transactions.

#### `get_statistics() -> Dict[str, Any]`
Get transaction manager statistics.

### Transaction

#### `add_entity(entity) -> Result[Entity]`
Add an entity within the transaction.

#### `remove_entity(entity_id) -> Result[bool]`
Remove an entity within the transaction.

#### `update_entity(entity) -> Result[Entity]`
Update an entity within the transaction.

#### `add_relation(relation) -> Result[Relation]`
Add a relation within the transaction.

#### `remove_relation(relation_id) -> Result[bool]`
Remove a relation within the transaction.

#### `create_savepoint(savepoint_name) -> Result[str]`
Create a savepoint within the transaction.

#### `rollback_to_savepoint(savepoint_name) -> Result[bool]`
Rollback to a specific savepoint.

#### `get_info() -> TransactionInfo`
Get information about the transaction.

### TransactionAwareBatchContext

#### `begin_transaction(isolation_level, timeout_seconds) -> Result[Transaction]`
Begin a transaction for this batch context.

#### `add_operation_with_transaction(operation) -> Result[BatchOperation]`
Add an operation that will be executed within the current transaction.

#### `commit_with_transaction() -> Result[Dict[str, Any]]`
Commit both the batch operations and the transaction.

#### `rollback_with_transaction() -> Result[bool]`
Rollback both the batch operations and the transaction.

#### `transactional_batch(isolation_level, timeout_seconds) -> ContextManager`
Context manager for transactional batch operations.

## Examples

### Complete Example: Knowledge Graph Construction with Transactions

```python
#!/usr/bin/env python3
"""
Complete example demonstrating transaction system usage for knowledge graph construction.
"""

from agraph.base.infrastructure.dao import MemoryDataAccessLayer
from agraph.base.transactions.transaction import TransactionManager, IsolationLevel, atomic_transaction
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.types import EntityType, RelationType
from agraph.base.transactions.batch import atomic_batch_operations
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example function."""

    # Initialize system
    dao = MemoryDataAccessLayer()
    transaction_manager = TransactionManager(dao)

    # Example 1: Simple transaction
    logger.info("Example 1: Simple transaction")
    simple_transaction_example(transaction_manager)

    # Example 2: Error handling and recovery
    logger.info("Example 2: Error handling and recovery")
    error_handling_example(transaction_manager)

    # Example 3: Batch operations with savepoints
    logger.info("Example 3: Batch operations with savepoints")
    batch_operations_example(dao, transaction_manager)

    # Example 4: Complex knowledge graph construction
    logger.info("Example 4: Complex knowledge graph construction")
    complex_kg_construction_example(dao, transaction_manager)

    # Example 5: Performance monitoring
    logger.info("Example 5: Performance monitoring")
    performance_monitoring_example(transaction_manager)

def simple_transaction_example(transaction_manager: TransactionManager):
    """Simple transaction example."""

    with transaction_manager.transaction() as tx:
        # Create a person
        person = Entity(
            id="john_doe",
            name="John Doe",
            entity_type=EntityType.PERSON,
            confidence=0.95,
            properties={"age": 30, "occupation": "Software Engineer"}
        )

        result = tx.add_entity(person)
        if not result.is_ok():
            raise Exception(f"Failed to add person: {result.error_message}")

        # Create a company
        company = Entity(
            id="tech_corp",
            name="Tech Corp Inc.",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.90,
            properties={"industry": "Technology", "founded": 2010}
        )

        tx.add_entity(company)

        # Create employment relation
        employment = Relation(
            id="john_works_at_tech_corp",
            head_entity=person,
            tail_entity=company,
            relation_type=RelationType.WORKS_FOR,
            confidence=0.85,
            properties={"start_date": "2020-01-15", "position": "Senior Developer"}
        )

        tx.add_relation(employment)

        logger.info("Successfully created person, company, and employment relation")

def error_handling_example(transaction_manager: TransactionManager):
    """Example demonstrating error handling and recovery."""

    try:
        with transaction_manager.transaction() as tx:
            # Add a valid entity
            valid_entity = Entity(
                id="valid_entity",
                name="Valid Entity",
                entity_type=EntityType.PERSON,
                confidence=0.8
            )

            result = tx.add_entity(valid_entity)
            if not result.is_ok():
                raise Exception(f"Failed to add valid entity: {result.error_message}")

            logger.info("Added valid entity successfully")

            # Create a savepoint before risky operation
            savepoint_result = tx.create_savepoint("before_risky_operation")
            if not savepoint_result.is_ok():
                raise Exception("Failed to create savepoint")

            # Try to add an invalid entity (duplicate ID)
            try:
                duplicate_entity = Entity(
                    id="valid_entity",  # Same ID - will cause error
                    name="Duplicate Entity",
                    entity_type=EntityType.PERSON,
                    confidence=0.7
                )

                result = tx.add_entity(duplicate_entity)
                if not result.is_ok():
                    logger.warning(f"Expected error occurred: {result.error_message}")

                    # Rollback to savepoint instead of entire transaction
                    rollback_result = tx.rollback_to_savepoint("before_risky_operation")
                    if rollback_result.is_ok():
                        logger.info("Successfully rolled back to savepoint")
                    else:
                        logger.error("Failed to rollback to savepoint")
                        raise Exception("Savepoint rollback failed")

            except Exception as e:
                logger.error(f"Unexpected error during risky operation: {e}")
                tx.rollback_to_savepoint("before_risky_operation")

            # Add another valid entity after recovery
            recovery_entity = Entity(
                id="recovery_entity",
                name="Recovery Entity",
                entity_type=EntityType.ORGANIZATION,
                confidence=0.9
            )

            tx.add_entity(recovery_entity)
            logger.info("Successfully added recovery entity")

    except Exception as e:
        logger.error(f"Transaction failed: {e}")

def batch_operations_example(dao: MemoryDataAccessLayer, transaction_manager: TransactionManager):
    """Example using batch operations with transactions."""

    # Prepare batch data
    entities_data = [
        {
            "id": f"batch_person_{i}",
            "name": f"Person {i}",
            "entity_type": EntityType.PERSON.value,
            "confidence": 0.8 + (i * 0.01),
            "properties": {"batch_id": "batch_001"}
        }
        for i in range(10)
    ]

    with atomic_batch_operations(
        dao,
        transaction_manager,
        isolation_level=IsolationLevel.READ_COMMITTED,
        timeout_seconds=60.0
    ) as batch_ctx:

        # Add entities in batch
        from agraph.base.transactions.batch import create_entity_batch_operation, BatchOperationType

        for entity_data in entities_data:
            operation = create_entity_batch_operation(
                BatchOperationType.ADD, entity_data
            )

            result = batch_ctx.add_operation_with_transaction(operation)
            if not result.is_ok():
                logger.error(f"Failed to add batch operation: {result.error_message}")
                break

        logger.info(f"Added {len(entities_data)} entities to batch")

        # Create savepoint before adding relations
        savepoint_result = batch_ctx.create_savepoint("before_relations")
        if savepoint_result.is_ok():
            logger.info("Created savepoint before relations")

        # Add relations between consecutive entities
        relations_added = 0
        try:
            from agraph.base.transactions.batch import create_relation_batch_operation

            for i in range(len(entities_data) - 1):
                relation_data = {
                    "id": f"knows_relation_{i}",
                    "head_entity_id": f"batch_person_{i}",
                    "tail_entity_id": f"batch_person_{i+1}",
                    "relation_type": RelationType.KNOWS.value,
                    "confidence": 0.7,
                    "properties": {"relationship": "colleague"}
                }

                relation_op = create_relation_batch_operation(
                    BatchOperationType.ADD, relation_data
                )

                result = batch_ctx.add_operation_with_transaction(relation_op)
                if result.is_ok():
                    relations_added += 1
                else:
                    logger.warning(f"Failed to add relation {i}: {result.error_message}")

        except Exception as e:
            logger.error(f"Error adding relations: {e}")
            # Could rollback to savepoint here if needed

        logger.info(f"Successfully added {relations_added} relations")

        # Get final metrics
        metrics = batch_ctx.get_metrics()
        logger.info(f"Batch metrics: {json.dumps(metrics, indent=2)}")

def complex_kg_construction_example(dao: MemoryDataAccessLayer, transaction_manager: TransactionManager):
    """Complex example showing knowledge graph construction with multiple entity types."""

    with transaction_manager.transaction(
        isolation_level=IsolationLevel.REPEATABLE_READ,
        timeout_seconds=120.0
    ) as tx:

        # Create people
        people = [
            Entity(
                id="alice",
                name="Alice Johnson",
                entity_type=EntityType.PERSON,
                confidence=0.95,
                properties={"role": "CEO", "experience": "15 years"}
            ),
            Entity(
                id="bob",
                name="Bob Smith",
                entity_type=EntityType.PERSON,
                confidence=0.92,
                properties={"role": "CTO", "experience": "12 years"}
            ),
            Entity(
                id="carol",
                name="Carol Davis",
                entity_type=EntityType.PERSON,
                confidence=0.88,
                properties={"role": "Developer", "experience": "5 years"}
            )
        ]

        # Create organizations
        organizations = [
            Entity(
                id="startup_inc",
                name="Startup Inc.",
                entity_type=EntityType.ORGANIZATION,
                confidence=0.90,
                properties={"industry": "AI/ML", "size": "50-100", "founded": 2020}
            ),
            Entity(
                id="big_corp",
                name="Big Corp",
                entity_type=EntityType.ORGANIZATION,
                confidence=0.95,
                properties={"industry": "Enterprise Software", "size": "1000+", "founded": 1995}
            )
        ]

        # Add all entities
        all_entities = people + organizations
        for entity in all_entities:
            result = tx.add_entity(entity)
            if not result.is_ok():
                raise Exception(f"Failed to add entity {entity.id}: {result.error_message}")

        logger.info(f"Added {len(all_entities)} entities")

        # Create employment relations
        employment_relations = [
            Relation(
                id="alice_ceo_startup",
                head_entity=people[0],  # Alice
                tail_entity=organizations[0],  # Startup Inc
                relation_type=RelationType.WORKS_FOR,
                confidence=0.95,
                properties={"position": "CEO", "start_date": "2020-01-01"}
            ),
            Relation(
                id="bob_cto_startup",
                head_entity=people[1],  # Bob
                tail_entity=organizations[0],  # Startup Inc
                relation_type=RelationType.WORKS_FOR,
                confidence=0.90,
                properties={"position": "CTO", "start_date": "2020-03-15"}
            ),
            Relation(
                id="carol_dev_startup",
                head_entity=people[2],  # Carol
                tail_entity=organizations[0],  # Startup Inc
                relation_type=RelationType.WORKS_FOR,
                confidence=0.85,
                properties={"position": "Senior Developer", "start_date": "2021-06-01"}
            )
        ]

        # Create interpersonal relations
        interpersonal_relations = [
            Relation(
                id="alice_manages_bob",
                head_entity=people[0],  # Alice
                tail_entity=people[1],  # Bob
                relation_type=RelationType.MANAGES,
                confidence=0.90,
                properties={"relationship_type": "direct_report"}
            ),
            Relation(
                id="bob_manages_carol",
                head_entity=people[1],  # Bob
                tail_entity=people[2],  # Carol
                relation_type=RelationType.MANAGES,
                confidence=0.88,
                properties={"relationship_type": "direct_report"}
            ),
            Relation(
                id="alice_knows_carol",
                head_entity=people[0],  # Alice
                tail_entity=people[2],  # Carol
                relation_type=RelationType.KNOWS,
                confidence=0.75,
                properties={"relationship_type": "colleague"}
            )
        ]

        # Add all relations
        all_relations = employment_relations + interpersonal_relations
        for relation in all_relations:
            result = tx.add_relation(relation)
            if not result.is_ok():
                raise Exception(f"Failed to add relation {relation.id}: {result.error_message}")

        logger.info(f"Added {len(all_relations)} relations")

        # Get transaction info before commit
        tx_info = tx.get_info()
        logger.info(f"Transaction info: {tx_info.operations_count} operations, "
                   f"status: {tx_info.status.value}, "
                   f"isolation: {tx_info.isolation_level.value}")

    # Verify the knowledge graph was created correctly
    verify_knowledge_graph(dao)

def verify_knowledge_graph(dao: MemoryDataAccessLayer):
    """Verify that the knowledge graph was constructed correctly."""

    # Check entities
    entities = dao.get_entities()
    logger.info(f"Total entities in graph: {len(entities)}")

    # Check relations
    relations = dao.get_relations()
    logger.info(f"Total relations in graph: {len(relations)}")

    # Check specific entities
    alice = dao.get_entity_by_id("alice")
    if alice:
        logger.info(f"Alice entity: {alice.name} ({alice.entity_type})")

        # Get Alice's relations
        alice_relations = dao.get_entity_relations("alice")
        logger.info(f"Alice has {len(alice_relations)} relations:")

        for relation in alice_relations:
            other_entity_id = (
                relation.tail_entity.id if relation.head_entity.id == "alice"
                else relation.head_entity.id
            )
            other_entity = dao.get_entity_by_id(other_entity_id)

            if other_entity:
                logger.info(f"  - {relation.relation_type.value} {other_entity.name}")

def performance_monitoring_example(transaction_manager: TransactionManager):
    """Example demonstrating performance monitoring."""

    # Get initial statistics
    initial_stats = transaction_manager.get_statistics()
    logger.info(f"Initial stats: {json.dumps(initial_stats, indent=2)}")

    # Perform multiple transactions
    num_transactions = 5

    for i in range(num_transactions):
        start_time = time.time()

        try:
            with transaction_manager.transaction() as tx:
                # Create test entities
                for j in range(3):
                    entity = Entity(
                        id=f"perf_test_{i}_{j}",
                        name=f"Performance Test Entity {i}-{j}",
                        entity_type=EntityType.PERSON,
                        confidence=0.8
                    )

                    result = tx.add_entity(entity)
                    if not result.is_ok():
                        logger.error(f"Failed to add entity: {result.error_message}")
                        break

                duration = time.time() - start_time
                logger.info(f"Transaction {i+1} completed in {duration:.3f}s")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Transaction {i+1} failed after {duration:.3f}s: {e}")

    # Get final statistics
    final_stats = transaction_manager.get_statistics()
    logger.info(f"Final stats: {json.dumps(final_stats, indent=2)}")

    # Show active transactions (should be empty)
    active_txs = transaction_manager.get_active_transactions()
    logger.info(f"Active transactions: {len(active_txs)}")

if __name__ == "__main__":
    main()
```

This comprehensive example demonstrates:

1. **Basic transaction usage** with context managers
2. **Error handling and recovery** with savepoints
3. **Batch operations** with transaction support
4. **Complex knowledge graph construction** with multiple entity types and relations
5. **Performance monitoring** and statistics collection

The transaction system provides enterprise-grade ACID compliance while maintaining ease of use through context managers and comprehensive error handling.
