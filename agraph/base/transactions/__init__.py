"""
Transaction system module for ACID operations.

This module provides comprehensive transaction support:
- Batch operations for efficient bulk processing
- ACID transaction system with isolation levels
- Deadlock detection and resolution
- Transactional integrity guarantees
"""

from .batch import BatchContext, BatchOperation, BatchOperationContext, TransactionAwareBatchContext
from .transaction import (
    IsolationLevel,
    Transaction,
    TransactionInfo,
    TransactionManager,
    TransactionStatus,
)

__all__ = [
    # Batch operations
    "BatchOperation",
    "BatchContext",
    "BatchOperationContext",
    "TransactionAwareBatchContext",
    # ACID transactions
    "TransactionManager",
    "TransactionStatus",
    "Transaction",
    "TransactionInfo",
    "IsolationLevel",
]
