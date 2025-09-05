"""
Unified manager implementations following the new architecture.

This module provides concrete implementations of the manager interfaces
that use the DAO layer for data access and return Result types for
consistent error handling.
"""

import time
from threading import RLock

# Forward declarations
from typing import Any, Dict, List, Optional, Union

from ..core.result import ErrorCode, ErrorDetail, Result
from ..core.types import ClusterType, EntityType, RelationType
from ..infrastructure.dao import DataAccessLayer
from ..models.clusters import Cluster
from ..models.entities import Entity
from ..models.relations import Relation
from ..models.text import TextChunk
from .interfaces import ClusterManager, EntityManager, RelationManager, TextChunkManager


class UnifiedEntityManager(EntityManager):
    """
    Unified entity manager implementation using DAO layer.

    This manager follows the new architecture pattern with:
    - Result-based error handling
    - DAO layer for data access
    - Consistent validation and business logic
    - Performance monitoring
    """

    def __init__(self, dao: DataAccessLayer, enable_validation: bool = True):
        self.dao = dao
        self.enable_validation = enable_validation
        self._lock = RLock()
        self._metrics = {
            "operations_count": 0,
            "errors_count": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _record_operation(self, operation_name: str, start_time: float, success: bool = True) -> None:
        """Record operation metrics."""
        execution_time = time.time() - start_time
        with self._lock:
            self._metrics["operations_count"] += 1
            if not success:
                self._metrics["errors_count"] += 1

            # Update average response time
            total_operations = self._metrics["operations_count"]
            current_avg = self._metrics["average_response_time"]
            self._metrics["average_response_time"] = (
                current_avg * (total_operations - 1) + execution_time
            ) / total_operations

    def add(self, item: "Entity") -> Result["Entity"]:
        """Add an entity to the knowledge graph."""
        start_time = time.time()

        try:
            # Validate the entity
            if self.enable_validation:
                validation_result = self.validate(item)
                if not validation_result.is_ok():
                    self._record_operation("add", start_time, False)
                    return Result.fail(
                        ErrorCode.VALIDATION_ERROR,
                        validation_result.error_message or "Validation failed",
                    )

            # Check for duplicates
            existing = self.dao.get_entity_by_id(item.id)
            if existing:
                self._record_operation("add", start_time, False)
                return Result.fail(ErrorCode.DUPLICATE_ENTRY, f"Entity with ID '{item.id}' already exists")

            # Save the entity
            self.dao.save_entity(item)
            self._record_operation("add", start_time)

            return Result.ok(
                item,
                metadata={
                    "operation": "add_entity",
                    "entity_type": str(item.entity_type),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("add", start_time, False)
            return Result.internal_error(e)

    def remove(self, item_id: str) -> Result[bool]:
        """Remove an entity from the knowledge graph."""
        start_time = time.time()

        try:
            # Check if entity exists
            existing = self.dao.get_entity_by_id(item_id)
            if not existing:
                self._record_operation("remove", start_time, False)
                return Result.not_found("Entity", item_id)

            # Remove the entity
            success = self.dao.delete_entity(item_id)
            self._record_operation("remove", start_time, success)

            return Result.ok(
                success,
                metadata={
                    "operation": "remove_entity",
                    "entity_id": item_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("remove", start_time, False)
            return Result.internal_error(e)

    def get(self, item_id: str) -> Result[Optional["Entity"]]:
        """Get an entity by ID."""
        start_time = time.time()

        try:
            entity = self.dao.get_entity_by_id(item_id)
            self._record_operation("get", start_time)

            return Result.ok(
                entity,
                metadata={
                    "operation": "get_entity",
                    "found": entity is not None,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get", start_time, False)
            return Result.internal_error(e)

    def list_all(self) -> Result[List["Entity"]]:
        """List all entities."""
        start_time = time.time()

        try:
            entities_dict = self.dao.get_entities()
            entities_list = list(entities_dict.values())
            self._record_operation("list_all", start_time)

            return Result.ok(
                entities_list,
                metadata={
                    "operation": "list_all_entities",
                    "count": len(entities_list),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_all", start_time, False)
            return Result.internal_error(e)

    def list_by_criteria(self, criteria: Dict[str, Any]) -> Result[List["Entity"]]:
        """List entities matching the given criteria."""
        start_time = time.time()

        try:
            all_entities = self.dao.get_entities()
            filtered_entities = []

            for entity in all_entities.values():
                match = True
                for key, value in criteria.items():
                    if key == "entity_type":
                        if entity.entity_type != value:
                            match = False
                            break
                    elif key == "name":
                        if value.lower() not in entity.name.lower():
                            match = False
                            break
                    elif key == "min_confidence":
                        if entity.confidence < value:
                            match = False
                            break
                    elif hasattr(entity, key):
                        if getattr(entity, key) != value:
                            match = False
                            break

                if match:
                    filtered_entities.append(entity)

            self._record_operation("list_by_criteria", start_time)

            return Result.ok(
                filtered_entities,
                metadata={
                    "operation": "list_entities_by_criteria",
                    "criteria": criteria,
                    "count": len(filtered_entities),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_criteria", start_time, False)
            return Result.internal_error(e)

    def search(self, query: str, limit: int = 10) -> Result[List["Entity"]]:
        """Search for entities matching the query."""
        start_time = time.time()

        try:
            entities = self.dao.search_entities(query, limit)
            self._record_operation("search", start_time)

            return Result.ok(
                entities,
                metadata={
                    "operation": "search_entities",
                    "query": query,
                    "limit": limit,
                    "count": len(entities),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("search", start_time, False)
            return Result.internal_error(e)

    def count(self) -> Result[int]:
        """Get the total count of entities."""
        start_time = time.time()

        try:
            entities_dict = self.dao.get_entities()
            count = len(entities_dict)
            self._record_operation("count", start_time)

            return Result.ok(
                count,
                metadata={
                    "operation": "count_entities",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("count", start_time, False)
            return Result.internal_error(e)

    def exists(self, item_id: str) -> Result[bool]:
        """Check if an entity exists."""
        start_time = time.time()

        try:
            entity = self.dao.get_entity_by_id(item_id)
            exists = entity is not None
            self._record_operation("exists", start_time)

            return Result.ok(
                exists,
                metadata={
                    "operation": "entity_exists",
                    "entity_id": item_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("exists", start_time, False)
            return Result.internal_error(e)

    def validate(self, item: "Entity") -> Result[bool]:
        """Validate an entity according to business rules."""
        start_time = time.time()

        try:
            errors = []

            # Check required fields
            if not item.id or not item.id.strip():
                errors.append(ErrorDetail(field_name="id", message="ID is required"))

            if not item.name or not item.name.strip():
                errors.append(ErrorDetail(field_name="name", message="Name is required"))

            if not item.entity_type:
                errors.append(ErrorDetail(field_name="entity_type", message="Entity type is required"))

            # Check confidence range
            if not 0.0 <= item.confidence <= 1.0:
                errors.append(ErrorDetail(field_name="confidence", message="Confidence must be between 0.0 and 1.0"))

            # Check name length
            if len(item.name) > 500:
                errors.append(ErrorDetail(field_name="name", message="Name must be less than 500 characters"))

            # Check description length
            if len(item.description) > 2000:
                errors.append(
                    ErrorDetail(
                        field_name="description",
                        message="Description must be less than 2000 characters",
                    )
                )

            self._record_operation("validate", start_time, len(errors) == 0)

            if errors:
                return Result.fail(
                    ErrorCode.INVALID_INPUT,
                    "Entity validation failed",
                    details=errors,
                    metadata={
                        "operation": "validate_entity",
                        "validation_errors": len(errors),
                        "execution_time": time.time() - start_time,
                    },
                )

            return Result.ok(
                True,
                metadata={
                    "operation": "validate_entity",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("validate", start_time, False)
            return Result.internal_error(e)

    def get_statistics(self) -> Result[Dict[str, Any]]:
        """Get statistics about the managed entities."""
        start_time = time.time()

        try:
            entities_dict = self.dao.get_entities()

            # Calculate type distribution
            type_distribution: Dict[str, int] = {}
            for entity in entities_dict.values():
                entity_type = str(entity.entity_type)
                type_distribution[entity_type] = type_distribution.get(entity_type, 0) + 1

            # Calculate confidence statistics
            confidences = [entity.confidence for entity in entities_dict.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            min_confidence = min(confidences) if confidences else 0.0
            max_confidence = max(confidences) if confidences else 0.0

            self._record_operation("get_statistics", start_time)

            stats = {
                "total_entities": len(entities_dict),
                "type_distribution": type_distribution,
                "confidence_stats": {
                    "average": avg_confidence,
                    "minimum": min_confidence,
                    "maximum": max_confidence,
                },
                "manager_metrics": self._metrics.copy(),
            }

            return Result.ok(
                stats,
                metadata={
                    "operation": "get_entity_statistics",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_statistics", start_time, False)
            return Result.internal_error(e)

    # EntityManager specific methods
    def list_by_type(self, entity_type: Union[EntityType, str]) -> Result[List["Entity"]]:
        """List entities by type."""
        start_time = time.time()

        try:
            entities = self.dao.get_entities_by_type(entity_type)
            self._record_operation("list_by_type", start_time)

            return Result.ok(
                entities,
                metadata={
                    "operation": "list_entities_by_type",
                    "entity_type": str(entity_type),
                    "count": len(entities),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_type", start_time, False)
            return Result.internal_error(e)

    def get_related_entities(
        self, entity_id: str, relation_types: Optional[List[RelationType]] = None
    ) -> Result[List["Entity"]]:
        """Get entities related to the given entity."""
        start_time = time.time()

        try:
            # Get all relations for this entity
            relations = self.dao.get_entity_relations(entity_id)

            # Filter by relation types if specified
            if relation_types:
                relations = [r for r in relations if r.relation_type in relation_types]

            # Collect related entity IDs
            related_entity_ids = set()
            for relation in relations:
                if relation.head_entity and relation.head_entity.id == entity_id:
                    if relation.tail_entity:
                        related_entity_ids.add(relation.tail_entity.id)
                elif relation.tail_entity and relation.tail_entity.id == entity_id:
                    if relation.head_entity:
                        related_entity_ids.add(relation.head_entity.id)

            # Get the actual entities
            related_entities = []
            for related_ent_id in related_entity_ids:
                entity = self.dao.get_entity_by_id(related_ent_id)
                if entity:
                    related_entities.append(entity)

            self._record_operation("get_related_entities", start_time)

            return Result.ok(
                related_entities,
                metadata={
                    "operation": "get_related_entities",
                    "source_entity_id": entity_id,
                    "relation_types": [str(rt) for rt in (relation_types or [])],
                    "count": len(related_entities),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_related_entities", start_time, False)
            return Result.internal_error(e)

    def update_confidence(self, entity_id: str, confidence: float) -> Result[bool]:
        """Update the confidence score of an entity."""
        start_time = time.time()

        try:
            # Validate confidence range
            if not 0.0 <= confidence <= 1.0:
                self._record_operation("update_confidence", start_time, False)
                return Result.invalid_input("Confidence must be between 0.0 and 1.0", field="confidence")

            # Get the entity
            entity = self.dao.get_entity_by_id(entity_id)
            if not entity:
                self._record_operation("update_confidence", start_time, False)
                return Result.not_found("Entity", entity_id)

            # Update confidence
            entity.confidence = confidence
            self.dao.save_entity(entity)

            self._record_operation("update_confidence", start_time)

            return Result.ok(
                True,
                metadata={
                    "operation": "update_entity_confidence",
                    "entity_id": entity_id,
                    "new_confidence": confidence,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("update_confidence", start_time, False)
            return Result.internal_error(e)


class UnifiedRelationManager(RelationManager):
    """
    Unified relation manager implementation using DAO layer.
    """

    def __init__(self, dao: DataAccessLayer, enable_validation: bool = True):
        self.dao = dao
        self.enable_validation = enable_validation
        self._lock = RLock()
        self._metrics = {
            "operations_count": 0,
            "errors_count": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _record_operation(self, operation_name: str, start_time: float, success: bool = True) -> None:
        """Record operation metrics."""
        execution_time = time.time() - start_time
        with self._lock:
            self._metrics["operations_count"] += 1
            if not success:
                self._metrics["errors_count"] += 1

            # Update average response time
            total_operations = self._metrics["operations_count"]
            current_avg = self._metrics["average_response_time"]
            self._metrics["average_response_time"] = (
                current_avg * (total_operations - 1) + execution_time
            ) / total_operations

    def add(self, item: "Relation") -> Result["Relation"]:
        """Add a relation to the knowledge graph."""
        start_time = time.time()

        try:
            # Validate the relation
            if self.enable_validation:
                validation_result = self.validate(item)
                if not validation_result.is_ok():
                    self._record_operation("add", start_time, False)
                    return Result.fail(
                        ErrorCode.VALIDATION_ERROR,
                        validation_result.error_message or "Validation failed",
                    )

            # Check for duplicates
            existing = self.dao.get_relation_by_id(item.id)
            if existing:
                self._record_operation("add", start_time, False)
                return Result.fail(ErrorCode.DUPLICATE_ENTRY, f"Relation with ID '{item.id}' already exists")

            # Verify that referenced entities exist
            if item.head_entity and not self.dao.get_entity_by_id(item.head_entity.id):
                self._record_operation("add", start_time, False)
                return Result.fail(
                    ErrorCode.DEPENDENCY_ERROR,
                    f"Head entity '{item.head_entity.id}' does not exist",
                )

            if item.tail_entity and not self.dao.get_entity_by_id(item.tail_entity.id):
                self._record_operation("add", start_time, False)
                return Result.fail(
                    ErrorCode.DEPENDENCY_ERROR,
                    f"Tail entity '{item.tail_entity.id}' does not exist",
                )

            # Save the relation
            self.dao.save_relation(item)
            self._record_operation("add", start_time)

            return Result.ok(
                item,
                metadata={
                    "operation": "add_relation",
                    "relation_type": str(item.relation_type),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("add", start_time, False)
            return Result.internal_error(e)

    def remove(self, item_id: str) -> Result[bool]:
        """Remove a relation from the knowledge graph."""
        start_time = time.time()

        try:
            # Check if relation exists
            existing = self.dao.get_relation_by_id(item_id)
            if not existing:
                self._record_operation("remove", start_time, False)
                return Result.not_found("Relation", item_id)

            # Remove the relation
            success = self.dao.delete_relation(item_id)
            self._record_operation("remove", start_time, success)

            return Result.ok(
                success,
                metadata={
                    "operation": "remove_relation",
                    "relation_id": item_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("remove", start_time, False)
            return Result.internal_error(e)

    def get(self, item_id: str) -> Result[Optional["Relation"]]:
        """Get a relation by ID."""
        start_time = time.time()

        try:
            relation = self.dao.get_relation_by_id(item_id)
            self._record_operation("get", start_time)

            return Result.ok(
                relation,
                metadata={
                    "operation": "get_relation",
                    "found": relation is not None,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get", start_time, False)
            return Result.internal_error(e)

    def list_all(self) -> Result[List["Relation"]]:
        """List all relations."""
        start_time = time.time()

        try:
            relations_dict = self.dao.get_relations()
            relations_list = list(relations_dict.values())
            self._record_operation("list_all", start_time)

            return Result.ok(
                relations_list,
                metadata={
                    "operation": "list_all_relations",
                    "count": len(relations_list),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_all", start_time, False)
            return Result.internal_error(e)

    def list_by_criteria(self, criteria: Dict[str, Any]) -> Result[List["Relation"]]:
        """List relations matching the given criteria."""
        start_time = time.time()

        try:
            all_relations = self.dao.get_relations()
            filtered_relations = []

            for relation in all_relations.values():
                match = True
                for key, value in criteria.items():
                    if key == "relation_type":
                        if relation.relation_type != value:
                            match = False
                            break
                    elif key == "head_entity_id":
                        if not relation.head_entity or relation.head_entity.id != value:
                            match = False
                            break
                    elif key == "tail_entity_id":
                        if not relation.tail_entity or relation.tail_entity.id != value:
                            match = False
                            break
                    elif hasattr(relation, key):
                        if getattr(relation, key) != value:
                            match = False
                            break

                if match:
                    filtered_relations.append(relation)

            self._record_operation("list_by_criteria", start_time)

            return Result.ok(
                filtered_relations,
                metadata={
                    "operation": "list_relations_by_criteria",
                    "criteria": criteria,
                    "count": len(filtered_relations),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_criteria", start_time, False)
            return Result.internal_error(e)

    def search(self, query: str, limit: int = 10) -> Result[List["Relation"]]:
        """Search for relations matching the query."""
        start_time = time.time()

        try:
            all_relations = self.dao.get_relations()
            query_lower = query.lower()
            matches = []

            for relation in all_relations.values():
                if query_lower in relation.description.lower() or query_lower in str(relation.relation_type).lower():
                    matches.append(relation)
                    if len(matches) >= limit:
                        break

            self._record_operation("search", start_time)

            return Result.ok(
                matches,
                metadata={
                    "operation": "search_relations",
                    "query": query,
                    "limit": limit,
                    "count": len(matches),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("search", start_time, False)
            return Result.internal_error(e)

    def count(self) -> Result[int]:
        """Get the total count of relations."""
        start_time = time.time()

        try:
            relations_dict = self.dao.get_relations()
            count = len(relations_dict)
            self._record_operation("count", start_time)

            return Result.ok(
                count,
                metadata={
                    "operation": "count_relations",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("count", start_time, False)
            return Result.internal_error(e)

    def exists(self, item_id: str) -> Result[bool]:
        """Check if a relation exists."""
        start_time = time.time()

        try:
            relation = self.dao.get_relation_by_id(item_id)
            exists = relation is not None
            self._record_operation("exists", start_time)

            return Result.ok(
                exists,
                metadata={
                    "operation": "relation_exists",
                    "relation_id": item_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("exists", start_time, False)
            return Result.internal_error(e)

    def validate(self, item: "Relation") -> Result[bool]:
        """Validate a relation according to business rules."""
        start_time = time.time()

        try:
            errors = []

            # Check required fields
            if not item.relation_type:
                errors.append(ErrorDetail(field_name="relation_type", message="Relation type is required"))

            if not item.head_entity and not item.tail_entity:
                errors.append(ErrorDetail(field_name="entities", message="At least one entity must be specified"))

            # Check description length
            if len(item.description) > 1000:
                errors.append(
                    ErrorDetail(
                        field_name="description",
                        message="Description must be less than 1000 characters",
                    )
                )

            self._record_operation("validate", start_time, len(errors) == 0)

            if errors:
                return Result.fail(
                    ErrorCode.VALIDATION_ERROR,
                    "Relation validation failed",
                    details=errors,
                    metadata={
                        "operation": "validate_relation",
                        "validation_errors": len(errors),
                        "execution_time": time.time() - start_time,
                    },
                )

            return Result.ok(
                True,
                metadata={
                    "operation": "validate_relation",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("validate", start_time, False)
            return Result.internal_error(e)

    def get_statistics(self) -> Result[Dict[str, Any]]:
        """Get statistics about the managed relations."""
        start_time = time.time()

        try:
            relations_dict = self.dao.get_relations()

            # Calculate type distribution
            type_distribution: Dict[str, int] = {}
            for relation in relations_dict.values():
                relation_type = str(relation.relation_type)
                type_distribution[relation_type] = type_distribution.get(relation_type, 0) + 1

            self._record_operation("get_statistics", start_time)

            stats = {
                "total_relations": len(relations_dict),
                "type_distribution": type_distribution,
                "manager_metrics": self._metrics.copy(),
            }

            return Result.ok(
                stats,
                metadata={
                    "operation": "get_relation_statistics",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_statistics", start_time, False)
            return Result.internal_error(e)

    # RelationManager specific methods
    def list_by_type(self, relation_type: Union[RelationType, str]) -> Result[List["Relation"]]:
        """List relations by type."""
        start_time = time.time()

        try:
            relations = self.dao.get_relations_by_type(relation_type)
            self._record_operation("list_by_type", start_time)

            return Result.ok(
                relations,
                metadata={
                    "operation": "list_relations_by_type",
                    "relation_type": str(relation_type),
                    "count": len(relations),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_type", start_time, False)
            return Result.internal_error(e)

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> Result[List["Relation"]]:
        """Get all relations connected to an entity."""
        start_time = time.time()

        try:
            all_relations = self.dao.get_entity_relations(entity_id)

            # Apply direction filter
            filtered_relations = []
            for relation in all_relations:
                if direction == "both":
                    filtered_relations.append(relation)
                elif direction == "outgoing":
                    if relation.head_entity and relation.head_entity.id == entity_id:
                        filtered_relations.append(relation)
                elif direction == "incoming":
                    if relation.tail_entity and relation.tail_entity.id == entity_id:
                        filtered_relations.append(relation)

            self._record_operation("get_entity_relations", start_time)

            return Result.ok(
                filtered_relations,
                metadata={
                    "operation": "get_entity_relations",
                    "entity_id": entity_id,
                    "direction": direction,
                    "count": len(filtered_relations),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_entity_relations", start_time, False)
            return Result.internal_error(e)

    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Result[List["Relation"]]:
        """Find a path between two entities."""
        start_time = time.time()
        # pylint: disable=too-many-nested-blocks
        try:
            # Simple BFS path finding
            if source_id == target_id:
                self._record_operation("find_path", start_time)
                return Result.ok(
                    [],
                    metadata={
                        "operation": "find_path",
                        "source_id": source_id,
                        "target_id": target_id,
                        "path_length": 0,
                        "execution_time": time.time() - start_time,
                    },
                )

            visited = set()
            queue: list[tuple[str, list["Relation"]]] = [(source_id, [])]

            for _ in range(max_depth):
                if not queue:
                    break

                next_queue = []
                for current_id, path in queue:
                    if current_id in visited:
                        continue
                    visited.add(current_id)

                    relations = self.dao.get_entity_relations(current_id)
                    for relation in relations:
                        next_id = None
                        if relation.head_entity and relation.head_entity.id == current_id:
                            if relation.tail_entity:
                                next_id = relation.tail_entity.id
                        elif relation.tail_entity and relation.tail_entity.id == current_id:
                            if relation.head_entity:
                                next_id = relation.head_entity.id

                        if next_id:
                            new_path = path + [relation]
                            if next_id == target_id:
                                self._record_operation("find_path", start_time)
                                return Result.ok(
                                    new_path,
                                    metadata={
                                        "operation": "find_path",
                                        "source_id": source_id,
                                        "target_id": target_id,
                                        "path_length": len(new_path),
                                        "execution_time": time.time() - start_time,
                                    },
                                )

                            if next_id not in visited:
                                next_queue.append((next_id, new_path))

                queue = next_queue

            # No path found
            self._record_operation("find_path", start_time)
            return Result.fail(
                ErrorCode.NOT_FOUND,
                f"No path found between '{source_id}' and '{target_id}' within {max_depth} steps",
                metadata={
                    "operation": "find_path",
                    "source_id": source_id,
                    "target_id": target_id,
                    "max_depth": max_depth,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("find_path", start_time, False)
            return Result.internal_error(e)


class UnifiedClusterManager(ClusterManager):
    """Unified cluster manager implementation using the DAO layer."""

    def __init__(self, dao: "DataAccessLayer"):
        """Initialize the unified cluster manager."""
        self.dao = dao
        self._metrics = {
            "operations_count": 0,
            "errors_count": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _record_operation(self, operation: str, start_time: float, success: bool = True) -> None:
        """Record operation metrics."""
        execution_time = time.time() - start_time
        self._metrics["operations_count"] += 1
        if not success:
            self._metrics["errors_count"] += 1

        # Update average response time
        total_operations = self._metrics["operations_count"]
        current_avg = self._metrics["average_response_time"]
        self._metrics["average_response_time"] = (
            current_avg * (total_operations - 1) + execution_time
        ) / total_operations

    # Base Manager interface methods
    def add(self, item: "Cluster") -> Result["Cluster"]:
        """Add a cluster to the managed collection."""
        start_time = time.time()

        try:
            # Validate cluster
            validation = self.validate(item)
            if not validation.is_ok():
                self._record_operation("add", start_time, False)
                return Result.fail(ErrorCode.VALIDATION_ERROR, validation.error_message or "Validation failed")

            # Check for duplicates
            existing = self.dao.get_cluster_by_id(item.id)
            if existing:
                self._record_operation("add", start_time, False)
                return Result.fail(ErrorCode.DUPLICATE_ENTRY, f"Cluster with ID '{item.id}' already exists")

            # Add cluster
            self.dao.save_cluster(item)
            self._record_operation("add", start_time)

            return Result.ok(
                item,
                metadata={
                    "operation": "add_cluster",
                    "cluster_id": item.id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("add", start_time, False)
            return Result.internal_error(e)

    def remove(self, item_id: str) -> Result[bool]:
        """Remove a cluster from the managed collection."""
        start_time = time.time()

        try:
            # Check if cluster exists
            existing = self.dao.get_cluster_by_id(item_id)
            if not existing:
                self._record_operation("remove", start_time, False)
                return Result.not_found("Cluster", item_id)

            # Remove cluster
            success = self.dao.delete_cluster(item_id)
            self._record_operation("remove", start_time, success)

            if success:
                return Result.ok(
                    True,
                    metadata={
                        "operation": "remove_cluster",
                        "cluster_id": item_id,
                        "execution_time": time.time() - start_time,
                    },
                )

            return Result.fail(ErrorCode.INTERNAL_ERROR, f"Failed to remove cluster with ID '{item_id}'")

        except Exception as e:
            self._record_operation("remove", start_time, False)
            return Result.internal_error(e)

    def get(self, item_id: str) -> Result[Optional["Cluster"]]:
        """Get a cluster by its ID."""
        start_time = time.time()

        try:
            cluster = self.dao.get_cluster_by_id(item_id)
            self._record_operation("get", start_time)

            return Result.ok(
                cluster,
                metadata={
                    "operation": "get_cluster",
                    "cluster_id": item_id,
                    "found": cluster is not None,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get", start_time, False)
            return Result.internal_error(e)

    def list_all(self) -> Result[List["Cluster"]]:
        """List all clusters in the managed collection."""
        start_time = time.time()

        try:
            clusters_dict = self.dao.get_clusters()
            clusters = list(clusters_dict.values())
            self._record_operation("list_all", start_time)

            return Result.ok(
                clusters,
                metadata={
                    "operation": "list_all_clusters",
                    "count": len(clusters),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_all", start_time, False)
            return Result.internal_error(e)

    def list_by_criteria(self, criteria: Dict[str, Any]) -> Result[List["Cluster"]]:
        """List clusters matching the given criteria."""
        start_time = time.time()

        try:
            clusters_dict = self.dao.get_clusters()
            matching_clusters = []

            for cluster in clusters_dict.values():
                matches = True
                for key, value in criteria.items():
                    if hasattr(cluster, key):
                        attr_value = getattr(cluster, key)
                        if attr_value != value:
                            matches = False
                            break
                    else:
                        matches = False
                        break

                if matches:
                    matching_clusters.append(cluster)

            self._record_operation("list_by_criteria", start_time)

            return Result.ok(
                matching_clusters,
                metadata={
                    "operation": "list_clusters_by_criteria",
                    "criteria": criteria,
                    "count": len(matching_clusters),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_criteria", start_time, False)
            return Result.internal_error(e)

    def search(self, query: str, limit: int = 10) -> Result[List["Cluster"]]:
        """Search for clusters matching the query."""
        start_time = time.time()

        try:
            clusters_dict = self.dao.get_clusters()
            matching_clusters = []
            query_lower = query.lower()

            for cluster in clusters_dict.values():
                # Search in cluster attributes
                def _matches_query(cluster: Any) -> bool:
                    """Check if cluster matches the search query."""
                    return (
                        query_lower in cluster.id.lower()
                        or (hasattr(cluster, "name") and cluster.name and query_lower in cluster.name.lower())
                        or (
                            hasattr(cluster, "description")
                            and cluster.description
                            and query_lower in cluster.description.lower()
                        )
                    )

                if _matches_query(cluster):
                    matching_clusters.append(cluster)
                    if len(matching_clusters) >= limit:
                        break

            self._record_operation("search", start_time)

            return Result.ok(
                matching_clusters,
                metadata={
                    "operation": "search_clusters",
                    "query": query,
                    "limit": limit,
                    "count": len(matching_clusters),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("search", start_time, False)
            return Result.internal_error(e)

    def count(self) -> Result[int]:
        """Get the total count of clusters."""
        start_time = time.time()

        try:
            clusters_dict = self.dao.get_clusters()
            count = len(clusters_dict)
            self._record_operation("count", start_time)

            return Result.ok(
                count,
                metadata={
                    "operation": "count_clusters",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("count", start_time, False)
            return Result.internal_error(e)

    def exists(self, item_id: str) -> Result[bool]:
        """Check if a cluster exists."""
        start_time = time.time()

        try:
            cluster = self.dao.get_cluster_by_id(item_id)
            exists = cluster is not None
            self._record_operation("exists", start_time)

            return Result.ok(
                exists,
                metadata={
                    "operation": "cluster_exists",
                    "cluster_id": item_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("exists", start_time, False)
            return Result.internal_error(e)

    def validate(self, item: "Cluster") -> Result[bool]:
        """Validate a cluster according to business rules."""
        start_time = time.time()

        try:
            # Basic validation
            if not isinstance(item, Cluster):
                self._record_operation("validate", start_time, False)
                return Result.invalid_input("Item must be a Cluster instance")

            if not item.id or not item.id.strip():
                self._record_operation("validate", start_time, False)
                return Result.invalid_input("Cluster ID cannot be empty", "id")

            # Check if cluster_type is valid (could be enum or string)
            valid_cluster_types = {ct.value for ct in ClusterType} | {str(ct) for ct in ClusterType}
            if item.cluster_type not in valid_cluster_types:
                self._record_operation("validate", start_time, False)
                return Result.invalid_input("Invalid cluster type", "cluster_type")

            self._record_operation("validate", start_time)

            return Result.ok(
                True,
                metadata={
                    "operation": "validate_cluster",
                    "cluster_id": item.id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("validate", start_time, False)
            return Result.internal_error(e)

    def get_statistics(self) -> Result[Dict[str, Any]]:
        """Get statistics about the managed clusters."""
        start_time = time.time()

        try:
            clusters_dict = self.dao.get_clusters()

            # Calculate type distribution
            type_distribution: Dict[str, int] = {}
            entity_counts = {}

            for cluster in clusters_dict.values():
                cluster_type = str(cluster.cluster_type)
                type_distribution[cluster_type] = type_distribution.get(cluster_type, 0) + 1

                # Count entities in cluster
                entities = self.dao.get_cluster_entities(cluster.id)
                entity_counts[cluster.id] = len(entities)

            self._record_operation("get_statistics", start_time)

            stats = {
                "total_clusters": len(clusters_dict),
                "type_distribution": type_distribution,
                "entity_counts": entity_counts,
                "average_entities_per_cluster": (
                    sum(entity_counts.values()) / len(entity_counts) if entity_counts else 0
                ),
                "manager_metrics": self._metrics.copy(),
            }

            return Result.ok(
                stats,
                metadata={
                    "operation": "get_cluster_statistics",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_statistics", start_time, False)
            return Result.internal_error(e)

    # ClusterManager specific methods
    def list_by_type(self, cluster_type: Union[ClusterType, str]) -> Result[List["Cluster"]]:
        """List clusters by type."""
        start_time = time.time()

        try:
            clusters = self.dao.get_clusters_by_type(cluster_type)
            self._record_operation("list_by_type", start_time)

            return Result.ok(
                clusters,
                metadata={
                    "operation": "list_clusters_by_type",
                    "cluster_type": str(cluster_type),
                    "count": len(clusters),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_type", start_time, False)
            return Result.internal_error(e)

    def get_cluster_entities(self, cluster_id: str) -> Result[List["Entity"]]:
        """Get all entities in a cluster."""
        start_time = time.time()

        try:
            entities = self.dao.get_cluster_entities(cluster_id)
            self._record_operation("get_cluster_entities", start_time)

            return Result.ok(
                entities,
                metadata={
                    "operation": "get_cluster_entities",
                    "cluster_id": cluster_id,
                    "count": len(entities),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_cluster_entities", start_time, False)
            return Result.internal_error(e)

    def add_entity_to_cluster(self, cluster_id: str, entity_id: str) -> Result[bool]:
        """Add an entity to a cluster."""
        start_time = time.time()

        try:
            # Verify cluster exists
            cluster = self.dao.get_cluster_by_id(cluster_id)
            if not cluster:
                self._record_operation("add_entity_to_cluster", start_time, False)
                return Result.not_found("Cluster", cluster_id)

            # Verify entity exists
            entity = self.dao.get_entity_by_id(entity_id)
            if not entity:
                self._record_operation("add_entity_to_cluster", start_time, False)
                return Result.not_found("Entity", entity_id)

            # Add entity to cluster
            success = self.dao.add_entity_to_cluster(cluster_id, entity_id)
            self._record_operation("add_entity_to_cluster", start_time, success)

            return Result.ok(
                success,
                metadata={
                    "operation": "add_entity_to_cluster",
                    "cluster_id": cluster_id,
                    "entity_id": entity_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("add_entity_to_cluster", start_time, False)
            return Result.internal_error(e)

    def remove_entity_from_cluster(self, cluster_id: str, entity_id: str) -> Result[bool]:
        """Remove an entity from a cluster."""
        start_time = time.time()

        try:
            # Verify cluster exists
            cluster = self.dao.get_cluster_by_id(cluster_id)
            if not cluster:
                self._record_operation("remove_entity_from_cluster", start_time, False)
                return Result.not_found("Cluster", cluster_id)

            # Remove entity from cluster
            success = self.dao.remove_entity_from_cluster(cluster_id, entity_id)
            self._record_operation("remove_entity_from_cluster", start_time, success)

            return Result.ok(
                success,
                metadata={
                    "operation": "remove_entity_from_cluster",
                    "cluster_id": cluster_id,
                    "entity_id": entity_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("remove_entity_from_cluster", start_time, False)
            return Result.internal_error(e)


class UnifiedTextChunkManager(TextChunkManager):
    """Unified text chunk manager implementation using the DAO layer."""

    def __init__(self, dao: "DataAccessLayer"):
        """Initialize the unified text chunk manager."""
        self.dao = dao
        self._metrics = {
            "operations_count": 0,
            "errors_count": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _record_operation(self, operation: str, start_time: float, success: bool = True) -> None:
        """Record operation metrics."""
        execution_time = time.time() - start_time
        self._metrics["operations_count"] += 1
        if not success:
            self._metrics["errors_count"] += 1

        # Update average response time
        total_operations = self._metrics["operations_count"]
        current_avg = self._metrics["average_response_time"]
        self._metrics["average_response_time"] = (
            current_avg * (total_operations - 1) + execution_time
        ) / total_operations

    # Base Manager interface methods
    def add(self, item: "TextChunk") -> Result["TextChunk"]:
        """Add a text chunk to the managed collection."""
        start_time = time.time()

        try:
            # Validate text chunk
            validation = self.validate(item)
            if not validation.is_ok():
                self._record_operation("add", start_time, False)
                return Result.fail(
                    validation.error_code or ErrorCode.INVALID_INPUT,
                    validation.error_message or "Validation failed",
                )

            # Check for duplicates
            existing = self.dao.get_text_chunk_by_id(item.id)
            if existing:
                self._record_operation("add", start_time, False)
                return Result.fail(ErrorCode.DUPLICATE_ENTRY, f"TextChunk with ID '{item.id}' already exists")

            # Add text chunk
            self.dao.save_text_chunk(item)
            self._record_operation("add", start_time)

            return Result.ok(
                item,
                metadata={
                    "operation": "add_text_chunk",
                    "chunk_id": item.id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("add", start_time, False)
            return Result.internal_error(e)

    def remove(self, item_id: str) -> Result[bool]:
        """Remove a text chunk from the managed collection."""
        start_time = time.time()

        try:
            # Check if text chunk exists
            existing = self.dao.get_text_chunk_by_id(item_id)
            if not existing:
                self._record_operation("remove", start_time, False)
                return Result.not_found("TextChunk", item_id)

            # Remove text chunk
            success = self.dao.delete_text_chunk(item_id)
            self._record_operation("remove", start_time, success)

            if success:
                return Result.ok(
                    True,
                    metadata={
                        "operation": "remove_text_chunk",
                        "chunk_id": item_id,
                        "execution_time": time.time() - start_time,
                    },
                )

            return Result.fail(ErrorCode.INTERNAL_ERROR, f"Failed to remove text chunk with ID '{item_id}'")

        except Exception as e:
            self._record_operation("remove", start_time, False)
            return Result.internal_error(e)

    def get(self, item_id: str) -> Result[Optional["TextChunk"]]:
        """Get a text chunk by its ID."""
        start_time = time.time()

        try:
            chunk = self.dao.get_text_chunk_by_id(item_id)
            self._record_operation("get", start_time)

            return Result.ok(
                chunk,
                metadata={
                    "operation": "get_text_chunk",
                    "chunk_id": item_id,
                    "found": chunk is not None,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get", start_time, False)
            return Result.internal_error(e)

    def list_all(self) -> Result[List["TextChunk"]]:
        """List all text chunks in the managed collection."""
        start_time = time.time()

        try:
            chunks_dict = self.dao.get_text_chunks()
            chunks = list(chunks_dict.values())
            self._record_operation("list_all", start_time)

            return Result.ok(
                chunks,
                metadata={
                    "operation": "list_all_text_chunks",
                    "count": len(chunks),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_all", start_time, False)
            return Result.internal_error(e)

    def list_by_criteria(self, criteria: Dict[str, Any]) -> Result[List["TextChunk"]]:
        """List text chunks matching the given criteria."""
        start_time = time.time()

        try:
            chunks_dict = self.dao.get_text_chunks()
            matching_chunks = []

            for chunk in chunks_dict.values():
                matches = True
                for key, value in criteria.items():
                    if hasattr(chunk, key):
                        attr_value = getattr(chunk, key)
                        if attr_value != value:
                            matches = False
                            break
                    else:
                        matches = False
                        break

                if matches:
                    matching_chunks.append(chunk)

            self._record_operation("list_by_criteria", start_time)

            return Result.ok(
                matching_chunks,
                metadata={
                    "operation": "list_text_chunks_by_criteria",
                    "criteria": criteria,
                    "count": len(matching_chunks),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_criteria", start_time, False)
            return Result.internal_error(e)

    def search(self, query: str, limit: int = 10) -> Result[List["TextChunk"]]:
        """Search for text chunks matching the query."""
        start_time = time.time()

        try:
            chunks_dict = self.dao.get_text_chunks()
            matching_chunks = []
            query_lower = query.lower()

            for chunk in chunks_dict.values():
                # Search in chunk content and metadata
                if (
                    query_lower in chunk.content.lower()
                    or query_lower in chunk.id.lower()
                    or (hasattr(chunk, "source") and chunk.source and query_lower in chunk.source.lower())
                ):
                    matching_chunks.append(chunk)
                    if len(matching_chunks) >= limit:
                        break

            self._record_operation("search", start_time)

            return Result.ok(
                matching_chunks,
                metadata={
                    "operation": "search_text_chunks",
                    "query": query,
                    "limit": limit,
                    "count": len(matching_chunks),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("search", start_time, False)
            return Result.internal_error(e)

    def count(self) -> Result[int]:
        """Get the total count of text chunks."""
        start_time = time.time()

        try:
            chunks_dict = self.dao.get_text_chunks()
            count = len(chunks_dict)
            self._record_operation("count", start_time)

            return Result.ok(
                count,
                metadata={
                    "operation": "count_text_chunks",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("count", start_time, False)
            return Result.internal_error(e)

    def exists(self, item_id: str) -> Result[bool]:
        """Check if a text chunk exists."""
        start_time = time.time()

        try:
            chunk = self.dao.get_text_chunk_by_id(item_id)
            exists = chunk is not None
            self._record_operation("exists", start_time)

            return Result.ok(
                exists,
                metadata={
                    "operation": "text_chunk_exists",
                    "chunk_id": item_id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("exists", start_time, False)
            return Result.internal_error(e)

    def validate(self, item: "TextChunk") -> Result[bool]:
        """Validate a text chunk according to business rules."""
        start_time = time.time()

        try:
            # Basic validation
            if not isinstance(item, TextChunk):
                self._record_operation("validate", start_time, False)
                return Result.invalid_input("Item must be a TextChunk instance")

            if not item.id or not item.id.strip():
                self._record_operation("validate", start_time, False)
                return Result.invalid_input("TextChunk ID cannot be empty", "id")

            if not item.content or not item.content.strip():
                self._record_operation("validate", start_time, False)
                return Result.invalid_input("TextChunk text cannot be empty", "text")

            self._record_operation("validate", start_time)

            return Result.ok(
                True,
                metadata={
                    "operation": "validate_text_chunk",
                    "chunk_id": item.id,
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("validate", start_time, False)
            return Result.internal_error(e)

    def get_statistics(self) -> Result[Dict[str, Any]]:
        """Get statistics about the managed text chunks."""
        start_time = time.time()

        try:
            chunks_dict = self.dao.get_text_chunks()

            # Calculate statistics
            total_chars = sum(len(chunk.content) for chunk in chunks_dict.values())
            source_distribution: dict[str, int] = {}

            for chunk in chunks_dict.values():
                source = getattr(chunk, "source", "unknown")
                source_distribution[source] = source_distribution.get(source, 0) + 1

            self._record_operation("get_statistics", start_time)

            stats = {
                "total_chunks": len(chunks_dict),
                "total_characters": total_chars,
                "average_chunk_size": total_chars / len(chunks_dict) if chunks_dict else 0,
                "source_distribution": source_distribution,
                "manager_metrics": self._metrics.copy(),
            }

            return Result.ok(
                stats,
                metadata={
                    "operation": "get_text_chunk_statistics",
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_statistics", start_time, False)
            return Result.internal_error(e)

    # TextChunkManager specific methods
    def list_by_source(self, source: str) -> Result[List["TextChunk"]]:
        """List text chunks by source."""
        start_time = time.time()

        try:
            chunks = self.dao.get_text_chunks_by_source(source)
            self._record_operation("list_by_source", start_time)

            return Result.ok(
                chunks,
                metadata={
                    "operation": "list_text_chunks_by_source",
                    "source": source,
                    "count": len(chunks),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("list_by_source", start_time, False)
            return Result.internal_error(e)

    def get_chunk_entities(self, chunk_id: str) -> Result[List["Entity"]]:
        """Get all entities referenced in a text chunk."""
        start_time = time.time()

        try:
            entities = self.dao.get_chunk_entities(chunk_id)
            self._record_operation("get_chunk_entities", start_time)

            return Result.ok(
                entities,
                metadata={
                    "operation": "get_chunk_entities",
                    "chunk_id": chunk_id,
                    "count": len(entities),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_chunk_entities", start_time, False)
            return Result.internal_error(e)

    def get_chunk_relations(self, chunk_id: str) -> Result[List["Relation"]]:
        """Get all relations referenced in a text chunk."""
        start_time = time.time()

        try:
            relations = self.dao.get_chunk_relations(chunk_id)
            self._record_operation("get_chunk_relations", start_time)

            return Result.ok(
                relations,
                metadata={
                    "operation": "get_chunk_relations",
                    "chunk_id": chunk_id,
                    "count": len(relations),
                    "execution_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._record_operation("get_chunk_relations", start_time, False)
            return Result.internal_error(e)
