"""
Event listeners for AGraph system components.

This module provides event listeners that integrate with the existing
IndexManager, CacheManager, and other system components to automatically
handle data changes through the event system.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from ..core.result import ErrorCode, Result
from ..infrastructure.cache import CacheManager
from ..infrastructure.indexes import IndexManager
from .events import EventListener, EventType, GraphEvent

if TYPE_CHECKING:
    from ..infrastructure.dao import DataAccessLayer
    from .events import EventManager


class IndexUpdateListener(EventListener):
    """
    Event listener that updates indexes in response to entity and relation changes.

    This listener ensures that all indexes remain consistent when entities
    or relations are added, updated, or removed.
    """

    def __init__(self, index_manager: IndexManager):
        """
        Initialize the index update listener.

        Args:
            index_manager: The IndexManager instance to update
        """
        self.index_manager = index_manager
        self.logger = logging.getLogger(__name__)

    def get_event_types(self) -> Set[EventType]:
        """Get the event types this listener handles."""
        return {
            EventType.ENTITY_ADDED,
            EventType.ENTITY_UPDATED,
            EventType.ENTITY_REMOVED,
            EventType.RELATION_ADDED,
            EventType.RELATION_UPDATED,
            EventType.RELATION_REMOVED,
            EventType.CLUSTER_ENTITY_ADDED,
            EventType.CLUSTER_ENTITY_REMOVED,
        }

    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """
        Handle an event by updating the appropriate indexes.

        Args:
            event: The event to handle

        Returns:
            Result indicating success or failure
        """
        try:
            if event.event_type == EventType.ENTITY_ADDED:
                return self._handle_entity_added(event)
            elif event.event_type == EventType.ENTITY_UPDATED:
                return self._handle_entity_updated(event)
            elif event.event_type == EventType.ENTITY_REMOVED:
                return self._handle_entity_removed(event)
            elif event.event_type == EventType.RELATION_ADDED:
                return self._handle_relation_added(event)
            elif event.event_type == EventType.RELATION_UPDATED:
                return self._handle_relation_updated(event)
            elif event.event_type == EventType.RELATION_REMOVED:
                return self._handle_relation_removed(event)
            elif event.event_type == EventType.CLUSTER_ENTITY_ADDED:
                return self._handle_cluster_entity_added(event)
            elif event.event_type == EventType.CLUSTER_ENTITY_REMOVED:
                return self._handle_cluster_entity_removed(event)
            else:
                return Result.ok(True)  # Event not handled, but not an error

        except Exception as e:
            self.logger.error(f"Error handling event {event.event_type}: {str(e)}")
            return Result.internal_error(e)

    def _handle_entity_added(self, event: GraphEvent) -> Result[bool]:
        """Handle entity added event."""
        if not event.target_id or not event.data:
            return Result.fail(ErrorCode.INVALID_INPUT, "Missing entity ID or data")

        entity_type = event.data.get("entity_type")
        if entity_type:
            self.index_manager.add_entity_to_type_index(event.target_id, entity_type)

        # Note: Search index functionality not available in current IndexManager implementation
        # Would add search terms indexing here when available

        return Result.ok(True)

    def _handle_entity_updated(self, event: GraphEvent) -> Result[bool]:
        """Handle entity updated event."""
        if not event.target_id or not event.data:
            return Result.fail(ErrorCode.INVALID_INPUT, "Missing entity ID or data")

        # For updates, we need to remove old indexes and add new ones
        # In a more sophisticated implementation, we'd compare old vs new data
        # Note: Search index functionality not available in current IndexManager implementation

        return Result.ok(True)

    def _handle_entity_removed(self, event: GraphEvent) -> Result[bool]:
        """Handle entity removed event."""
        if not event.target_id:
            return Result.fail(ErrorCode.INVALID_INPUT, "Missing entity ID")

        entity_id = event.target_id

        # Remove from type index
        if event.data and "entity_type" in event.data:
            entity_type = event.data["entity_type"]
            self.index_manager.remove_entity_from_type_index(entity_id, entity_type)

        # Remove from all indexes (comprehensive cleanup)
        self.index_manager.remove_entity_from_all_indexes(entity_id)

        # Note: Search index removal not available in current IndexManager

        # Note: Cluster index cleanup handled by remove_entity_from_all_indexes above

        return Result.ok(True)

    def _handle_relation_added(self, event: GraphEvent) -> Result[bool]:
        """Handle relation added event."""
        if not event.target_id or not event.data:
            return Result.fail(ErrorCode.INVALID_INPUT, "Missing relation ID or data")

        relation_id = event.target_id
        head_entity_id = None
        tail_entity_id = None

        # Extract entity IDs from relation data
        if "head_entity" in event.data:
            head_data = event.data["head_entity"]
            if isinstance(head_data, dict) and "id" in head_data:
                head_entity_id = head_data["id"]
            elif isinstance(head_data, str):
                head_entity_id = head_data

        if "tail_entity" in event.data:
            tail_data = event.data["tail_entity"]
            if isinstance(tail_data, dict) and "id" in tail_data:
                tail_entity_id = tail_data["id"]
            elif isinstance(tail_data, str):
                tail_entity_id = tail_data

        # Add to relation indexes using existing method
        if head_entity_id and tail_entity_id:
            self.index_manager.add_relation_to_index(relation_id, head_entity_id, tail_entity_id)

        # Note: Relation type index not available in current IndexManager

        return Result.ok(True)

    def _handle_relation_updated(self, event: GraphEvent) -> Result[bool]:
        """Handle relation updated event."""
        # For relation updates, the most common case is confidence score changes
        # The structural relationship (head/tail entities) typically doesn't change
        # So we don't need to update indexes for most relation updates
        return Result.ok(True)

    def _handle_relation_removed(self, event: GraphEvent) -> Result[bool]:
        """Handle relation removed event."""
        if not event.target_id:
            return Result.fail(ErrorCode.INVALID_INPUT, "Missing relation ID")

        relation_id = event.target_id

        # Remove from relation indexes using existing method
        self.index_manager.remove_relation_from_index(relation_id)

        # Note: Relation type index not available in current IndexManager

        return Result.ok(True)

    def _handle_cluster_entity_added(self, event: GraphEvent) -> Result[bool]:
        """Handle cluster entity added event."""
        if not event.data:
            return Result.ok(True)

        cluster_id = event.data.get("cluster_id")
        entity_id = event.data.get("entity_id")

        if cluster_id and entity_id:
            self.index_manager.add_entity_to_cluster_index(cluster_id, entity_id)

        return Result.ok(True)

    def _handle_cluster_entity_removed(self, event: GraphEvent) -> Result[bool]:
        """Handle cluster entity removed event."""
        if not event.data:
            return Result.ok(True)

        cluster_id = event.data.get("cluster_id")
        entity_id = event.data.get("entity_id")

        if cluster_id and entity_id:
            self.index_manager.remove_entity_from_cluster_index(cluster_id, entity_id)

        return Result.ok(True)


class CacheInvalidationListener(EventListener):
    """
    Event listener that invalidates cache entries in response to data changes.

    This listener ensures that cached data remains consistent by selectively
    invalidating cache entries when the underlying data changes.
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize the cache invalidation listener.

        Args:
            cache_manager: The CacheManager instance to update
        """
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

    def get_event_types(self) -> Set[EventType]:
        """Get the event types this listener handles."""
        return {
            EventType.ENTITY_ADDED,
            EventType.ENTITY_UPDATED,
            EventType.ENTITY_REMOVED,
            EventType.RELATION_ADDED,
            EventType.RELATION_UPDATED,
            EventType.RELATION_REMOVED,
            EventType.CLUSTER_ADDED,
            EventType.CLUSTER_UPDATED,
            EventType.CLUSTER_REMOVED,
            EventType.TEXT_CHUNK_ADDED,
            EventType.TEXT_CHUNK_UPDATED,
            EventType.TEXT_CHUNK_REMOVED,
            EventType.GRAPH_CLEARED,
        }

    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """
        Handle an event by invalidating relevant cache entries.

        Args:
            event: The event to handle

        Returns:
            Result indicating success or failure
        """
        try:
            if event.event_type in {
                EventType.ENTITY_ADDED,
                EventType.ENTITY_UPDATED,
                EventType.ENTITY_REMOVED,
            }:
                return self._handle_entity_change(event)
            elif event.event_type in {
                EventType.RELATION_ADDED,
                EventType.RELATION_UPDATED,
                EventType.RELATION_REMOVED,
            }:
                return self._handle_relation_change(event)
            elif event.event_type in {
                EventType.CLUSTER_ADDED,
                EventType.CLUSTER_UPDATED,
                EventType.CLUSTER_REMOVED,
            }:
                return self._handle_cluster_change(event)
            elif event.event_type in {
                EventType.TEXT_CHUNK_ADDED,
                EventType.TEXT_CHUNK_UPDATED,
                EventType.TEXT_CHUNK_REMOVED,
            }:
                return self._handle_text_chunk_change(event)
            elif event.event_type == EventType.GRAPH_CLEARED:
                return self._handle_graph_cleared(event)
            else:
                return Result.ok(True)  # Event not handled, but not an error

        except Exception as e:
            self.logger.error(
                f"Error handling cache invalidation for event {event.event_type}: {str(e)}"
            )
            return Result.internal_error(e)

    def _handle_entity_change(self, event: GraphEvent) -> Result[bool]:
        """Handle entity-related changes."""
        if not event.target_id:
            return Result.ok(True)

        # Invalidate caches with entity-related tags
        tags_to_invalidate = [
            "entities",
            f"entity:{event.target_id}",
            "graph_stats",
            "entity_counts",
        ]

        # If we know the entity type, also invalidate type-specific caches
        if event.data and "entity_type" in event.data:
            entity_type = event.data["entity_type"]
            tags_to_invalidate.append(f"entity_type:{entity_type}")

        # Convert to set and invalidate
        tags_set = set(tags_to_invalidate)
        self.cache_manager.invalidate_by_tags(tags_set)

        return Result.ok(True)

    def _handle_relation_change(self, event: GraphEvent) -> Result[bool]:
        """Handle relation-related changes."""
        if not event.target_id:
            return Result.ok(True)

        # Invalidate caches with relation-related tags
        tags_to_invalidate = [
            "relations",
            f"relation:{event.target_id}",
            "graph_stats",
            "relation_counts",
        ]

        # If we know the relation type, also invalidate type-specific caches
        if event.data and "relation_type" in event.data:
            relation_type = event.data["relation_type"]
            tags_to_invalidate.append(f"relation_type:{relation_type}")

        # If we know the connected entities, invalidate their relation caches
        if event.data:
            for entity_key in ["head_entity", "tail_entity"]:
                if entity_key in event.data:
                    entity_data = event.data[entity_key]
                    entity_id = None
                    if isinstance(entity_data, dict) and "id" in entity_data:
                        entity_id = entity_data["id"]
                    elif isinstance(entity_data, str):
                        entity_id = entity_data

                    if entity_id:
                        tags_to_invalidate.append(f"entity_relations:{entity_id}")

        # Convert to set and invalidate
        tags_set = set(tags_to_invalidate)
        self.cache_manager.invalidate_by_tags(tags_set)

        return Result.ok(True)

    def _handle_cluster_change(self, event: GraphEvent) -> Result[bool]:
        """Handle cluster-related changes."""
        if not event.target_id:
            return Result.ok(True)

        # Invalidate caches with cluster-related tags
        tags_to_invalidate = [
            "clusters",
            f"cluster:{event.target_id}",
            "graph_stats",
            "cluster_counts",
        ]

        # Convert to set and invalidate
        tags_set = set(tags_to_invalidate)
        self.cache_manager.invalidate_by_tags(tags_set)

        return Result.ok(True)

    def _handle_text_chunk_change(self, event: GraphEvent) -> Result[bool]:
        """Handle text chunk-related changes."""
        if not event.target_id:
            return Result.ok(True)

        # Invalidate caches with text chunk-related tags
        tags_to_invalidate = ["text_chunks", f"text_chunk:{event.target_id}", "graph_stats"]

        # Convert to set and invalidate
        tags_set = set(tags_to_invalidate)
        self.cache_manager.invalidate_by_tags(tags_set)

        return Result.ok(True)

    def _handle_graph_cleared(self, event: GraphEvent) -> Result[bool]:
        """Handle graph cleared event."""
        # Clear all caches when the graph is cleared
        self.cache_manager.clear()
        return Result.ok(True)


class IntegrityCheckListener(EventListener):
    """
    Event listener that performs integrity checks in response to data changes.

    This listener validates data consistency and can detect issues like:
    - Dangling references
    - Circular dependencies
    - Invalid relationships
    - Data constraint violations
    """

    def __init__(self, dao_layer: Optional["DataAccessLayer"] = None):
        """
        Initialize the integrity check listener.

        Args:
            dao_layer: The DAO layer for data access during integrity checks
        """
        self.dao_layer = dao_layer
        self.logger = logging.getLogger(__name__)
        self.integrity_issues: List[Dict[str, Any]] = []

    def get_event_types(self) -> Set[EventType]:
        """Get the event types this listener handles."""
        return {
            EventType.ENTITY_ADDED,
            EventType.ENTITY_REMOVED,
            EventType.RELATION_ADDED,
            EventType.RELATION_REMOVED,
            EventType.CLUSTER_ENTITY_ADDED,
            EventType.CLUSTER_ENTITY_REMOVED,
        }

    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """
        Handle an event by performing integrity checks.

        Args:
            event: The event to handle

        Returns:
            Result indicating success or failure
        """
        try:
            if event.event_type == EventType.ENTITY_REMOVED:
                return self._check_dangling_relations(event)
            elif event.event_type == EventType.RELATION_ADDED:
                return self._check_relation_validity(event)
            elif event.event_type in {
                EventType.CLUSTER_ENTITY_ADDED,
                EventType.CLUSTER_ENTITY_REMOVED,
            }:
                return self._check_cluster_consistency(event)
            else:
                return Result.ok(True)  # Event not handled, but not an error

        except Exception as e:
            self.logger.error(
                f"Error performing integrity check for event {event.event_type}: {str(e)}"
            )
            return Result.internal_error(e)

    def _check_dangling_relations(self, event: GraphEvent) -> Result[bool]:
        """Check for dangling relations after entity removal."""
        if not self.dao_layer or not event.target_id:
            return Result.ok(True)

        entity_id = event.target_id

        # Find relations that reference the removed entity
        all_relations = self.dao_layer.get_relations()
        dangling_relations = []

        for relation_id, relation in all_relations.items():
            head_id = getattr(relation.head_entity, "id", None) if relation.head_entity else None
            tail_id = getattr(relation.tail_entity, "id", None) if relation.tail_entity else None

            if head_id == entity_id or tail_id == entity_id:
                dangling_relations.append(relation_id)
                self.integrity_issues.append(
                    {
                        "type": "dangling_relation",
                        "relation_id": relation_id,
                        "removed_entity_id": entity_id,
                        "timestamp": event.timestamp,
                    }
                )

        if dangling_relations:
            self.logger.warning(
                f"Found {len(dangling_relations)} dangling relations after removing entity {entity_id}"
            )
            # In a production system, you might want to automatically clean up these relations
            # or trigger a separate cleanup process

        return Result.ok(True, metadata={"dangling_relations_count": len(dangling_relations)})

    def _check_relation_validity(self, event: GraphEvent) -> Result[bool]:
        """Check if a new relation is valid."""
        if not self.dao_layer or not event.data:
            return Result.ok(True)

        # Extract entity IDs from relation data
        head_entity_id = None
        tail_entity_id = None

        if "head_entity" in event.data:
            head_data = event.data["head_entity"]
            if isinstance(head_data, dict) and "id" in head_data:
                head_entity_id = head_data["id"]
            elif isinstance(head_data, str):
                head_entity_id = head_data

        if "tail_entity" in event.data:
            tail_data = event.data["tail_entity"]
            if isinstance(tail_data, dict) and "id" in tail_data:
                tail_entity_id = tail_data["id"]
            elif isinstance(tail_data, str):
                tail_entity_id = tail_data

        # Check if both entities exist
        issues = []
        if head_entity_id and not self.dao_layer.get_entity_by_id(head_entity_id):
            issues.append(f"Head entity {head_entity_id} does not exist")

        if tail_entity_id and not self.dao_layer.get_entity_by_id(tail_entity_id):
            issues.append(f"Tail entity {tail_entity_id} does not exist")

        # Check for self-references (entity relating to itself)
        if head_entity_id == tail_entity_id:
            issues.append(f"Self-reference detected: entity {head_entity_id} relates to itself")

        if issues:
            for issue in issues:
                self.integrity_issues.append(
                    {
                        "type": "relation_validity",
                        "relation_id": event.target_id,
                        "issue": issue,
                        "timestamp": event.timestamp,
                    }
                )
                self.logger.warning(f"Relation integrity issue: {issue}")

        return Result.ok(True, metadata={"integrity_issues_count": len(issues)})

    def _check_cluster_consistency(self, event: GraphEvent) -> Result[bool]:
        """Check cluster consistency after entity additions/removals."""
        if not self.dao_layer or not event.data:
            return Result.ok(True)

        cluster_id = event.data.get("cluster_id")
        entity_id = event.data.get("entity_id")

        if not cluster_id or not entity_id:
            return Result.ok(True)

        # Verify cluster exists
        cluster = self.dao_layer.get_cluster_by_id(cluster_id)
        if not cluster:
            self.integrity_issues.append(
                {
                    "type": "cluster_consistency",
                    "cluster_id": cluster_id,
                    "issue": f"Cluster {cluster_id} does not exist",
                    "timestamp": event.timestamp,
                }
            )
            return Result.ok(True, metadata={"integrity_issues_count": 1})

        # Verify entity exists (for additions)
        if event.event_type == EventType.CLUSTER_ENTITY_ADDED:
            entity = self.dao_layer.get_entity_by_id(entity_id)
            if not entity:
                self.integrity_issues.append(
                    {
                        "type": "cluster_consistency",
                        "cluster_id": cluster_id,
                        "entity_id": entity_id,
                        "issue": f"Entity {entity_id} does not exist",
                        "timestamp": event.timestamp,
                    }
                )
                return Result.ok(True, metadata={"integrity_issues_count": 1})

        return Result.ok(True)

    def get_integrity_issues(self) -> List[Dict[str, Any]]:
        """Get all recorded integrity issues."""
        return self.integrity_issues.copy()

    def clear_integrity_issues(self) -> None:
        """Clear all recorded integrity issues."""
        self.integrity_issues.clear()


# Factory functions for creating listeners


def create_index_update_listener(index_manager: IndexManager) -> IndexUpdateListener:
    """Create an index update listener."""
    return IndexUpdateListener(index_manager)


def create_cache_invalidation_listener(cache_manager: CacheManager) -> CacheInvalidationListener:
    """Create a cache invalidation listener."""
    return CacheInvalidationListener(cache_manager)


def create_integrity_check_listener(
    dao_layer: Optional["DataAccessLayer"] = None,
) -> IntegrityCheckListener:
    """Create an integrity check listener."""
    return IntegrityCheckListener(dao_layer)


def setup_default_listeners(
    event_manager: "EventManager",
    index_manager: Optional[IndexManager] = None,
    cache_manager: Optional[CacheManager] = None,
    dao_layer: Optional["DataAccessLayer"] = None,
) -> Dict[str, str]:
    """
    Set up default event listeners for a knowledge graph system.

    Args:
        event_manager: The event manager to register listeners with
        index_manager: Optional index manager for index updates
        cache_manager: Optional cache manager for cache invalidation
        dao_layer: Optional DAO layer for integrity checks

    Returns:
        Dictionary mapping listener types to their subscription IDs
    """
    subscriptions = {}

    # Set up index update listener
    if index_manager:
        index_listener = create_index_update_listener(index_manager)
        result = event_manager.subscribe(index_listener)
        if result.is_ok():
            subscriptions["index_update"] = result.data

    # Set up cache invalidation listener
    if cache_manager:
        cache_listener = create_cache_invalidation_listener(cache_manager)
        result = event_manager.subscribe(cache_listener)
        if result.is_ok():
            subscriptions["cache_invalidation"] = result.data

    # Set up integrity check listener
    if dao_layer:
        integrity_listener = create_integrity_check_listener(dao_layer)
        result = event_manager.subscribe(integrity_listener)
        if result.is_ok():
            subscriptions["integrity_check"] = result.data

    return subscriptions
