"""Cache data viewing router."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ...agraph import AGraph
from ...logger import logger
from ..dependencies import get_agraph_instance_dependency
from ..models import CacheDataResponse, ResponseStatus

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/text-chunks", response_model=CacheDataResponse)
async def get_cached_text_chunks(
    page: int = 1,
    page_size: int = 10,
    filter_by: Optional[str] = None,
    agraph: AGraph = Depends(get_agraph_instance_dependency),
) -> CacheDataResponse:
    """Get cached text chunks with pagination."""
    try:
        if not agraph.knowledge_graph or not agraph.knowledge_graph.text_chunks:
            return CacheDataResponse(
                status=ResponseStatus.SUCCESS,
                message="No text chunks found in cache",
                data={
                    "text_chunks": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                },
            )

        text_chunks = list(agraph.knowledge_graph.text_chunks.values())

        # Apply filter if provided
        if filter_by:
            text_chunks = [
                chunk
                for chunk in text_chunks
                if filter_by.lower() in chunk.content.lower()
                or (chunk.title and filter_by.lower() in chunk.title.lower())
                or (chunk.source and filter_by.lower() in chunk.source.lower())
            ]

        total_count = len(text_chunks)
        total_pages = (total_count + page_size - 1) // page_size

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_chunks = text_chunks[start_idx:end_idx]

        # Convert to API models
        chunk_data = []
        for chunk in paginated_chunks:
            chunk_data.append(
                {
                    "id": chunk.id,
                    "content": (
                        chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    ),
                    "full_content": chunk.content,
                    "title": chunk.title,
                    "source": chunk.source,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "content_length": len(chunk.content),
                }
            )

        return CacheDataResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(chunk_data)} text chunks from cache",
            data={
                "text_chunks": chunk_data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "filter_by": filter_by,
            },
        )

    except Exception as e:
        logger.error(f"Failed to get cached text chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entities", response_model=CacheDataResponse)
async def get_cached_entities(
    page: int = 1,
    page_size: int = 10,
    filter_by: Optional[str] = None,
    agraph: AGraph = Depends(get_agraph_instance_dependency),
) -> CacheDataResponse:
    """Get cached entities with pagination."""
    try:
        if not agraph.knowledge_graph or not agraph.knowledge_graph.entities:
            return CacheDataResponse(
                status=ResponseStatus.SUCCESS,
                message="No entities found in cache",
                data={
                    "entities": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                },
            )

        entities = list(agraph.knowledge_graph.entities.values())

        # Apply filter if provided
        if filter_by:
            entities = [
                entity
                for entity in entities
                if filter_by.lower() in entity.name.lower()
                or (entity.description and filter_by.lower() in entity.description.lower())
                or filter_by.lower()
                in (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else str(entity.entity_type)
                ).lower()
            ]

        total_count = len(entities)
        total_pages = (total_count + page_size - 1) // page_size

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_entities = entities[start_idx:end_idx]

        # Convert to API models
        entity_data = []
        for entity in paginated_entities:
            entity_data.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": (
                        entity.entity_type.value
                        if hasattr(entity.entity_type, "value")
                        else str(entity.entity_type)
                    ),
                    "description": entity.description,
                    "properties": entity.properties,
                    "confidence": entity.confidence,
                    "aliases": list(entity.aliases) if entity.aliases else [],
                    "text_chunks": list(entity.text_chunks) if entity.text_chunks else [],
                }
            )

        return CacheDataResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(entity_data)} entities from cache",
            data={
                "entities": entity_data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "filter_by": filter_by,
            },
        )

    except Exception as e:
        logger.error(f"Failed to get cached entities: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/relations", response_model=CacheDataResponse)
async def get_cached_relations(
    page: int = 1,
    page_size: int = 10,
    filter_by: Optional[str] = None,
    agraph: AGraph = Depends(get_agraph_instance_dependency),
) -> CacheDataResponse:
    """Get cached relations with pagination."""
    try:
        if not agraph.knowledge_graph or not agraph.knowledge_graph.relations:
            return CacheDataResponse(
                status=ResponseStatus.SUCCESS,
                message="No relations found in cache",
                data={
                    "relations": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                },
            )

        relations = list(agraph.knowledge_graph.relations.values())

        # Apply filter if provided
        if filter_by:
            relations = [
                relation
                for relation in relations
                if filter_by.lower()
                in (
                    relation.relation_type.value
                    if hasattr(relation.relation_type, "value")
                    else str(relation.relation_type)
                ).lower()
                or (relation.description and filter_by.lower() in relation.description.lower())
                or (relation.head_entity and filter_by.lower() in relation.head_entity.name.lower())
                or (relation.tail_entity and filter_by.lower() in relation.tail_entity.name.lower())
            ]

        total_count = len(relations)
        total_pages = (total_count + page_size - 1) // page_size

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_relations = relations[start_idx:end_idx]

        # Convert to API models
        relation_data = []
        for relation in paginated_relations:
            relation_data.append(
                {
                    "id": relation.id,
                    "head_entity": {
                        "id": relation.head_entity.id if relation.head_entity else None,
                        "name": relation.head_entity.name if relation.head_entity else None,
                        "entity_type": (
                            (
                                relation.head_entity.entity_type.value
                                if hasattr(relation.head_entity.entity_type, "value")
                                else str(relation.head_entity.entity_type)
                            )
                            if relation.head_entity
                            else None
                        ),
                    },
                    "tail_entity": {
                        "id": relation.tail_entity.id if relation.tail_entity else None,
                        "name": relation.tail_entity.name if relation.tail_entity else None,
                        "entity_type": (
                            (
                                relation.tail_entity.entity_type.value
                                if hasattr(relation.tail_entity.entity_type, "value")
                                else str(relation.tail_entity.entity_type)
                            )
                            if relation.tail_entity
                            else None
                        ),
                    },
                    "relation_type": (
                        relation.relation_type.value
                        if hasattr(relation.relation_type, "value")
                        else str(relation.relation_type)
                    ),
                    "description": relation.description,
                    "properties": relation.properties,
                    "confidence": relation.confidence,
                    "text_chunks": list(relation.text_chunks) if relation.text_chunks else [],
                }
            )

        return CacheDataResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(relation_data)} relations from cache",
            data={
                "relations": relation_data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "filter_by": filter_by,
            },
        )

    except Exception as e:
        logger.error(f"Failed to get cached relations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/clusters", response_model=CacheDataResponse)
async def get_cached_clusters(
    page: int = 1,
    page_size: int = 10,
    filter_by: Optional[str] = None,
    agraph: AGraph = Depends(get_agraph_instance_dependency),
) -> CacheDataResponse:
    """Get cached clusters with pagination."""
    try:
        if not agraph.knowledge_graph or not agraph.knowledge_graph.clusters:
            return CacheDataResponse(
                status=ResponseStatus.SUCCESS,
                message="No clusters found in cache",
                data={
                    "clusters": [],
                    "total_count": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                },
            )

        clusters = list(agraph.knowledge_graph.clusters.values())

        # Apply filter if provided
        if filter_by:
            clusters = [
                cluster
                for cluster in clusters
                if filter_by.lower() in cluster.name.lower()
                or (cluster.description and filter_by.lower() in cluster.description.lower())
            ]

        total_count = len(clusters)
        total_pages = (total_count + page_size - 1) // page_size

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_clusters = clusters[start_idx:end_idx]

        # Convert to API models
        cluster_data = []
        for cluster in paginated_clusters:
            cluster_data.append(
                {
                    "id": cluster.id,
                    "name": cluster.name,
                    "description": cluster.description,
                    "entities": (
                        [
                            {
                                "id": entity.id if hasattr(entity, "id") else str(entity),
                                "name": entity.name if hasattr(entity, "name") else str(entity),
                                "entity_type": (
                                    entity.entity_type.value
                                    if hasattr(entity, "entity_type")
                                    and hasattr(entity.entity_type, "value")
                                    else str(
                                        entity.entity_type if hasattr(entity, "entity_type") else ""
                                    )
                                ),
                            }
                            for entity in cluster.entities
                        ]
                        if cluster.entities
                        else []
                    ),
                    "relations": (
                        [
                            {
                                "id": relation.id if hasattr(relation, "id") else str(relation),
                                "relation_type": (
                                    relation.relation_type.value
                                    if hasattr(relation, "relation_type")
                                    and hasattr(relation.relation_type, "value")
                                    else str(
                                        relation.relation_type
                                        if hasattr(relation, "relation_type")
                                        else ""
                                    )
                                ),
                                "head_entity_name": (
                                    relation.head_entity.name
                                    if hasattr(relation, "head_entity") and relation.head_entity
                                    else None
                                ),
                                "tail_entity_name": (
                                    relation.tail_entity.name
                                    if hasattr(relation, "tail_entity") and relation.tail_entity
                                    else None
                                ),
                            }
                            for relation in cluster.relations
                        ]
                        if cluster.relations
                        else []
                    ),
                    "entity_count": len(cluster.entities) if cluster.entities else 0,
                    "relation_count": len(cluster.relations) if cluster.relations else 0,
                }
            )

        return CacheDataResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(cluster_data)} clusters from cache",
            data={
                "clusters": cluster_data,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "filter_by": filter_by,
            },
        )

    except Exception as e:
        logger.error(f"Failed to get cached clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
