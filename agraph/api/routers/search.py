"""Search functionality router."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...logger import logger
from ..dependencies import get_agraph_instance_dependency
from ..models import ResponseStatus, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest, agraph: Any = Depends(get_agraph_instance_dependency)) -> SearchResponse:
    """Search entities, relations, or text chunks."""
    try:
        results = []
        if request.search_type == "entities":
            search_results = await agraph.search_entities(
                query=request.query, top_k=request.top_k, filter_dict=request.filter_dict
            )
            results = [
                {
                    "entity": {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type.value,
                        "description": entity.description,
                        "confidence": entity.confidence,
                    },
                    "score": score,
                }
                for entity, score in search_results
            ]
        elif request.search_type == "relations":
            search_results = await agraph.search_relations(
                query=request.query, top_k=request.top_k, filter_dict=request.filter_dict
            )
            results = [
                {
                    "relation": {
                        "id": relation.id,
                        "head_entity_id": relation.head_entity.id if relation.head_entity else None,
                        "tail_entity_id": relation.tail_entity.id if relation.tail_entity else None,
                        "relation_type": relation.relation_type.value,
                        "description": relation.description,
                        "confidence": relation.confidence,
                    },
                    "score": score,
                }
                for relation, score in search_results
            ]
        elif request.search_type == "text_chunks":
            search_results = await agraph.search_text_chunks(
                query=request.query, top_k=request.top_k, filter_dict=request.filter_dict
            )
            results = [
                {
                    "text_chunk": {
                        "id": chunk.id,
                        "content": chunk.content,
                        "title": chunk.title,
                        "source": chunk.source,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                    },
                    "score": score,
                }
                for chunk, score in search_results
            ]
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid search_type. Must be 'entities', 'relations', or 'text_chunks'",
            )

        return SearchResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Search completed successfully. Found {len(results)} results",
            data={
                "query": request.query,
                "search_type": request.search_type,
                "results": results,
                "total_count": len(results),
            },
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
