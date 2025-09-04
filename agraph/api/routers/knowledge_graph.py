"""Knowledge graph construction router."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ...base.models.entities import Entity
from ...logger import logger
from ...processor.factory import DocumentProcessorManager
from ..dependencies import get_agraph_instance, get_document_manager
from ..models import (
    KnowledgeGraphBuildRequest,
    KnowledgeGraphBuildResponse,
    KnowledgeGraphGetResponse,
    KnowledgeGraphStatusResponse,
    KnowledgeGraphUpdateRequest,
    KnowledgeGraphUpdateResponse,
    KnowledgeGraphVisualizationRequest,
    KnowledgeGraphVisualizationResponse,
    ResponseStatus,
    TextChunkSearchRequest,
    TextChunkSearchResponse,
)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


def _extract_text_from_document(doc: Dict[str, Any]) -> str:
    """Extract text content from document, handling both text and binary formats.

    Args:
        doc: Document dictionary containing content and metadata

    Returns:
        Extracted text content

    Raises:
        ValueError: If content cannot be processed or extracted
    """
    content = doc["content"]

    # Check if this is a document with preserved original format
    metadata = doc.get("metadata", {})
    is_original_format = metadata.get("original_format", False)

    # If it's already text content (string), return as-is
    if isinstance(content, str):
        return content

    # If it's binary content, we need to extract text
    if isinstance(content, bytes) and is_original_format:
        filename = doc.get("filename", "unknown.txt")

        # Create temporary file to process the binary content
        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, delete=False
            ) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Use document processor to extract text
                processor_manager = DocumentProcessorManager()
                if processor_manager.can_process(temp_file_path):
                    text_content = processor_manager.process(temp_file_path)
                    return text_content

                # If processor can't handle it, try to decode as text
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Cannot extract text from binary content of file: {filename}"
                    ) from exc
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    logger.warning(f"Failed to delete temporary file: {temp_file_path}")

        except Exception as e:
            logger.error(f"Failed to extract text from document {filename}: {e}")
            raise ValueError(f"Failed to extract text from document: {str(e)}") from e

    # Handle other cases - try to convert to string
    try:
        return str(content)
    except Exception as e:
        raise ValueError(f"Cannot convert document content to text: {str(e)}") from e


@router.post("/build", response_model=KnowledgeGraphBuildResponse)
async def build_knowledge_graph(
    request: KnowledgeGraphBuildRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
) -> KnowledgeGraphBuildResponse:
    """Build knowledge graph from stored documents or direct texts."""
    try:
        # Get project-specific instances
        agraph = await get_agraph_instance(project_name)
        doc_manager = get_document_manager(project_name)
        texts_to_process = []
        document_info = []

        # Process document IDs if provided
        if request.document_ids:
            documents = doc_manager.get_documents_by_ids(request.document_ids)
            if not documents:
                raise HTTPException(status_code=404, detail="No documents found with provided IDs")

            for doc in documents:
                try:
                    text_content = _extract_text_from_document(doc)
                    texts_to_process.append(text_content)
                    document_info.append(
                        {
                            "id": doc["id"],
                            "filename": doc.get("filename", "Unknown"),
                            "content_length": len(text_content),
                        }
                    )
                except ValueError as e:
                    logger.warning(f"Skipping document {doc.get('filename', 'Unknown')}: {e}")
                    continue

            logger.info(f"Loaded {len(documents)} documents from storage")

        # Add direct texts if provided
        elif request.texts:
            texts_to_process.extend(request.texts)
            for i, text in enumerate(request.texts):
                document_info.append(
                    {
                        "id": f"direct_text_{i+1}",
                        "filename": f"direct_text_{i+1}.txt",
                        "content_length": len(text),
                    }
                )

        # If no document IDs or texts provided, use all documents from project storage
        else:
            all_doc_list, _ = doc_manager.list_documents(
                page=1, page_size=10000
            )  # Get all document IDs
            if not all_doc_list:
                raise HTTPException(
                    status_code=400,
                    detail="No documents found in project storage. Please upload documents first or provide texts.",
                )

            # Get full document content using IDs
            all_doc_ids = [doc["id"] for doc in all_doc_list]
            all_documents = doc_manager.get_documents_by_ids(all_doc_ids)

            for doc in all_documents:
                try:
                    text_content = _extract_text_from_document(doc)
                    texts_to_process.append(text_content)
                    document_info.append(
                        {
                            "id": doc["id"],
                            "filename": doc.get("filename", "Unknown"),
                            "content_length": len(text_content),
                        }
                    )
                except ValueError as e:
                    logger.warning(f"Skipping document {doc.get('filename', 'Unknown')}: {e}")
                    continue

            logger.info(f"Using all {len(all_documents)} documents from project storage")

        if not texts_to_process:
            raise HTTPException(
                status_code=400,
                detail="No documents or texts available for knowledge graph construction",
            )

        logger.info(f"Building knowledge graph from {len(texts_to_process)} text sources")

        # Build knowledge graph from texts
        kg = await agraph.build_from_texts(
            texts=texts_to_process,
            graph_name=request.graph_name,
            graph_description=request.graph_description,
            use_cache=request.use_cache,
            save_to_vector_store=request.save_to_vector_store,
        )

        response_data = {
            "graph_name": kg.name,
            "graph_description": kg.description,
            "entities_count": len(kg.entities),
            "relations_count": len(kg.relations),
            "clusters_count": len(kg.clusters),
            "text_chunks_count": len(kg.text_chunks),
            "source_documents": document_info,
            "total_texts_processed": len(texts_to_process),
            "from_stored_documents": len(
                [d for d in document_info if not d["id"].startswith("direct_text")]
            ),
            "from_direct_texts": len(
                [d for d in document_info if d["id"].startswith("direct_text")]
            ),
        }

        return KnowledgeGraphBuildResponse(
            status=ResponseStatus.SUCCESS,
            message="Knowledge graph built successfully",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/update", response_model=KnowledgeGraphUpdateResponse)
async def update_knowledge_graph(
    request: KnowledgeGraphUpdateRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
) -> KnowledgeGraphUpdateResponse:
    """Update existing knowledge graph with additional documents or texts."""
    try:
        # Get project-specific instances
        agraph = await get_agraph_instance(project_name)
        doc_manager = get_document_manager(project_name)
        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=400,
                detail="No existing knowledge graph found. Use /build endpoint first.",
            )

        additional_texts = []
        document_info = []

        # Process additional document IDs
        if request.additional_document_ids:
            documents = doc_manager.get_documents_by_ids(request.additional_document_ids)
            if documents:
                for doc in documents:
                    try:
                        text_content = _extract_text_from_document(doc)
                        additional_texts.append(text_content)
                        document_info.append(
                            {
                                "id": doc["id"],
                                "filename": doc.get("filename", "Unknown"),
                                "content_length": len(text_content),
                            }
                        )
                    except ValueError as e:
                        logger.warning(f"Skipping document {doc.get('filename', 'Unknown')}: {e}")
                        continue
                logger.info(f"Loaded {len(documents)} additional documents from storage")

        # Add additional direct texts
        elif request.additional_texts:
            additional_texts.extend(request.additional_texts)
            for i, text in enumerate(request.additional_texts):
                document_info.append(
                    {
                        "id": f"additional_text_{i+1}",
                        "filename": f"additional_text_{i+1}.txt",
                        "content_length": len(text),
                    }
                )

        # If no additional document IDs or texts provided, use all documents from project storage
        else:
            all_doc_list, _ = doc_manager.list_documents(
                page=1, page_size=10000
            )  # Get all document IDs
            if not all_doc_list:
                raise HTTPException(
                    status_code=400,
                    detail="No documents found in project storage. Please upload documents first or provide additional texts.",
                )

            # Get full document content using IDs
            all_doc_ids = [doc["id"] for doc in all_doc_list]
            all_documents = doc_manager.get_documents_by_ids(all_doc_ids)

            for doc in all_documents:
                try:
                    text_content = _extract_text_from_document(doc)
                    additional_texts.append(text_content)
                    document_info.append(
                        {
                            "id": doc["id"],
                            "filename": doc.get("filename", "Unknown"),
                            "content_length": len(text_content),
                        }
                    )
                except ValueError as e:
                    logger.warning(f"Skipping document {doc.get('filename', 'Unknown')}: {e}")
                    continue

            logger.info(f"Using all {len(all_documents)} documents from project storage for update")

        if not additional_texts:
            raise HTTPException(
                status_code=400, detail="No additional documents or texts available for update"
            )

        logger.info(
            f"Updating knowledge graph with {len(additional_texts)} additional text sources"
        )

        # agraph.enable_knowledge_graph = request.enable_graph  # Commented out - attribute not found

        # Get current knowledge graph info
        current_kg = agraph.knowledge_graph
        if not current_kg:
            raise HTTPException(status_code=400, detail="Current knowledge graph not accessible")

        current_stats = {
            "entities": len(current_kg.entities),
            "relations": len(current_kg.relations),
            "clusters": len(current_kg.clusters),
            "text_chunks": len(current_kg.text_chunks),
        }

        # Build updated knowledge graph (this will merge with existing)
        # Note: This is a simplified approach. In production, you might want
        # more sophisticated incremental update logic
        all_texts = []

        # Add existing text chunks
        for chunk in current_kg.text_chunks.values():
            all_texts.append(chunk.content)

        # Add new texts
        all_texts.extend(additional_texts)

        updated_kg = await agraph.build_from_texts(
            texts=all_texts,
            graph_name=current_kg.name,
            graph_description=current_kg.description,
            use_cache=request.use_cache,
            save_to_vector_store=request.save_to_vector_store,
        )

        new_stats = {
            "entities": len(updated_kg.entities),
            "relations": len(updated_kg.relations),
            "clusters": len(updated_kg.clusters),
            "text_chunks": len(updated_kg.text_chunks),
        }

        response_data = {
            "graph_name": updated_kg.name,
            "graph_description": updated_kg.description,
            "previous_stats": current_stats,
            "updated_stats": new_stats,
            "changes": {
                "entities_added": new_stats["entities"] - current_stats["entities"],
                "relations_added": new_stats["relations"] - current_stats["relations"],
                "clusters_added": new_stats["clusters"] - current_stats["clusters"],
                "text_chunks_added": new_stats["text_chunks"] - current_stats["text_chunks"],
            },
            "additional_sources": document_info,
            "total_additional_texts": len(additional_texts),
        }

        return KnowledgeGraphUpdateResponse(
            status=ResponseStatus.SUCCESS,
            message="Knowledge graph updated successfully",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status", response_model=KnowledgeGraphStatusResponse)
async def get_knowledge_graph_status(
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
) -> KnowledgeGraphStatusResponse:
    """Get current knowledge graph status and statistics."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)
        if not agraph.has_knowledge_graph:
            return KnowledgeGraphStatusResponse(
                status=ResponseStatus.SUCCESS,
                message="No knowledge graph currently loaded",
                data={
                    "exists": False,
                    "graph_name": None,
                    "graph_description": None,
                    "statistics": {"entities": 0, "relations": 0, "clusters": 0, "text_chunks": 0},
                },
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Gather statistics
        entity_types: Dict[str, int] = {}
        relation_types: Dict[str, int] = {}

        for entity in kg.entities.values():
            entity_type = (
                entity.entity_type.value
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type)
            )
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        for relation in kg.relations.values():
            relation_type = (
                relation.relation_type.value
                if hasattr(relation.relation_type, "value")
                else str(relation.relation_type)
            )
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1

        response_data = {
            "exists": True,
            "graph_name": kg.name,
            "graph_description": kg.description,
            "statistics": {
                "entities": len(kg.entities),
                "relations": len(kg.relations),
                "clusters": len(kg.clusters),
                "text_chunks": len(kg.text_chunks),
            },
            "entity_types": entity_types,
            "relation_types": relation_types,
            "system_info": {
                "agraph_initialized": agraph.is_initialized,
                "vector_store_type": agraph.vector_store_type,
                "enable_knowledge_graph": agraph.enable_knowledge_graph,
            },
        }

        return KnowledgeGraphStatusResponse(
            status=ResponseStatus.SUCCESS,
            message="Knowledge graph status retrieved successfully",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get knowledge graph status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/get", response_model=KnowledgeGraphGetResponse)
async def get_knowledge_graph(
    include_text_chunks: bool = Query(default=False, description="Include text chunks in response"),
    include_clusters: bool = Query(default=False, description="Include clusters in response"),
    entity_limit: Optional[int] = Query(default=None, description="Limit number of entities"),
    relation_limit: Optional[int] = Query(default=None, description="Limit number of relations"),
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
) -> KnowledgeGraphGetResponse:
    """Get complete knowledge graph data."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Convert entities to API format
        entities = []
        entity_list = list(kg.entities.values())
        if entity_limit:
            entity_list = entity_list[:entity_limit]

        for entity in entity_list:
            entities.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": (
                        entity.entity_type.value
                        if hasattr(entity.entity_type, "value")
                        else str(entity.entity_type)
                    ),
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "properties": entity.properties or {},
                    "aliases": entity.aliases or [],
                }
            )

        # Convert relations to API format
        relations = []
        relation_list = list(kg.relations.values())
        if relation_limit:
            relation_list = relation_list[:relation_limit]

        for relation in relation_list:
            relations.append(
                {
                    "id": relation.id,
                    "head_entity_id": relation.head_entity.id if relation.head_entity else None,
                    "tail_entity_id": relation.tail_entity.id if relation.tail_entity else None,
                    "relation_type": (
                        relation.relation_type.value
                        if hasattr(relation.relation_type, "value")
                        else str(relation.relation_type)
                    ),
                    "description": relation.description,
                    "confidence": relation.confidence,
                    "properties": relation.properties or {},
                }
            )

        response_data = {
            "graph_name": kg.name,
            "graph_description": kg.description,
            "entities": entities,
            "relations": relations,
        }

        # Include clusters if requested
        if include_clusters:
            clusters = []
            for cluster in kg.clusters.values():
                clusters.append(
                    {
                        "id": cluster.id,
                        "name": cluster.name,  # Use name instead of title for consistency
                        "description": cluster.description,
                        "entities": list(cluster.entities),
                        "relations": [],  # Add relations field for frontend compatibility
                        "confidence": getattr(cluster, "confidence", 1.0),
                    }
                )
            response_data["clusters"] = clusters

        # Include text chunks if requested
        if include_text_chunks:
            text_chunks = []
            for chunk in kg.text_chunks.values():
                text_chunks.append(
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "source": chunk.source,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "entities": getattr(chunk, "entities", []),
                        "relations": [],  # Add relations field for frontend compatibility
                    }
                )
            response_data["text_chunks"] = text_chunks

        return KnowledgeGraphGetResponse(
            status=ResponseStatus.SUCCESS,
            message="Knowledge graph retrieved successfully",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/visualization-data", response_model=KnowledgeGraphVisualizationResponse)
async def get_visualization_data(
    request: KnowledgeGraphVisualizationRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
) -> KnowledgeGraphVisualizationResponse:
    """Get knowledge graph data optimized for visualization."""
    try:  # pylint: disable=too-many-nested-blocks
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Filter entities by type and confidence
        filtered_entities = []
        for entity in kg.entities.values():
            # Filter by entity type
            entity_type = (
                entity.entity_type.value
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type)
            )
            if request.entity_types and entity_type not in request.entity_types:
                continue
            # Filter by confidence
            if entity.confidence < request.min_confidence:
                continue
            filtered_entities.append(entity)

        # Limit entities
        if len(filtered_entities) > request.max_entities:
            # Sort by confidence and take top entities
            filtered_entities = sorted(filtered_entities, key=lambda x: x.confidence, reverse=True)[
                : request.max_entities
            ]

        # Get entity IDs for filtering relations
        entity_ids = {entity.id for entity in filtered_entities}

        # Filter relations
        filtered_relations = []
        for relation in kg.relations.values():
            # Only include relations between filtered entities
            if (
                not relation.head_entity
                or not relation.tail_entity
                or (
                    relation.head_entity.id not in entity_ids
                    or relation.tail_entity.id not in entity_ids
                )
            ):
                continue
            # Filter by relation type
            relation_type = (
                relation.relation_type.value
                if hasattr(relation.relation_type, "value")
                else str(relation.relation_type)
            )
            if request.relation_types and relation_type not in request.relation_types:
                continue
            # Filter by confidence
            if relation.confidence < request.min_confidence:
                continue
            filtered_relations.append(relation)

        # Limit relations
        if len(filtered_relations) > request.max_relations:
            filtered_relations = sorted(
                filtered_relations, key=lambda x: x.confidence, reverse=True
            )[: request.max_relations]

        # Convert to visualization format
        nodes = []
        for entity in filtered_entities:
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": "entity",
                    "entityType": (
                        entity.entity_type.value
                        if hasattr(entity.entity_type, "value")
                        else str(entity.entity_type)
                    ),
                    "confidence": entity.confidence,
                    "properties": entity.properties or {},
                }
            )

        edges = []
        for relation in filtered_relations:
            relation_type = (
                relation.relation_type.value
                if hasattr(relation.relation_type, "value")
                else str(relation.relation_type)
            )
            edges.append(
                {
                    "id": relation.id,
                    "source": relation.head_entity.id if relation.head_entity else None,
                    "target": relation.tail_entity.id if relation.tail_entity else None,
                    "label": relation_type,
                    "relationType": relation_type,
                    "confidence": relation.confidence,
                    "properties": relation.properties or {},
                }
            )

        response_data = {
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_entities": len(kg.entities),
                "filtered_entities": len(nodes),
                "total_relations": len(kg.relations),
                "filtered_relations": len(edges),
            },
            "filters_applied": {
                "entity_types": request.entity_types,
                "relation_types": request.relation_types,
                "min_confidence": request.min_confidence,
                "max_entities": request.max_entities,
                "max_relations": request.max_relations,
            },
        }

        # Include clusters if requested
        if request.include_clusters:
            clusters = []
            for cluster in kg.clusters.values():
                # Only include clusters that have entities in the filtered set
                cluster_entity_ids = [
                    entity_id for entity_id in cluster.entities if entity_id in entity_ids
                ]
                if cluster_entity_ids:
                    clusters.append(
                        {
                            "id": cluster.id,
                            "name": cluster.name,  # Use name instead of title for consistency
                            "description": cluster.description,
                            "entities": cluster_entity_ids,
                            "relations": [],  # Add relations field for frontend compatibility
                            "confidence": getattr(cluster, "confidence", 1.0),
                        }
                    )
            response_data["clusters"] = clusters

        # Include text chunks if requested
        if request.include_text_chunks:
            text_chunks = []
            for chunk in kg.text_chunks.values():
                # Get all entities associated with this text chunk
                chunk_entities = getattr(chunk, "entities", [])
                all_chunk_entity_objs = []
                filtered_chunk_entity_objs = []

                if isinstance(chunk_entities, set):
                    # If entities is a set of IDs, need to get the actual entity objects
                    for entity_id in chunk_entities:
                        chunk_entity: Optional[Entity] = kg.entities.get(entity_id)
                        if chunk_entity is not None and hasattr(chunk_entity, "id"):
                            all_chunk_entity_objs.append(chunk_entity)
                            # Also track which entities are in the filtered set
                            if chunk_entity.id in entity_ids:
                                filtered_chunk_entity_objs.append(chunk_entity)
                else:
                    # entities is already a list of entity objects
                    for chunk_entity in chunk_entities:
                        if hasattr(chunk_entity, "id"):
                            all_chunk_entity_objs.append(chunk_entity)
                            # Also track which entities are in the filtered set
                            if chunk_entity.id in entity_ids:
                                filtered_chunk_entity_objs.append(chunk_entity)

                # Include all text chunks, but mark which entities are in the filtered set
                text_chunks.append(
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "source": chunk.source,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "entities": [entity.id for entity in all_chunk_entity_objs],
                        "filtered_entities": [entity.id for entity in filtered_chunk_entity_objs],
                        "relations": [],  # Add relations field for frontend compatibility
                        "entity_details": [
                            {
                                "id": entity.id,
                                "name": entity.name,
                                "entity_type": (
                                    entity.entity_type.value
                                    if hasattr(entity.entity_type, "value")
                                    else str(entity.entity_type)
                                ),
                                "in_filtered_set": entity.id in entity_ids,
                            }
                            for entity in all_chunk_entity_objs
                        ],
                    }
                )
            response_data["text_chunks"] = text_chunks

        return KnowledgeGraphVisualizationResponse(
            status=ResponseStatus.SUCCESS,
            message="Visualization data retrieved successfully",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/text-chunks", response_model=Dict[str, Any])
async def get_text_chunks(
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
    limit: Optional[int] = Query(default=None, description="Limit number of text chunks"),
    offset: int = Query(default=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """Get all text chunks from the knowledge graph."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Get all text chunks
        all_chunks = list(kg.text_chunks.values())
        total_count = len(all_chunks)

        # Apply pagination
        start_idx = offset
        if limit:
            end_idx = start_idx + limit
            paginated_chunks = all_chunks[start_idx:end_idx]
        else:
            paginated_chunks = all_chunks[start_idx:]
            end_idx = len(all_chunks)

        # Convert to API format
        text_chunks = []
        for chunk in paginated_chunks:
            text_chunks.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "entities": list(chunk.entities) if hasattr(chunk, "entities") else [],
                    "relations": list(chunk.relations) if hasattr(chunk, "relations") else [],
                    "created_at": (
                        chunk.created_at.isoformat() if hasattr(chunk, "created_at") else None
                    ),
                    "updated_at": (
                        chunk.updated_at.isoformat() if hasattr(chunk, "updated_at") else None
                    ),
                }
            )

        return {
            "status": "success",
            "message": f"Retrieved {len(text_chunks)} text chunks",
            "data": {
                "text_chunks": text_chunks,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": end_idx < total_count,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get text chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/text-chunks", response_model=TextChunkSearchResponse)
async def search_text_chunks(
    request: TextChunkSearchRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
) -> TextChunkSearchResponse:
    """Search and retrieve text chunks from the knowledge graph."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Get all text chunks
        all_chunks = list(kg.text_chunks.values())
        filtered_chunks = []

        for chunk in all_chunks:
            # Filter by entity ID if provided
            if request.entity_id:
                # chunk.entities is a set of entity IDs (strings)
                if request.entity_id not in chunk.entities:
                    continue

            # Filter by search query if provided
            if request.search:
                search_lower = request.search.lower()
                if (
                    search_lower not in chunk.content.lower()
                    and search_lower not in (chunk.source or "").lower()
                ):
                    continue

            filtered_chunks.append(chunk)

        # Apply pagination
        total_count = len(filtered_chunks)
        start_idx = request.offset
        end_idx = start_idx + request.limit
        paginated_chunks = filtered_chunks[start_idx:end_idx]

        # Convert to API format
        text_chunks = []
        for chunk in paginated_chunks:
            # Get entities associated with this text chunk
            chunk_entities = getattr(chunk, "entities", [])
            if isinstance(chunk_entities, set):
                # If entities is a set of IDs, need to get the actual entity objects
                chunk_entity_objs = []
                for entity_id in chunk_entities:
                    entity = kg.entities.get(entity_id)
                    if entity:
                        chunk_entity_objs.append(entity)
            else:
                chunk_entity_objs = chunk_entities

            text_chunks.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "entities": [
                        entity.id if hasattr(entity, "id") else str(entity)
                        for entity in chunk_entity_objs
                    ],
                    "relations": [],  # Add relations field for frontend compatibility
                    "entity_details": [
                        {
                            "id": entity.id,
                            "name": entity.name,
                            "entity_type": (
                                entity.entity_type.value
                                if hasattr(entity.entity_type, "value")
                                else str(entity.entity_type)
                            ),
                        }
                        for entity in chunk_entity_objs
                        if hasattr(entity, "id")
                    ],
                }
            )

        response_data = {
            "text_chunks": text_chunks,
            "pagination": {
                "total": total_count,
                "limit": request.limit,
                "offset": request.offset,
                "has_more": end_idx < total_count,
            },
            "filters": {
                "search": request.search,
                "entity_id": request.entity_id,
            },
        }

        return TextChunkSearchResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Found {total_count} text chunks (returning {len(text_chunks)})",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search text chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/entities", response_model=Dict[str, Any])
async def get_entities(
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
    entity_type: Optional[str] = Query(default=None, description="Filter by entity type"),
    limit: Optional[int] = Query(default=None, description="Limit number of entities"),
    offset: int = Query(default=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """Get all entities from the knowledge graph."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Get all entities
        all_entities = list(kg.entities.values())

        # Filter by entity type if provided
        if entity_type:
            filtered_entities = []
            for entity in all_entities:
                entity_type_value = (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else str(entity.entity_type)
                )
                if entity_type_value == entity_type:
                    filtered_entities.append(entity)
            all_entities = filtered_entities

        total_count = len(all_entities)

        # Apply pagination
        start_idx = offset
        if limit:
            end_idx = start_idx + limit
            paginated_entities = all_entities[start_idx:end_idx]
        else:
            paginated_entities = all_entities[start_idx:]
            end_idx = len(all_entities)

        # Convert to API format
        entities = []
        for entity in paginated_entities:
            entities.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": (
                        entity.entity_type.value
                        if hasattr(entity.entity_type, "value")
                        else str(entity.entity_type)
                    ),
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "properties": entity.properties or {},
                    "aliases": entity.aliases or [],
                    "text_chunks": (
                        list(entity.text_chunks) if hasattr(entity, "text_chunks") else []
                    ),
                    "created_at": (
                        entity.created_at.isoformat() if hasattr(entity, "created_at") else None
                    ),
                    "updated_at": (
                        entity.updated_at.isoformat() if hasattr(entity, "updated_at") else None
                    ),
                }
            )

        return {
            "status": "success",
            "message": f"Retrieved {len(entities)} entities",
            "data": {
                "entities": entities,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": end_idx < total_count,
                },
                "filters": {
                    "entity_type": entity_type,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entities: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/relations", response_model=Dict[str, Any])
async def get_relations(
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
    relation_type: Optional[str] = Query(default=None, description="Filter by relation type"),
    entity_id: Optional[str] = Query(
        default=None, description="Filter by entity ID (head or tail)"
    ),
    limit: Optional[int] = Query(default=None, description="Limit number of relations"),
    offset: int = Query(default=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """Get all relations from the knowledge graph."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Get all relations
        all_relations = list(kg.relations.values())

        # Filter by relation type if provided
        if relation_type:
            filtered_relations = []
            for relation in all_relations:
                relation_type_value = (
                    relation.relation_type.value
                    if hasattr(relation.relation_type, "value")
                    else str(relation.relation_type)
                )
                if relation_type_value == relation_type:
                    filtered_relations.append(relation)
            all_relations = filtered_relations

        # Filter by entity ID if provided
        if entity_id:
            filtered_relations = []
            for relation in all_relations:
                if (relation.head_entity and relation.head_entity.id == entity_id) or (
                    relation.tail_entity and relation.tail_entity.id == entity_id
                ):
                    filtered_relations.append(relation)
            all_relations = filtered_relations

        total_count = len(all_relations)

        # Apply pagination
        start_idx = offset
        if limit:
            end_idx = start_idx + limit
            paginated_relations = all_relations[start_idx:end_idx]
        else:
            paginated_relations = all_relations[start_idx:]
            end_idx = len(all_relations)

        # Convert to API format
        relations = []
        for relation in paginated_relations:
            relations.append(
                {
                    "id": relation.id,
                    "head_entity_id": relation.head_entity.id if relation.head_entity else None,
                    "tail_entity_id": relation.tail_entity.id if relation.tail_entity else None,
                    "head_entity_name": relation.head_entity.name if relation.head_entity else None,
                    "tail_entity_name": relation.tail_entity.name if relation.tail_entity else None,
                    "relation_type": (
                        relation.relation_type.value
                        if hasattr(relation.relation_type, "value")
                        else str(relation.relation_type)
                    ),
                    "description": relation.description,
                    "confidence": relation.confidence,
                    "properties": relation.properties or {},
                    "text_chunks": (
                        list(relation.text_chunks) if hasattr(relation, "text_chunks") else []
                    ),
                    "created_at": (
                        relation.created_at.isoformat() if hasattr(relation, "created_at") else None
                    ),
                    "updated_at": (
                        relation.updated_at.isoformat() if hasattr(relation, "updated_at") else None
                    ),
                }
            )

        return {
            "status": "success",
            "message": f"Retrieved {len(relations)} relations",
            "data": {
                "relations": relations,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": end_idx < total_count,
                },
                "filters": {
                    "relation_type": relation_type,
                    "entity_id": entity_id,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/clusters", response_model=Dict[str, Any])
async def get_clusters(
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
    cluster_type: Optional[str] = Query(default=None, description="Filter by cluster type"),
    limit: Optional[int] = Query(default=None, description="Limit number of clusters"),
    offset: int = Query(default=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """Get all clusters from the knowledge graph."""
    try:
        # Get project-specific instance
        agraph = await get_agraph_instance(project_name)

        if not agraph.has_knowledge_graph:
            raise HTTPException(
                status_code=404, detail="No knowledge graph found. Please build one first."
            )

        kg = agraph.knowledge_graph
        if not kg:
            raise HTTPException(
                status_code=500, detail="Knowledge graph exists but is not accessible"
            )

        # Get all clusters
        all_clusters = list(kg.clusters.values())

        # Filter by cluster type if provided
        if cluster_type:
            filtered_clusters = []
            for cluster in all_clusters:
                cluster_type_value = (
                    cluster.cluster_type.value
                    if hasattr(cluster, "cluster_type") and hasattr(cluster.cluster_type, "value")
                    else str(getattr(cluster, "cluster_type", ""))
                )
                if cluster_type_value == cluster_type:
                    filtered_clusters.append(cluster)
            all_clusters = filtered_clusters

        total_count = len(all_clusters)

        # Apply pagination
        start_idx = offset
        if limit:
            end_idx = start_idx + limit
            paginated_clusters = all_clusters[start_idx:end_idx]
        else:
            paginated_clusters = all_clusters[start_idx:]
            end_idx = len(all_clusters)

        # Convert to API format
        clusters = []
        for cluster in paginated_clusters:
            clusters.append(
                {
                    "id": cluster.id,
                    "name": cluster.name,
                    "description": cluster.description,
                    "cluster_type": (
                        cluster.cluster_type.value
                        if hasattr(cluster, "cluster_type")
                        and hasattr(cluster.cluster_type, "value")
                        else str(getattr(cluster, "cluster_type", ""))
                    ),
                    "entities": list(cluster.entities) if hasattr(cluster, "entities") else [],
                    "relations": list(cluster.relations) if hasattr(cluster, "relations") else [],
                    "confidence": getattr(cluster, "confidence", 1.0),
                    "properties": getattr(cluster, "properties", {}),
                    "created_at": (
                        cluster.created_at.isoformat() if hasattr(cluster, "created_at") else None
                    ),
                    "updated_at": (
                        cluster.updated_at.isoformat() if hasattr(cluster, "updated_at") else None
                    ),
                }
            )

        return {
            "status": "success",
            "message": f"Retrieved {len(clusters)} clusters",
            "data": {
                "clusters": clusters,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": end_idx < total_count,
                },
                "filters": {
                    "cluster_type": cluster_type,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
