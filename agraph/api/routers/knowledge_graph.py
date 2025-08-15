"""Knowledge graph construction router."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...logger import logger
from ..dependencies import (
    get_agraph_instance,
    get_agraph_instance_dependency,
    get_document_manager,
    get_document_manager_dependency,
)
from ..models import (
    KnowledgeGraphBuildRequest,
    KnowledgeGraphBuildResponse,
    KnowledgeGraphStatusResponse,
    KnowledgeGraphUpdateRequest,
    KnowledgeGraphUpdateResponse,
    ResponseStatus,
)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.post("/build", response_model=KnowledgeGraphBuildResponse)
async def build_knowledge_graph(
    request: KnowledgeGraphBuildRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for isolated knowledge graph"
    ),
    agraph: Any = Depends(get_agraph_instance_dependency),
    doc_manager: Any = Depends(get_document_manager_dependency),
) -> KnowledgeGraphBuildResponse:
    """Build knowledge graph from stored documents or direct texts."""
    try:
        # Use project-specific instances if project_name is provided
        if project_name:
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
                texts_to_process.append(doc["content"])
                document_info.append(
                    {
                        "id": doc["id"],
                        "filename": doc.get("filename", "Unknown"),
                        "content_length": len(doc["content"]),
                    }
                )

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
                texts_to_process.append(doc["content"])
                document_info.append(
                    {
                        "id": doc["id"],
                        "filename": doc.get("filename", "Unknown"),
                        "content_length": len(doc["content"]),
                    }
                )

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
    agraph: Any = Depends(get_agraph_instance_dependency),
    doc_manager: Any = Depends(get_document_manager_dependency),
) -> KnowledgeGraphUpdateResponse:
    """Update existing knowledge graph with additional documents or texts."""
    try:
        # Use project-specific instances if project_name is provided
        if project_name:
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
                    additional_texts.append(doc["content"])
                    document_info.append(
                        {
                            "id": doc["id"],
                            "filename": doc.get("filename", "Unknown"),
                            "content_length": len(doc["content"]),
                        }
                    )
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
                additional_texts.append(doc["content"])
                document_info.append(
                    {
                        "id": doc["id"],
                        "filename": doc.get("filename", "Unknown"),
                        "content_length": len(doc["content"]),
                    }
                )

            logger.info(f"Using all {len(all_documents)} documents from project storage for update")

        if not additional_texts:
            raise HTTPException(
                status_code=400, detail="No additional documents or texts available for update"
            )

        logger.info(
            f"Updating knowledge graph with {len(additional_texts)} additional text sources"
        )

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
    agraph: Any = Depends(get_agraph_instance_dependency),
) -> KnowledgeGraphStatusResponse:
    """Get current knowledge graph status and statistics."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
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
            entity_type = entity.entity_type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        for relation in kg.relations.values():
            relation_type = relation.relation_type.value
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
