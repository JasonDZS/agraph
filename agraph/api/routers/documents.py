"""Document management router."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ...logger import logger
from ...processor.factory import DocumentProcessorManager
from ..dependencies import get_document_manager, get_document_manager_dependency
from ..document_manager import DocumentManager
from ..models import (
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    ResponseStatus,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    metadata: str = "{}",  # JSON string for metadata
    tags: str = "[]",  # JSON string for tags
    project_name: Optional[str] = Query(
        default=None, description="Project name for document upload"
    ),
    doc_manager: DocumentManager = Depends(get_document_manager_dependency),
) -> DocumentUploadResponse:
    """Upload documents to storage."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
            doc_manager = get_document_manager(project_name)

        # Parse metadata and tags
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
            tags_list = json.loads(tags) if tags else []
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON in metadata or tags: {e}"
            ) from e

        uploaded_docs = []

        # Initialize document processor manager
        processor_manager = DocumentProcessorManager()

        for file in files:
            if not file.filename:
                logger.warning("File uploaded without filename")
                continue

            # Read file content
            content = await file.read()

            try:
                # Create a temporary file to process with the document processors
                with tempfile.NamedTemporaryFile(
                    suffix=Path(file.filename).suffix, delete=False
                ) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                try:
                    # Check if the processor can handle this file type
                    if not processor_manager.can_process(temp_file_path):
                        supported_extensions = processor_manager.get_supported_extensions()
                        raise HTTPException(
                            status_code=400,
                            detail=f"File type {Path(file.filename).suffix} is not supported. "
                            f"Supported extensions: {supported_extensions}",
                        )

                    # Extract metadata using the processor for indexing purposes
                    file_metadata = processor_manager.extract_metadata(temp_file_path)

                    # Store document with original binary content preserved
                    doc_id = doc_manager.store_document(
                        content=content,  # Store original binary content as-is
                        filename=file.filename,
                        metadata={
                            **metadata_dict,
                            "content_type": file.content_type,
                            "file_size": len(content),
                            "extracted_metadata": file_metadata,
                            "original_format": True,  # Flag to indicate original format preservation
                        },
                        tags=tags_list,
                    )

                    uploaded_docs.append(
                        {
                            "id": doc_id,
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size": len(content),
                            "content_length": len(content),
                            "extracted_metadata": file_metadata,
                        }
                    )

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except OSError:
                        logger.warning(f"Failed to delete temporary file: {temp_file_path}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Failed to process file {file.filename}: {str(e)}"
                ) from e

        response_data = {
            "uploaded_documents": uploaded_docs,
            "total_uploaded": len(uploaded_docs),
            "metadata": metadata_dict,
            "tags": tags_list,
        }

        return DocumentUploadResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Successfully uploaded {len(uploaded_docs)} documents",
            data=response_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/from-text", response_model=DocumentUploadResponse)
async def upload_texts(
    request: DocumentUploadRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for document upload"
    ),
    doc_manager: DocumentManager = Depends(get_document_manager_dependency),
) -> DocumentUploadResponse:
    """Upload texts directly to storage."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
            doc_manager = get_document_manager(project_name)

        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        uploaded_docs = []

        for i, text in enumerate(request.texts):
            doc_id = doc_manager.store_document(
                content=text,
                filename=f"text_{i+1}.txt",
                metadata=request.metadata or {},
                tags=request.tags or [],
            )

            uploaded_docs.append(
                {"id": doc_id, "filename": f"text_{i+1}.txt", "content_length": len(text)}
            )

        response_data = {
            "uploaded_documents": uploaded_docs,
            "total_uploaded": len(uploaded_docs),
            "metadata": request.metadata,
            "tags": request.tags,
        }

        return DocumentUploadResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Successfully uploaded {len(uploaded_docs)} text documents",
            data=response_data,
        )

    except Exception as e:
        logger.error(f"Failed to upload texts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 10,
    tag_filter: str = "[]",  # JSON array string
    search_query: str = "",
    project_name: Optional[str] = Query(
        default=None, description="Project name for document listing"
    ),
    doc_manager: DocumentManager = Depends(get_document_manager_dependency),
) -> DocumentListResponse:
    """List stored documents with pagination and filtering."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
            doc_manager = get_document_manager(project_name)

        # Parse tag filter
        try:
            tags_filter = json.loads(tag_filter) if tag_filter else []
        except json.JSONDecodeError:
            tags_filter = []

        documents, total_count = doc_manager.list_documents(
            page=page,
            page_size=page_size,
            tag_filter=tags_filter if tags_filter else None,
            search_query=search_query if search_query else None,
        )

        response_data = {
            "documents": documents,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size,
            },
            "filters": {"tag_filter": tags_filter, "search_query": search_query},
        }

        return DocumentListResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Retrieved {len(documents)} documents",
            data=response_data,
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{doc_id}")
async def get_document(
    doc_id: str,
    project_name: Optional[str] = Query(
        default=None, description="Project name for document retrieval"
    ),
    doc_manager: DocumentManager = Depends(get_document_manager_dependency),
) -> Any:
    """Get a specific document by ID."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
            doc_manager = get_document_manager(project_name)

        document = doc_manager.get_document(doc_id)

        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        return {
            "status": ResponseStatus.SUCCESS,
            "message": "Document retrieved successfully",
            "data": document,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/delete", response_model=DocumentDeleteResponse)
async def delete_documents(
    request: DocumentDeleteRequest,
    project_name: Optional[str] = Query(
        default=None, description="Project name for document deletion"
    ),
    doc_manager: DocumentManager = Depends(get_document_manager_dependency),
) -> DocumentDeleteResponse:
    """Delete documents by IDs."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
            doc_manager = get_document_manager(project_name)

        results = doc_manager.delete_documents(request.document_ids)

        successful_deletes = [doc_id for doc_id, success in results.items() if success]
        failed_deletes = [doc_id for doc_id, success in results.items() if not success]

        response_data = {
            "deleted_documents": successful_deletes,
            "failed_deletes": failed_deletes,
            "total_requested": len(request.document_ids),
            "total_deleted": len(successful_deletes),
        }

        message = f"Successfully deleted {len(successful_deletes)} documents"
        if failed_deletes:
            message += f", failed to delete {len(failed_deletes)} documents"

        return DocumentDeleteResponse(
            status=ResponseStatus.SUCCESS, message=message, data=response_data
        )

    except Exception as e:
        logger.error(f"Failed to delete documents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats/summary")
async def get_document_stats(
    project_name: Optional[str] = Query(
        default=None, description="Project name for document stats"
    ),
    doc_manager: DocumentManager = Depends(get_document_manager_dependency),
) -> Any:
    """Get document storage statistics."""
    try:
        # Use project-specific instance if project_name is provided
        if project_name:
            doc_manager = get_document_manager(project_name)

        stats = doc_manager.get_stats()

        return {
            "status": ResponseStatus.SUCCESS,
            "message": "Document statistics retrieved successfully",
            "data": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
