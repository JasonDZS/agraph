"""Document storage and management for AGraph API."""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _get_logger() -> Any:
    """Get logger instance with lazy import to avoid circular imports."""
    from ..logger import logger  # pylint: disable=import-outside-toplevel

    return logger


class DocumentManager:
    """Simple document storage and management system."""

    def __init__(self, storage_dir: str = "./document_storage", project_name: Optional[str] = None):
        """Initialize document manager."""
        self.storage_dir = Path(storage_dir)
        self.project_name = project_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.documents_dir = self.storage_dir / "documents"
        self.metadata_dir = self.storage_dir / "metadata"
        self.index_file = self.storage_dir / "index.json"

        self.documents_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Load or create index
        self._load_index()

    def _load_index(self) -> None:
        """Load document index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except Exception as e:
                _get_logger().warning(f"Failed to load document index: {e}")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self) -> None:
        """Save document index to file."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _get_logger().error(f"Failed to save document index: {e}")

    def _generate_document_id(self, content: Any, filename: Optional[str] = None) -> str:
        """Generate unique document ID based on content."""
        if isinstance(content, bytes):
            content_hash = hashlib.md5(content).hexdigest()
        else:
            content_hash = hashlib.md5(str(content).encode("utf-8")).hexdigest()
        timestamp = str(int(time.time()))
        if filename:
            filename_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
            return f"doc_{timestamp}_{filename_hash}_{content_hash[:8]}"
        return f"doc_{timestamp}_{content_hash[:8]}"

    def store_document(
        self,
        content: Union[str, bytes],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store a document and return its ID."""
        doc_id = self._generate_document_id(content, filename)

        # Determine if this is binary content and file extension
        is_binary = isinstance(content, bytes)
        if is_binary and filename:
            # Keep original file extension for binary files
            file_extension = Path(filename).suffix
        else:
            file_extension = ".txt"

        # Store document content
        doc_file = self.documents_dir / f"{doc_id}{file_extension}"
        if is_binary:
            with open(doc_file, "wb") as f:
                f.write(content)  # type: ignore[arg-type]
        else:
            with open(doc_file, "w", encoding="utf-8") as f:
                f.write(content)  # type: ignore[arg-type]

        # Store metadata
        if is_binary:
            content_hash = hashlib.md5(content).hexdigest()  # type: ignore[arg-type]
        else:
            content_hash = hashlib.md5(str(content).encode("utf-8")).hexdigest()

        doc_metadata = {
            "id": doc_id,
            "filename": filename,
            "content_length": len(content),
            "stored_at": datetime.now().isoformat(),
            "project_name": self.project_name,
            "metadata": metadata or {},
            "tags": tags or [],
            "content_hash": content_hash,
            "is_binary": is_binary,
            "file_extension": file_extension,
        }

        metadata_file = self.metadata_dir / f"{doc_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(doc_metadata, f, indent=2, ensure_ascii=False)

        # Update index
        self.index[doc_id] = {
            "filename": filename,
            "stored_at": doc_metadata["stored_at"],
            "content_length": len(content),
            "project_name": self.project_name,
            "tags": tags or [],
            "metadata": metadata or {},
            "is_binary": is_binary,
            "file_extension": file_extension,
        }
        self._save_index()

        _get_logger().info(f"Document stored with ID: {doc_id} (binary: {is_binary})")
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document content and metadata by ID."""
        if doc_id not in self.index:
            return None

        try:
            # Load metadata first to determine file type
            metadata_file = self.metadata_dir / f"{doc_id}.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = self.index[doc_id]

            # Determine file extension and binary status
            is_binary = metadata.get("is_binary", False)
            file_extension = metadata.get("file_extension", ".txt")

            # Load content
            doc_file = self.documents_dir / f"{doc_id}{file_extension}"
            if not doc_file.exists():
                # Fallback to .txt for old documents
                doc_file = self.documents_dir / f"{doc_id}.txt"
                if not doc_file.exists():
                    _get_logger().warning(f"Document file not found: {doc_file}")
                    return None

            if is_binary:
                with open(doc_file, "rb") as f:
                    content: Union[str, bytes] = f.read()
            else:
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()

            return {"id": doc_id, "content": content, **metadata}

        except Exception as e:
            _get_logger().error(f"Failed to load document {doc_id}: {e}")
            return None

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs."""
        documents = []
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc:
                documents.append(doc)
            else:
                _get_logger().warning(f"Document not found: {doc_id}")
        return documents

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 10,
        tag_filter: Optional[List[str]] = None,
        search_query: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List documents with pagination and filtering."""
        all_docs = []

        for doc_id, index_data in self.index.items():
            # Apply tag filter
            if tag_filter:
                if not any(tag in index_data.get("tags", []) for tag in tag_filter):
                    continue

            # Apply search filter
            if search_query:
                search_lower = search_query.lower()
                filename = index_data.get("filename", "").lower()
                tags_str = " ".join(index_data.get("tags", [])).lower()

                if search_lower not in filename and search_lower not in tags_str:
                    continue

            doc_info = {"id": doc_id, **index_data}
            all_docs.append(doc_info)

        # Sort by stored_at (newest first)
        all_docs.sort(key=lambda x: x.get("stored_at", ""), reverse=True)

        # Pagination
        total_count = len(all_docs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = all_docs[start_idx:end_idx]

        return paginated_docs, total_count

    def delete_documents(self, doc_ids: List[str]) -> Dict[str, bool]:
        """Delete documents by IDs."""
        results = {}

        for doc_id in doc_ids:
            try:
                if doc_id not in self.index:
                    results[doc_id] = False
                    continue

                # Get file extension from index or metadata
                index_data = self.index[doc_id]
                file_extension = index_data.get("file_extension", ".txt")

                # Delete files
                doc_file = self.documents_dir / f"{doc_id}{file_extension}"
                # Also try .txt for backward compatibility
                doc_file_txt = self.documents_dir / f"{doc_id}.txt"
                metadata_file = self.metadata_dir / f"{doc_id}.json"

                if doc_file.exists():
                    doc_file.unlink()
                elif doc_file_txt.exists():
                    doc_file_txt.unlink()

                if metadata_file.exists():
                    metadata_file.unlink()

                # Remove from index
                del self.index[doc_id]
                results[doc_id] = True

                _get_logger().info(f"Document deleted: {doc_id}")

            except Exception as e:
                _get_logger().error(f"Failed to delete document {doc_id}: {e}")
                results[doc_id] = False

        self._save_index()
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_docs = len(self.index)
        total_size = 0
        tags_count: Dict[str, int] = {}

        for doc_info in self.index.values():
            total_size += doc_info.get("content_length", 0)
            for tag in doc_info.get("tags", []):
                tags_count[tag] = tags_count.get(tag, 0) + 1

        return {
            "total_documents": total_docs,
            "total_content_size": total_size,
            "storage_path": str(self.storage_dir),
            "project_name": self.project_name,
            "tags": tags_count,
        }
