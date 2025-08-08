"""
Text chunking utilities for splitting documents into manageable pieces.

This module provides various text chunking strategies including token-based chunking.
"""

import re
from typing import Any, List, Optional

import tiktoken


class TokenChunker:
    """
    A text chunker that splits text based on token count.

    This chunker uses tiktoken to calculate token counts and splits text into
    chunks that don't exceed the specified token limit.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, model: str = "gpt-3.5-turbo"):
        """
        Initialize the TokenChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            model: The model name to use for token counting (default: gpt-3.5-turbo)

        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If chunk_overlap is greater than or equal to chunk_size
        """
        if tiktoken is None:
            raise ImportError("tiktoken is required for token-based chunking. Install it with: pip install tiktoken")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.encoding: Any

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding if model is not found
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        return len(self.encoding.encode(text))

    def split_text(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Split text into chunks based on token count.

        Args:
            text: The text to split into chunks
            separators: List of separators to use for splitting (default: paragraph, sentence, word)

        Returns:
            List of text chunks, each with token count <= chunk_size
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

        chunks = []
        current_chunk = ""

        # First, try to split by separators in order of preference
        for separator in separators:
            if separator == "":
                # Last resort: split character by character
                chunks.extend(self._split_by_character(text))
                break

            parts = text.split(separator)
            if len(parts) <= 1:
                continue

            for i, part in enumerate(parts):
                # Add separator back except for the last part
                if i < len(parts) - 1 and separator:
                    part += separator

                # Check if adding this part would exceed chunk size
                test_chunk = current_chunk + part
                if self.count_tokens(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk if it's not empty
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    # If this part alone exceeds chunk size, split it further
                    if self.count_tokens(part) > self.chunk_size:
                        # Recursively split this part with remaining separators
                        sub_chunks = self._split_text_recursive(part, separators[separators.index(separator) + 1 :])
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part

            # Add the last chunk if it exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            break

        # Handle overlaps
        if self.chunk_overlap > 0:
            chunks = self._add_overlaps(chunks)

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the remaining separators.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text chunks
        """
        if not separators:
            return self._split_by_character(text)

        separator = separators[0]
        if separator == "":
            return self._split_by_character(text)

        parts = text.split(separator)
        if len(parts) <= 1:
            return self._split_text_recursive(text, separators[1:])

        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            if i < len(parts) - 1 and separator:
                part += separator

            test_chunk = current_chunk + part
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                if self.count_tokens(part) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(part, separators[1:])
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_character(self, text: str) -> List[str]:
        """
        Split text character by character as a last resort.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""

        for char in text:
            test_chunk = current_chunk + char
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = char

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _add_overlaps(self, chunks: List[str]) -> List[str]:
        """
        Add overlaps between chunks.

        Args:
            chunks: List of chunks to add overlaps to

        Returns:
            List of chunks with overlaps
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap = self._get_overlap_text(prev_chunk, self.chunk_overlap)

            # Combine overlap with current chunk
            combined = overlap + chunk

            # Make sure combined chunk doesn't exceed size limit
            if self.count_tokens(combined) <= self.chunk_size + self.chunk_overlap:
                overlapped_chunks.append(combined)
            else:
                overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens worth of text for overlap.

        Args:
            text: Source text
            overlap_tokens: Number of tokens to get for overlap

        Returns:
            Text containing approximately overlap_tokens tokens
        """
        if overlap_tokens == 0:
            return ""

        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text

        # Take the last overlap_tokens tokens
        overlap_token_slice = tokens[-overlap_tokens:]
        return str(self.encoding.decode(overlap_token_slice))

    def chunk_documents(self, documents: List[str]) -> List[dict]:
        """
        Chunk multiple documents and return results with metadata.

        Args:
            documents: List of document texts to chunk

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        results = []

        for doc_idx, document in enumerate(documents):
            chunks = self.split_text(document)

            for chunk_idx, chunk in enumerate(chunks):
                results.append(
                    {
                        "text": chunk,
                        "document_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "token_count": self.count_tokens(chunk),
                        "char_count": len(chunk),
                    }
                )

        return results


class SimpleTokenChunker:
    """
    A simplified token chunker that doesn't require tiktoken.

    Uses approximate token counting based on word count (1 token H 0.75 words).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the SimpleTokenChunker.

        Args:
            chunk_size: Maximum number of approximate tokens per chunk
            chunk_overlap: Number of approximate tokens to overlap between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count based on word count.

        Args:
            text: The text to count tokens for

        Returns:
            Approximate number of tokens
        """
        # Rough approximation: 1 token H 0.75 words
        words = len(re.findall(r"\b\w+\b", text))
        return int(words / 0.75)

    def split_text(self, text: str) -> List[str]:
        """
        Split text into approximate token-based chunks.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        # Use similar logic to TokenChunker but with approximate counting
        separators = ["\n\n", "\n", ". ", "! ", "? ", " "]

        chunks = []
        current_chunk = ""

        for separator in separators:
            parts = text.split(separator)
            if len(parts) <= 1:
                continue

            for i, part in enumerate(parts):
                if i < len(parts) - 1 and separator:
                    part += separator

                test_chunk = current_chunk + part
                if self.count_tokens(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = part

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            break

        return [chunk for chunk in chunks if chunk.strip()]
