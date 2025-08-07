"""HTML document processor implementation.

This module provides functionality for extracting text and metadata from HTML files.
It supports structured text extraction, link extraction, and comprehensive HTML analysis
while properly handling different encodings and document structures.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

from .base import DocumentProcessor, ProcessingError


class HTMLProcessor(DocumentProcessor):
    """Document processor for HTML files.

    This processor extracts text content from HTML documents while preserving
    meaningful structure. It can optionally extract links and provides both
    plain text and structured text extraction modes.

    Features:
    - Clean text extraction with script/style removal
    - Structured text extraction preserving headings and hierarchy
    - Link extraction with context
    - Metadata extraction from HTML meta tags
    - Multiple encoding support with fallback
    - Element counting and document analysis

    Dependencies:
        beautifulsoup4: Required for HTML parsing and content extraction.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported HTML file extensions.

        Returns:
            List containing '.html' and '.htm' extensions.
        """
        return [".html", ".htm"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text content from HTML files.

        This method parses HTML content and extracts readable text while removing
        scripts, styles, and other non-content elements. It can preserve document
        structure and optionally extract links.

        Args:
            file_path: Path to the HTML file to process.
            **kwargs: Additional processing parameters:
                - encoding (str): Text encoding to use (default: 'utf-8')
                - extract_links (bool): Whether to append extracted links (default: False)
                - preserve_structure (bool): Whether to preserve heading hierarchy
                  and document structure (default: False)

        Returns:
            Extracted text content, optionally with links appended.

        Raises:
            ProcessingError: If beautifulsoup4 is not available, file cannot be
                           decoded, or parsing fails.
        """
        self.validate_file(file_path)

        encoding = kwargs.get("encoding", "utf-8")
        extract_links = kwargs.get("extract_links", False)
        preserve_structure = kwargs.get("preserve_structure", False)

        # Read HTML content with encoding fallback
        html_content = self._read_html_with_encoding_fallback(file_path, encoding)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ProcessingError(
                "beautifulsoup4 is required for HTML processing. Install with: pip install beautifulsoup4"
            )

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script, style, and other non-content elements
            self._remove_non_content_elements(soup)

            # Extract text based on structure preference
            if preserve_structure:
                text = self._extract_structured_text(soup)
            else:
                text = self._extract_plain_text(soup)

            # Optionally extract and append links
            if extract_links:
                links = self._extract_links(soup)
                if links:
                    text += "\n\nExtracted Links:\n" + "\n".join(links)

            return text

        except Exception as e:
            raise ProcessingError(f"Failed to process HTML file {file_path}: {str(e)}")

    def _read_html_with_encoding_fallback(self, file_path: Union[str, Path], preferred_encoding: str) -> str:
        """Read HTML file with encoding detection and fallback.

        Args:
            file_path: Path to the HTML file.
            preferred_encoding: Preferred encoding to try first.

        Returns:
            HTML content as string.

        Raises:
            ProcessingError: If file cannot be decoded with any encoding.
        """
        try:
            with open(file_path, "r", encoding=preferred_encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            # Try alternative encodings common for HTML files
            for alt_encoding in ["latin-1", "cp1252", "iso-8859-1", "utf-16"]:
                try:
                    with open(file_path, "r", encoding=alt_encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ProcessingError(f"Could not decode HTML file {file_path} with any supported encoding")

    def _remove_non_content_elements(self, soup: "BeautifulSoup") -> None:
        """Remove non-content elements from BeautifulSoup object.

        Args:
            soup: BeautifulSoup object to clean.
        """
        # Remove scripts, styles, and other non-content elements
        for element in soup(["script", "style", "meta", "link", "noscript", "svg"]):
            element.decompose()

        # Remove comments
        from bs4 import Comment

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

    def _extract_plain_text(self, soup: "BeautifulSoup") -> str:
        """Extract plain text from HTML with basic cleanup.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Clean plain text.
        """
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
        return text

    def _extract_structured_text(self, soup: "BeautifulSoup") -> str:
        """Extract text while preserving document structure.

        This method maintains headings, paragraphs, lists, and tables
        in a readable format that preserves the document's logical structure.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Structured text content.
        """
        text_parts = []

        # Extract title
        title = soup.find("title")
        if title and title.get_text().strip():
            text_parts.append(f"Title: {title.get_text().strip()}")

        # Extract headings with hierarchy
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            if hasattr(heading, "name") and heading.name:
                level = int(heading.name[1])
                indent = "  " * (level - 1)
                heading_text = heading.get_text().strip()
                if heading_text:
                    text_parts.append(f"{indent}{heading_text}")

        # Extract paragraphs
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if text:
                text_parts.append(text)

        # Extract list items with bullets
        for ul in soup.find_all("ul"):
            if hasattr(ul, "find_all"):
                for li in ul.find_all("li"):
                    text = li.get_text().strip()
                    if text:
                        text_parts.append(f"• {text}")

        # Extract ordered list items with numbers
        for ol in soup.find_all("ol"):
            if hasattr(ol, "find_all"):
                for i, li in enumerate(ol.find_all("li"), 1):
                    text = li.get_text().strip()
                    if text:
                        text_parts.append(f"{i}. {text}")

        # Extract table data
        for table in soup.find_all("table"):
            if hasattr(table, "find_all"):
                text_parts.append("Table:")
                for row in table.find_all("tr"):
                    if hasattr(row, "find_all"):
                        cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                        if any(cells):
                            text_parts.append("  " + " | ".join(cells))

        # Extract any remaining content that wasn't captured
        remaining_content = []
        for element in soup.find_all(text=True):
            if hasattr(element, "parent") and element.parent and hasattr(element.parent, "name"):
                if element.parent.name not in ["title", "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]:
                    text = element.strip() if hasattr(element, "strip") else str(element).strip()
                    if text and len(text) > 3:  # Avoid single characters and very short strings
                        remaining_content.append(text)

        if remaining_content:
            text_parts.extend(remaining_content)

        return "\n\n".join(text_parts)

    def _extract_links(self, soup: "BeautifulSoup") -> List[str]:
        """Extract all external links from the HTML.

        Args:
            soup: BeautifulSoup object.

        Returns:
            List of formatted link strings.
        """
        links = []
        for a in soup.find_all("a", href=True):
            if hasattr(a, "get") and hasattr(a, "get_text"):
                href = a.get("href")
                text = a.get_text().strip()
                if href and isinstance(href, str):
                    # Only include external links
                    if href.startswith(("http://", "https://", "ftp://", "mailto:")):
                        if text:
                            links.append(f"{text}: {href}")
                        else:
                            links.append(href)
        return links

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata from HTML files.

        This method extracts file system metadata, HTML-specific metadata from
        meta tags, and document structure information.

        Args:
            file_path: Path to the HTML file.

        Returns:
            Dictionary containing metadata with keys:
            - file_path: Original file path
            - file_size: File size in bytes
            - file_type: File extension
            - created/modified: File timestamps
            - title: Document title from <title> tag
            - meta_tags: Dictionary of meta tag content
            - language: Document language if specified
            - element_counts: Count of various HTML elements
            - parsing_error: Error message if parsing fails
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        stat = file_path.stat()
        metadata = {
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_type": file_path.suffix.lower(),
            "created": str(stat.st_ctime),
            "modified": str(stat.st_mtime),
        }

        try:
            # Read HTML content
            html_content = self._read_html_with_encoding_fallback(file_path, "utf-8")

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_content, "html.parser")

                # Extract basic HTML metadata
                title = soup.find("title")
                metadata["title"] = title.get_text().strip() if title else ""

                # Extract meta tags
                meta_tags = {}
                for meta in soup.find_all("meta"):
                    if hasattr(meta, "get"):  # Check if it's a Tag object
                        name = meta.get("name") or meta.get("property") or meta.get("http-equiv")
                        content = meta.get("content")
                        if name and content:
                            meta_tags[name] = content

                metadata["meta_tags"] = meta_tags

                # Count various HTML elements for document analysis
                element_counts = {
                    "heading_count": len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
                    "paragraph_count": len(soup.find_all("p")),
                    "link_count": len(soup.find_all("a", href=True)),
                    "image_count": len(soup.find_all("img")),
                    "table_count": len(soup.find_all("table")),
                    "list_count": len(soup.find_all(["ul", "ol"])),
                    "form_count": len(soup.find_all("form")),
                    "div_count": len(soup.find_all("div")),
                }
                metadata.update(element_counts)

                # Extract language information
                html_tag = soup.find("html")
                if html_tag and hasattr(html_tag, "get") and html_tag.get("lang"):
                    metadata["language"] = html_tag.get("lang")

                # Analyze content structure
                self._analyze_content_structure(soup, metadata)

            except ImportError:
                metadata["parsing_error"] = "beautifulsoup4 not available for HTML parsing"
            except Exception as e:
                metadata["parsing_error"] = str(e)

        except Exception as e:
            metadata["content_analysis_error"] = str(e)

        return metadata

    def _analyze_content_structure(self, soup: "BeautifulSoup", metadata: Dict[str, Any]) -> None:
        """Analyze HTML content structure and add to metadata.

        Args:
            soup: BeautifulSoup object.
            metadata: Metadata dictionary to update.
        """
        # Analyze heading structure
        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if headings:
            heading_levels = [int(h.name[1]) for h in headings if hasattr(h, "name") and h.name and len(h.name) > 1]
            if heading_levels:
                metadata["heading_levels"] = list(set(heading_levels))
                metadata["max_heading_level"] = max(heading_levels)
            metadata["min_heading_level"] = min(heading_levels)

        # Check for common page types
        nav = soup.find("nav")
        footer = soup.find("footer")
        header = soup.find("header")
        main = soup.find("main")

        metadata["has_navigation"] = nav is not None
        metadata["has_footer"] = footer is not None
        metadata["has_header"] = header is not None
        metadata["has_main_content"] = main is not None

        # Check for semantic HTML5 elements
        semantic_elements = ["article", "aside", "section", "nav", "header", "footer", "main"]
        semantic_count = sum(len(soup.find_all(element)) for element in semantic_elements)
        metadata["semantic_elements_count"] = semantic_count
        metadata["uses_semantic_html"] = semantic_count > 0
