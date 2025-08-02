from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

from .base import DocumentProcessor, ProcessingError


class HTMLProcessor(DocumentProcessor):
    """Processor for HTML documents."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".html", ".htm"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from HTML file.

        Args:
            file_path: Path to the HTML file
            **kwargs: Additional parameters (extract_links, preserve_structure, etc.)

        Returns:
            Extracted text content
        """
        self.validate_file(file_path)

        encoding = kwargs.get("encoding", "utf-8")
        extract_links = kwargs.get("extract_links", False)
        preserve_structure = kwargs.get("preserve_structure", False)

        try:
            with open(file_path, "r", encoding=encoding) as file:
                html_content = file.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=alt_encoding) as file:
                        html_content = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ProcessingError(f"Could not decode HTML file {file_path} with any supported encoding")

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ProcessingError(
                "beautifulsoup4 is required for HTML processing. Install with: pip install beautifulsoup4"
            )

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()

            if preserve_structure:
                text = self._extract_structured_text(soup)
            else:
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                text = "\n".join(line for line in lines if line)

            # Optionally extract and append links
            if extract_links:
                links = self._extract_links(soup)
                if links:
                    text += "\n\nExtracted Links:\n" + "\n".join(links)

            return text

        except Exception as e:
            raise ProcessingError(f"Failed to process HTML file {file_path}: {str(e)}")

    def _extract_structured_text(self, soup: "BeautifulSoup") -> str:
        """Extract text while preserving some structure."""
        text_parts = []

        # Extract title
        title = soup.find("title")
        if title:
            text_parts.append(f"Title: {title.get_text().strip()}")

        # Extract headings with hierarchy
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            if hasattr(heading, "name") and heading.name:
                level = int(heading.name[1])
                indent = "  " * (level - 1)
                text_parts.append(f"{indent}{heading.get_text().strip()}")

        # Extract paragraphs
        for p in soup.find_all("p"):
            text = p.get_text().strip()
            if text:
                text_parts.append(text)

        # Extract list items
        for li in soup.find_all("li"):
            text = li.get_text().strip()
            if text:
                text_parts.append(f"• {text}")

        # Extract table data
        for table in soup.find_all("table"):
            if hasattr(table, "find_all"):
                text_parts.append("Table:")
                for row in table.find_all("tr"):
                    if hasattr(row, "find_all"):
                        cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                        if any(cells):
                            text_parts.append("  " + " | ".join(cells))

        return "\n\n".join(text_parts)

    def _extract_links(self, soup: "BeautifulSoup") -> List[str]:
        """Extract all links from the HTML."""
        links = []
        for a in soup.find_all("a", href=True):
            if hasattr(a, "get") and hasattr(a, "get_text"):
                href = a.get("href")
                text = a.get_text().strip()
                if href and isinstance(href, str) and href.startswith(("http", "https", "ftp")):
                    if text:
                        links.append(f"{text}: {href}")
                    else:
                        links.append(href)
        return links

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from HTML file.

        Args:
            file_path: Path to the HTML file

        Returns:
            Dictionary containing metadata
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
            with open(file_path, "r", encoding="utf-8") as file:
                html_content = file.read()

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

                # Count elements
                metadata.update(
                    {
                        "heading_count": len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
                        "paragraph_count": len(soup.find_all("p")),
                        "link_count": len(soup.find_all("a", href=True)),
                        "image_count": len(soup.find_all("img")),
                        "table_count": len(soup.find_all("table")),
                        "list_count": len(soup.find_all(["ul", "ol"])),
                    }
                )

                # Extract language
                html_tag = soup.find("html")
                if html_tag and hasattr(html_tag, "get") and html_tag.get("lang"):
                    metadata["language"] = html_tag.get("lang")

            except ImportError:
                metadata["parsing_error"] = "beautifulsoup4 not available"
            except Exception as e:
                metadata["parsing_error"] = str(e)

        except Exception as e:
            metadata["content_analysis_error"] = str(e)

        return metadata
