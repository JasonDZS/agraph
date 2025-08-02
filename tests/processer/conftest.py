"""Test fixtures and utilities for processer tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    text_file = temp_dir / "sample.txt"
    text_file.write_text("This is a sample text file.\nWith multiple lines.\nFor testing purposes.")
    return text_file


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file for testing."""
    md_file = temp_dir / "sample.md"
    md_content = """---
title: Sample Document
author: Test Author
---

# Main Title

This is a **bold** text and *italic* text.

## Subsection

- List item 1
- List item 2

[Link text](https://example.com)

```python
print("Hello, World!")
```

> This is a blockquote
"""
    md_file.write_text(md_content)
    return md_file


@pytest.fixture
def sample_html_file(temp_dir: Path) -> Path:
    """Create a sample HTML file for testing."""
    html_file = temp_dir / "sample.html"
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample HTML</title>
    <meta name="description" content="A sample HTML document">
    <meta name="keywords" content="sample, test, html">
</head>
<body>
    <h1>Main Heading</h1>
    <h2>Subheading</h2>
    <p>This is a paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
    </ul>
    <a href="https://example.com">External link</a>
    <table>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
        <tr>
            <td>Cell 1</td>
            <td>Cell 2</td>
        </tr>
    </table>
    <script>console.log("test");</script>
    <style>body { margin: 0; }</style>
</body>
</html>"""
    html_file.write_text(html_content)
    return html_file


@pytest.fixture
def sample_json_file(temp_dir: Path) -> Path:
    """Create a sample JSON file for testing."""
    json_file = temp_dir / "sample.json"
    json_content = """{
    "name": "Test Document",
    "description": "A sample JSON document for testing",
    "version": "1.0.0",
    "metadata": {
        "author": "Test Author",
        "created": "2023-01-01",
        "tags": ["test", "sample"]
    },
    "content": [
        {
            "type": "text",
            "value": "This is sample text content"
        },
        {
            "type": "number",
            "value": 42
        }
    ]
}"""
    json_file.write_text(json_content)
    return json_file


@pytest.fixture
def sample_jsonl_file(temp_dir: Path) -> Path:
    """Create a sample JSONL file for testing."""
    jsonl_file = temp_dir / "sample.jsonl"
    jsonl_content = """{"id": 1, "text": "First line of text", "category": "A"}
{"id": 2, "text": "Second line of text", "category": "B"}
{"id": 3, "text": "Third line of text", "category": "A"}
invalid_json_line
{"id": 4, "text": "Fourth line of text", "category": "C"}"""
    jsonl_file.write_text(jsonl_content)
    return jsonl_file


@pytest.fixture
def sample_csv_file(temp_dir: Path) -> Path:
    """Create a sample CSV file for testing."""
    csv_file = temp_dir / "sample.csv"
    csv_content = """Name,Age,City,Occupation
John Doe,30,New York,Engineer
Jane Smith,25,London,Designer
Bob Johnson,35,Tokyo,Manager
Alice Brown,28,Paris,Developer"""
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def empty_file(temp_dir: Path) -> Path:
    """Create an empty file for testing."""
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")
    return empty_file


@pytest.fixture
def nonexistent_file(temp_dir: Path) -> Path:
    """Return path to a non-existent file."""
    return temp_dir / "nonexistent.txt"


@pytest.fixture
def binary_file(temp_dir: Path) -> Path:
    """Create a binary file for testing."""
    binary_file = temp_dir / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    return binary_file


# Mock API responses for image processing tests
@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test image showing a sample scene with various objects."
                }
            }
        ]
    }


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response."""
    return {
        "content": [
            {
                "text": "This is a test image analysis from Claude showing detailed description of the scene."
            }
        ]
    }


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "requires_pypdf: mark test as requiring pypdf")
    config.addinivalue_line("markers", "requires_docx: mark test as requiring python-docx")
    config.addinivalue_line("markers", "requires_pandas: mark test as requiring pandas")
    config.addinivalue_line("markers", "requires_beautifulsoup: mark test as requiring beautifulsoup4")
    config.addinivalue_line("markers", "requires_pillow: mark test as requiring Pillow")
    config.addinivalue_line("markers", "requires_openai: mark test as requiring openai")
    config.addinivalue_line("markers", "requires_anthropic: mark test as requiring anthropic")


def skip_if_no_module(module_name: str):
    """Skip test if module is not available."""
    try:
        __import__(module_name)
        return False
    except ImportError:
        return True
