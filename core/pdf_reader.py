"""
pdf_reader.py - PDF text extraction using PyMuPDF (fitz).

Provides page-level text extraction and document metadata retrieval.
Only text-based PDFs are supported; scanned or image-only documents
will return empty page text without raising.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required: pip install pymupdf"
    ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF file.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of page dicts, each containing:
            page_number (int) - 1-based page index
            text        (str) - cleaned extracted text
            char_count  (int) - character length of text

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the file cannot be opened as a PDF.
    """
    _assert_file_exists(pdf_path)
    pages: List[Dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(
            f"Cannot open PDF '{pdf_path}': {exc}"
        ) from exc

    with doc:
        for i, page in enumerate(doc, start=1):
            raw_text: str = page.get_text("text") or ""
            cleaned = _clean_text(raw_text)
            pages.append(
                {
                    "page_number": i,
                    "text":        cleaned,
                    "char_count":  len(cleaned),
                }
            )

    return pages


def get_document_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Return high-level metadata for a PDF document.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Dict containing:
            file_name  (str)
            page_count (int)
            title      (str)
            author     (str)
            subject    (str)

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the file cannot be opened as a PDF.
    """
    _assert_file_exists(pdf_path)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(
            f"Cannot open PDF '{pdf_path}': {exc}"
        ) from exc

    with doc:
        meta = doc.metadata or {}
        return {
            "file_name":  os.path.basename(pdf_path),
            "page_count": doc.page_count,
            "title":      meta.get("title",  "Unknown") or "Unknown",
            "author":     meta.get("author", "Unknown") or "Unknown",
            "subject":    meta.get("subject", "")        or "",
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_file_exists(path: str) -> None:
    """Raise FileNotFoundError if the given path does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")


def _clean_text(text: str) -> str:
    """
    Normalise whitespace while preserving paragraph structure.

    Args:
        text: Raw text string extracted from a PDF page.

    Returns:
        Cleaned text with collapsed inline whitespace and trimmed runs
        of blank lines.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()