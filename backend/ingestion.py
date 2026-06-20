"""
PDF ingestion module.
Extracts text and metadata (page numbers, section headers) from research paper PDFs.
"""
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Regex patterns for detecting section headers
SECTION_PATTERNS = [
    re.compile(r"^(\d+\.?\d*\.?\d*)\s+([A-Z].*)", re.MULTILINE),           # "1. Introduction", "2.1 Background"
    re.compile(r"^(Abstract|Introduction|Conclusion|References|Acknowledgements|Related Work|Methodology|Methods|Results|Discussion|Experiments|Appendix)", re.MULTILINE | re.IGNORECASE),
]


@dataclass
class Document:
    """Represents extracted content from a single page of a PDF."""
    text: str
    metadata: dict = field(default_factory=dict)


def _detect_section(text: str) -> str:
    """Try to detect the most recent section header in the text."""
    sections_found = []
    for pattern in SECTION_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            for m in matches:
                if isinstance(m, tuple):
                    sections_found.append(" ".join(m).strip())
                else:
                    sections_found.append(m.strip())
    return sections_found[-1] if sections_found else "Unknown"


def ingest_pdf(pdf_path: Path) -> List[Document]:
    """
    Extract text and metadata from a PDF file.

    Returns a list of Document objects, one per page.
    """
    documents = []
    filename = pdf_path.name

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        logger.info(f"Ingesting '{filename}' ({total_pages} pages)")

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            if not text or not text.strip():
                logger.debug(f"  Skipping empty page {page_num + 1} in '{filename}'")
                continue

            section = _detect_section(text)

            documents.append(Document(
                text=text.strip(),
                metadata={
                    "source": filename,
                    "page": page_num + 1,
                    "total_pages": total_pages,
                    "section": section,
                }
            ))

        doc.close()
        logger.info(f"  Extracted {len(documents)} non-empty pages from '{filename}'")

    except Exception as e:
        logger.error(f"Failed to ingest '{filename}': {e}")

    return documents


def ingest_directory(directory: Path) -> List[Document]:
    """Ingest all PDF files in a directory."""
    all_docs = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return all_docs

    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

    for pdf_path in pdf_files:
        docs = ingest_pdf(pdf_path)
        all_docs.extend(docs)

    logger.info(f"Total documents extracted: {len(all_docs)}")
    return all_docs
