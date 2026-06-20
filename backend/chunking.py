"""
Semantic chunking module.
Splits documents into overlapping chunks using RecursiveCharacterTextSplitter
with sentence-boundary-aware splitting for improved retrieval quality.
"""
import logging
import re
import hashlib
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.ingestion import Document

logger = logging.getLogger(__name__)


def _generate_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Generate a deterministic unique chunk ID."""
    raw = f"{source}::page{page}::chunk{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _clean_text(text: str) -> str:
    """Clean extracted PDF text for better chunking quality."""
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)
    # Fix broken hyphenation from PDF line breaks (e.g. "trans-\nformer" -> "transformer")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Collapse multiple newlines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into semantic chunks with overlapping context.

    Uses sentence-boundary-aware splitting to avoid cutting mid-sentence.
    Each chunk inherits the parent document's metadata and gets
    a unique chunk_id.
    """
    # Sentence-boundary-aware separators (prefer splitting at sentence ends)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",       # Paragraph breaks (strongest boundary)
            "\n",          # Line breaks
            ". ",          # Sentence endings
            "? ",          # Question endings
            "! ",          # Exclamation endings
            "; ",          # Semicolons
            ", ",          # Commas
            " ",           # Words
            "",            # Characters (last resort)
        ],
    )

    chunks = []
    page_chunk_counters = {}  # track per-page chunk index

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        key = f"{source}::page{page}"

        # Clean the text before splitting
        cleaned_text = _clean_text(doc.text)
        splits = splitter.split_text(cleaned_text)

        for split_text in splits:
            if not split_text.strip():
                continue

            chunk_index = page_chunk_counters.get(key, 0)
            page_chunk_counters[key] = chunk_index + 1

            chunk_id = _generate_chunk_id(source, page, chunk_index)

            chunks.append(Document(
                text=split_text.strip(),
                metadata={
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            ))

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents "
                f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
