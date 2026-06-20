"""
Indexing module.
Manages ChromaDB vector store for document chunks.
"""
import logging
from typing import List, Optional, Set

import chromadb

from backend.config import CHROMA_DIR, COLLECTION_NAME, DISTANCE_METRIC
from backend.ingestion import Document
from backend.embedding import embed_texts

logger = logging.getLogger(__name__)

# Singleton ChromaDB client and collection
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        logger.info(f"Initializing ChromaDB at {CHROMA_DIR}")
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )
        logger.info(f"Collection '{COLLECTION_NAME}' ready ({_collection.count()} existing documents)")
    return _collection


def get_indexed_sources() -> Set[str]:
    """Get set of already-indexed source filenames."""
    collection = _get_collection()
    if collection.count() == 0:
        return set()

    # Get all metadata to find unique sources
    results = collection.get(include=["metadatas"])
    sources = set()
    if results and results["metadatas"]:
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
    return sources


def index_chunks(chunks: List[Document]) -> int:
    """
    Index document chunks into ChromaDB.
    Skips chunks that are already indexed (by chunk_id).

    Returns the number of newly indexed chunks.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    # Prepare data
    ids = [c.metadata["chunk_id"] for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {k: str(v) for k, v in c.metadata.items()}  # ChromaDB needs string values
        meta["text"] = c.text[:500]  # Store truncated text for retrieval display
        metadatas.append(meta)

    # Check which IDs already exist
    existing = set()
    try:
        existing_results = collection.get(ids=ids)
        if existing_results and existing_results["ids"]:
            existing = set(existing_results["ids"])
    except Exception:
        pass

    # Filter out already indexed
    new_ids = []
    new_texts = []
    new_metadatas = []
    for i, cid in enumerate(ids):
        if cid not in existing:
            new_ids.append(cid)
            new_texts.append(texts[i])
            new_metadatas.append(metadatas[i])

    if not new_ids:
        logger.info("All chunks already indexed, skipping.")
        return 0

    # Generate embeddings for new chunks
    logger.info(f"Generating embeddings for {len(new_ids)} new chunks...")
    embeddings = embed_texts(new_texts)

    # Add to collection in batches
    batch_size = 500
    for i in range(0, len(new_ids), batch_size):
        end = min(i + batch_size, len(new_ids))
        collection.add(
            ids=new_ids[i:end],
            embeddings=embeddings[i:end],
            documents=new_texts[i:end],
            metadatas=new_metadatas[i:end],
        )

    logger.info(f"Indexed {len(new_ids)} new chunks (total: {collection.count()})")
    return len(new_ids)


def get_collection_stats() -> dict:
    """Get statistics about the indexed collection."""
    collection = _get_collection()
    count = collection.count()

    sources = get_indexed_sources()

    return {
        "total_chunks": count,
        "total_papers": len(sources),
        "papers": sorted(list(sources)),
    }


def delete_paper(source_name: str) -> int:
    """Delete all chunks for a given paper from the index."""
    collection = _get_collection()

    # Find all chunk IDs for this source
    results = collection.get(
        where={"source": source_name},
        include=["metadatas"],
    )

    if not results or not results["ids"]:
        return 0

    ids_to_delete = results["ids"]
    collection.delete(ids=ids_to_delete)
    logger.info(f"Deleted {len(ids_to_delete)} chunks for '{source_name}'")
    return len(ids_to_delete)


def clear_index() -> bool:
    """Delete and recreate the entire collection (reset index)."""
    global _client, _collection
    try:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        _get_collection()  # Recreate
        logger.info("Vector index reset successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to reset index: {e}")
        return False
