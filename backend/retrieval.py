"""
Retrieval module.
Performs top-k similarity search against ChromaDB.
"""
import logging
from typing import List, Dict, Any

from backend.config import TOP_K_RETRIEVAL
from backend.embedding import embed_query
from backend.indexing import _get_collection

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar chunks for a given query.

    Args:
        query: The user's question.
        top_k: Number of results to retrieve.

    Returns:
        List of dicts with keys: id, text, metadata, distance.
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("Collection is empty. No documents indexed.")
        return []

    # Embed the query
    query_embedding = embed_query(query)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    retrieved = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

    logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:80]}...'")
    return retrieved
