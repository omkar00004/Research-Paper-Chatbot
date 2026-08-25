"""
Semantic Cache module.
Caches query embeddings + responses in SQLite. On cosine similarity ≥ threshold,
returns the cached response and skips the LLM call.

Uses the same all-MiniLM-L6-v2 embeddings already loaded by the pipeline.
"""
import json
import logging
import pickle
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np

from backend.config import CACHE_DB, CACHE_SIMILARITY_THRESHOLD
from backend.embedding import embed_query

logger = logging.getLogger(__name__)


def _get_connection():
    """Get SQLite connection for the cache database."""
    import sqlite3
    conn = sqlite3.connect(str(CACHE_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    # Create table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_embedding BLOB NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def check_cache(query: str) -> Optional[Dict[str, Any]]:
    """
    Check if a semantically similar query exists in the cache.

    Args:
        query: The incoming user query.

    Returns:
        Cached response dict if similarity ≥ threshold, else None.
    """
    start_time = time.time()

    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, query, query_embedding, response FROM semantic_cache"
        ).fetchall()

        if not rows:
            return None

        # Embed the incoming query
        query_embedding = np.array(embed_query(query), dtype=np.float32)

        best_similarity = -1.0
        best_response = None
        best_cached_query = None

        for row in rows:
            cached_embedding = pickle.loads(row[2])
            # Cosine similarity via dot product (embeddings are L2-normalized)
            similarity = float(np.dot(query_embedding, cached_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_response = row[3]
                best_cached_query = row[1]

        lookup_time = time.time() - start_time

        if best_similarity >= CACHE_SIMILARITY_THRESHOLD:
            logger.info(
                f"Cache HIT (similarity={best_similarity:.4f}, "
                f"threshold={CACHE_SIMILARITY_THRESHOLD}) for query: '{query[:60]}...' "
                f"matched: '{best_cached_query[:60]}...' in {lookup_time:.3f}s"
            )
            response = json.loads(best_response)
            response["cache_hit"] = True
            response["cache_similarity"] = round(best_similarity, 4)
            response["cache_lookup_time"] = round(lookup_time, 4)
            return response

        logger.debug(
            f"Cache MISS (best_similarity={best_similarity:.4f}, "
            f"threshold={CACHE_SIMILARITY_THRESHOLD}) for query: '{query[:60]}...'"
        )
        return None

    except Exception as e:
        logger.warning(f"Semantic cache lookup failed: {e}")
        return None
    finally:
        conn.close()


def store_cache(query: str, response: Dict[str, Any]) -> None:
    """
    Store a query embedding + response in the cache.

    Args:
        query: The user query.
        response: The full response dict from the pipeline.
    """
    try:
        answer = response.get("answer", "")
        # Don't cache error responses or empty answers
        if not answer or "An error occurred" in answer or "Rate limit reached" in answer:
            logger.debug("Skipping caching of error or empty response")
            return

        query_embedding = np.array(embed_query(query), dtype=np.float32)
        embedding_blob = pickle.dumps(query_embedding)

        # Store response without cache metadata
        response_to_store = {k: v for k, v in response.items()
                            if k not in ("cache_hit", "cache_similarity", "cache_lookup_time")}

        conn = _get_connection()
        try:
            conn.execute(
                "INSERT INTO semantic_cache (query, query_embedding, response, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    query,
                    embedding_blob,
                    json.dumps(response_to_store, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            logger.info(f"Cached response for query: '{query[:60]}...'")
        finally:
            conn.close()

    except Exception as e:
        logger.warning(f"Failed to store in semantic cache: {e}")


def clear_cache() -> int:
    """
    Clear all entries from the semantic cache.

    Returns:
        Number of entries deleted.
    """
    conn = _get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        conn.execute("DELETE FROM semantic_cache")
        conn.commit()
        logger.info(f"Cleared {count} entries from semantic cache")
        return count
    finally:
        conn.close()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the semantic cache."""
    conn = _get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        return {"total_entries": count, "threshold": CACHE_SIMILARITY_THRESHOLD}
    finally:
        conn.close()
