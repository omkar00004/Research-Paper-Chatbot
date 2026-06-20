"""
Hybrid Retrieval module.
Combines dense vector search (ChromaDB) with BM25 sparse keyword search
and merges results using Reciprocal Rank Fusion (RRF) for maximum accuracy
on research papers.
"""
import logging
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from backend.config import TOP_K_DENSE, TOP_K_BM25, RRF_K, TOP_K_RETRIEVAL
from backend.embedding import embed_query
from backend.indexing import _get_collection

logger = logging.getLogger(__name__)


# ── BM25 In-Memory Index ─────────────────────────────

class BM25Index:
    """In-memory BM25 index built from ChromaDB chunks."""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_metadatas: List[dict] = []
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25. Lowercased, alphanumeric tokens."""
        text = text.lower()
        # Keep alphanumeric + hyphens (for terms like "self-attention")
        tokens = re.findall(r'[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?', text)
        return tokens

    def build(self) -> int:
        """
        Build BM25 index from all documents in ChromaDB.
        Returns the number of documents indexed.
        """
        collection = _get_collection()
        count = collection.count()

        if count == 0:
            logger.warning("ChromaDB collection is empty. BM25 index not built.")
            self._built = False
            return 0

        logger.info(f"Building BM25 index from {count} ChromaDB documents...")

        # Fetch all documents from ChromaDB
        results = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        self._doc_ids = results["ids"]
        self._doc_texts = results["documents"]
        self._doc_metadatas = results["metadatas"]

        # Tokenize all documents
        tokenized_corpus = [self._tokenize(text) for text in self._doc_texts]

        # Build BM25
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._built = True

        logger.info(f"BM25 index built successfully ({count} documents)")
        return count

    def refresh(self) -> int:
        """Rebuild the BM25 index (call after new ingestion)."""
        logger.info("Refreshing BM25 index...")
        return self.build()

    def search(self, query: str, top_k: int = TOP_K_BM25) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for the given query.

        Returns list of dicts with: id, text, metadata, bm25_score
        """
        if not self._built or self._bm25 is None:
            logger.warning("BM25 index not built. Building now...")
            self.build()
            if not self._built:
                return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        scored_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for idx, score in scored_indices:
            if score > 0:  # Only include documents with non-zero BM25 score
                results.append({
                    "id": self._doc_ids[idx],
                    "text": self._doc_texts[idx],
                    "metadata": self._doc_metadatas[idx] if self._doc_metadatas[idx] else {},
                    "bm25_score": float(score),
                })

        logger.info(f"BM25 search returned {len(results)} results for query: '{query[:60]}...'")
        return results


# Singleton BM25 index
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Get or create the singleton BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


# ── Dense Retrieval ───────────────────────────────────

def _dense_retrieve(query: str, top_k: int = TOP_K_DENSE) -> List[Dict[str, Any]]:
    """Retrieve chunks using dense vector similarity (ChromaDB)."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

    logger.info(f"Dense retrieval returned {len(retrieved)} results")
    return retrieved


# ── Reciprocal Rank Fusion ────────────────────────────

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    RRF(d) = Σ 1/(k + rank_i(d))

    where rank_i(d) is the rank of document d in list i (1-indexed).
    Documents appearing in both lists get boosted scores.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Score from dense results
    for rank, doc in enumerate(dense_results, 1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    # Score from BM25 results
    for rank, doc in enumerate(sparse_results, 1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        fused_results.append(doc)

    logger.info(
        f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(fused_results)} unique results"
    )
    return fused_results


# ── Hybrid Retrieval (main entry point) ───────────────

def hybrid_retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    dense_k: int = TOP_K_DENSE,
    bm25_k: int = TOP_K_BM25,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid retrieval combining dense + BM25 with RRF.

    Args:
        query: User's question.
        top_k: Number of final results to return after fusion.
        dense_k: Number of results from dense retrieval.
        bm25_k: Number of results from BM25 retrieval.

    Returns:
        Fused and sorted list of chunks.
    """
    # 1. Dense retrieval
    dense_results = _dense_retrieve(query, top_k=dense_k)

    # 2. BM25 retrieval
    bm25_index = get_bm25_index()
    if not bm25_index.is_built:
        bm25_index.build()

    sparse_results = bm25_index.search(query, top_k=bm25_k)

    # 3. If one method returns nothing, fall back to the other
    if not dense_results and not sparse_results:
        return []
    if not sparse_results:
        logger.info("BM25 returned no results, using dense-only")
        return dense_results[:top_k]
    if not dense_results:
        logger.info("Dense returned no results, using BM25-only")
        return sparse_results[:top_k]

    # 4. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    logger.info(f"Hybrid retrieval: returning top {min(top_k, len(fused))} of {len(fused)} fused results")
    return fused[:top_k]
