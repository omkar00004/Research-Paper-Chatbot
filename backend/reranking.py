"""
Re-ranking module.
Uses FlashRank to re-score and re-order retrieved chunks for better relevance.
"""
import logging
from typing import List, Dict, Any

from flashrank import Ranker, RerankRequest

from backend.config import TOP_K_RERANK

logger = logging.getLogger(__name__)

# Singleton ranker
_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    """Lazy-load the FlashRank ranker."""
    global _ranker
    if _ranker is None:
        logger.info("Loading FlashRank ranker model...")
        _ranker = Ranker()
        logger.info("FlashRank ranker loaded successfully")
    return _ranker


def rerank(query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks using FlashRank with source diversity.

    After FlashRank scoring, applies a diversity-aware selection to ensure
    chunks from multiple papers are included (not just the single highest-scoring paper).

    Args:
        query: The user's question.
        retrieved_chunks: List of chunks from retrieval.
        top_k: Number of top results to return after re-ranking.

    Returns:
        Re-ranked list of chunks (top_k) with source diversity.
    """
    if not retrieved_chunks:
        return []

    ranker = _get_ranker()

    # Prepare passages for FlashRank
    passages = []
    for chunk in retrieved_chunks:
        passages.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "meta": chunk.get("metadata", {}),
        })

    # Re-rank
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)

    # Build scored chunks list (map back to original format)
    scored_chunks = []
    for result in reranked_results:
        original = None
        for chunk in retrieved_chunks:
            if chunk["id"] == result["id"]:
                original = chunk
                break
        if original:
            scored_chunks.append({
                **original,
                "rerank_score": result["score"],
            })

    # ── Source-diversity selection ───────────────────────
    # Group by source paper, keeping order within each group
    from collections import OrderedDict
    source_groups = OrderedDict()
    for chunk in scored_chunks:
        source = chunk.get("metadata", {}).get("source", "unknown")
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(chunk)

    # Round-robin pick from each source to ensure diversity
    diverse_results = []
    seen_ids = set()
    max_per_source_first_pass = max(2, top_k // max(len(source_groups), 1))

    # First pass: take top chunks from each source (round-robin)
    for round_idx in range(max_per_source_first_pass):
        for source, chunks_list in source_groups.items():
            if round_idx < len(chunks_list) and len(diverse_results) < top_k:
                chunk = chunks_list[round_idx]
                if chunk["id"] not in seen_ids:
                    diverse_results.append(chunk)
                    seen_ids.add(chunk["id"])

    # Second pass: fill remaining slots with best remaining scores
    if len(diverse_results) < top_k:
        for chunk in scored_chunks:
            if chunk["id"] not in seen_ids and len(diverse_results) < top_k:
                diverse_results.append(chunk)
                seen_ids.add(chunk["id"])

    # Log diversity info
    sources_in_result = set(c.get("metadata", {}).get("source", "?") for c in diverse_results)
    logger.info(
        f"Re-ranked {len(retrieved_chunks)} chunks → {len(diverse_results)} results "
        f"from {len(sources_in_result)} papers: {sources_in_result}"
    )

    return diverse_results

