"""
Embedding module.
Generates dense semantic embeddings using sentence-transformers all-MiniLM-L6-v2.
"""
import logging
from typing import List

from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Singleton model instance
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors (list of floats).
    """
    model = _get_model()
    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = _get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
