"""
Langfuse observability module (v3 SDK).
Uses @observe decorator for automatic trace creation and context propagation.
"""
import logging
import os
from backend.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

logger = logging.getLogger(__name__)


def configure_langfuse():
    """Set Langfuse env vars so the SDK auto-configures. Must be called before any @observe."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY or "xxxx" in LANGFUSE_PUBLIC_KEY:
        logger.info("Langfuse not configured (missing keys). Tracing disabled.")
        return False

    # Langfuse v3 reads these env vars automatically
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

    logger.info(f"Langfuse tracing configured → {LANGFUSE_HOST}")
    return True


def flush():
    """Flush any pending Langfuse events."""
    try:
        from langfuse import get_client
        client = get_client()
        if client:
            client.flush()
    except Exception:
        pass


# Configure on import
_tracing_enabled = configure_langfuse()


def is_enabled() -> bool:
    return _tracing_enabled
