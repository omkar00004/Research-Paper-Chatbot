"""
Central configuration for the RAG pipeline.
Loads credentials from .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Paths ──────────────────────────────────────────────
PAPERS_DIR = BASE_DIR / "Research Paper"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_DIR = BASE_DIR / "uploads"
CONVERSATIONS_DB = BASE_DIR / "conversations.db"

# Ensure directories exist
CHROMA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Embedding Model ───────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# ── ChromaDB ──────────────────────────────────────────
COLLECTION_NAME = "research_papers"
DISTANCE_METRIC = "cosine"

# ── Retrieval ─────────────────────────────────────────
TOP_K_RETRIEVAL = 20          # legacy (used as fallback)
TOP_K_DENSE = 30              # dense vector retrieval window
TOP_K_BM25 = 30               # BM25 sparse retrieval window
TOP_K_RERANK = 10             # final reranked results for LLM
RRF_K = 60                    # reciprocal rank fusion constant

# ── Groq LLM ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MAX_TOKENS = 3072
GROQ_TEMPERATURE = 0.4

# ── Langfuse Observability ────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Semantic Cache ────────────────────────────────────
CACHE_DB = BASE_DIR / "semantic_cache.db"
CACHE_SIMILARITY_THRESHOLD = 0.92

# ── Groq Pricing (per million tokens, for cost estimation) ─
GROQ_INPUT_COST_PER_M = 0.59   # $/M input tokens
GROQ_OUTPUT_COST_PER_M = 0.79  # $/M output tokens
