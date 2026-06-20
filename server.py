"""
FastAPI server for Research Paper QA Chatbot.
Serves the REST API and static frontend.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "server.log", mode="a"),
    ],
)
logger = logging.getLogger("Server")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from backend.pipeline import ingest_papers, ingest_single_paper, query
from backend.indexing import get_collection_stats, delete_paper, clear_index
from backend.config import UPLOAD_DIR
from backend.conversations import (
    create_conversation,
    list_conversations,
    get_conversation,
    update_conversation_title,
    delete_conversation,
    add_message,
)

# ── FastAPI App ───────────────────────────────────────

app = FastAPI(
    title="Scholar -  RAG Research Assistant",
    description="Upload academic PDFs and ask grounded questions. Every answer cited.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = "New conversation"


class ConversationUpdate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    role: str
    content: str
    sources: Optional[list] = []
    pipeline: Optional[list] = []
    time: Optional[float] = 0


# ── API Endpoints ─────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    """Get collection statistics."""
    try:
        stats = get_collection_stats()
        return stats
    except Exception as e:
        return {"total_chunks": 0, "total_papers": 0, "papers": []}


@app.get("/api/papers")
async def api_papers():
    """List indexed papers with stats."""
    try:
        stats = get_collection_stats()
        return {"papers": stats["papers"], "total": stats["total_papers"]}
    except Exception:
        return {"papers": [], "total": 0}


@app.delete("/api/papers/{paper_name:path}")
async def api_delete_paper(paper_name: str):
    """Delete a paper from the index."""
    count = delete_paper(paper_name)
    if count > 0:
        return {"status": "success", "deleted_chunks": count}
    raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/api/ingest")
async def api_ingest():
    """Trigger bulk ingestion of papers from the Research Paper directory."""
    result = ingest_papers()
    return result


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Upload and ingest a PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    upload_path = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    result = ingest_single_paper(upload_path)
    return result


@app.post("/api/query")
async def api_query(req: QueryRequest):
    """Process a question through the hybrid RAG pipeline."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = query(req.question)

    # If conversation_id provided, save messages
    if req.conversation_id:
        try:
            # Save user message
            add_message(req.conversation_id, "user", req.question)
            # Save assistant message
            add_message(
                req.conversation_id,
                "assistant",
                result.get("answer", ""),
                sources=result.get("sources", []),
                pipeline=result.get("pipeline", []),
                time_taken=result.get("time", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to save messages: {e}")

    return result


@app.post("/api/reset")
async def api_reset():
    """Reset the vector index."""
    success = clear_index()
    if success:
        return {"status": "success", "message": "Vector index cleared"}
    raise HTTPException(status_code=500, detail="Failed to clear index")


# ── Conversation Endpoints ────────────────────────────

@app.get("/api/conversations")
async def api_list_conversations():
    """List all conversations."""
    return list_conversations()


@app.post("/api/conversations")
async def api_create_conversation(req: ConversationCreate):
    """Create a new conversation."""
    return create_conversation(req.title)


@app.get("/api/conversations/{conv_id}")
async def api_get_conversation(conv_id: str):
    """Get a conversation with all messages."""
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.put("/api/conversations/{conv_id}")
async def api_update_conversation(conv_id: str, req: ConversationUpdate):
    """Update conversation title."""
    success = update_conversation_title(conv_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success"}


@app.delete("/api/conversations/{conv_id}")
async def api_delete_conversation(conv_id: str):
    """Delete a conversation."""
    success = delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success"}


@app.post("/api/conversations/{conv_id}/messages")
async def api_add_message(conv_id: str, req: MessageCreate):
    """Add a message to a conversation."""
    msg = add_message(
        conv_id,
        req.role,
        req.content,
        sources=req.sources,
        pipeline=req.pipeline,
        time_taken=req.time,
    )
    return msg


# ── Static Frontend ───────────────────────────────────

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Startup Event ─────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize BM25 index on startup."""
    logger.info("Starting Research Paper QA Server...")
    try:
        from backend.hybrid_retrieval import get_bm25_index
        bm25 = get_bm25_index()
        count = bm25.build()
        logger.info(f"BM25 index initialized with {count} documents")
    except Exception as e:
        logger.warning(f"BM25 index init skipped (no documents yet): {e}")
    logger.info("Server ready!")
