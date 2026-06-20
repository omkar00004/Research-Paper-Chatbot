"""
RAG Pipeline Orchestrator.
Coordinates the full ingestion and query pipelines.
Uses hybrid retrieval (dense + BM25 + RRF) for maximum accuracy.
Instrumented with Langfuse v3 @observe decorator for automatic tracing.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# IMPORTANT: tracing must be imported first to set env vars before any @observe
import backend.tracing  # noqa: F401 -  configures Langfuse env vars on import

from langfuse import observe, get_client

from backend.config import PAPERS_DIR, UPLOAD_DIR
from backend.ingestion import ingest_pdf, ingest_directory
from backend.chunking import chunk_documents
from backend.indexing import index_chunks, get_collection_stats, get_indexed_sources, delete_paper
from backend.hybrid_retrieval import hybrid_retrieve, get_bm25_index
from backend.reranking import rerank
from backend.generation import generate

logger = logging.getLogger(__name__)


# ── Ingestion Pipeline ────────────────────────────────

@observe(name="paper_ingestion")
def ingest_papers(directory: Optional[Path] = None) -> Dict[str, Any]:
    """Full ingestion pipeline: PDF → pages → chunks → embeddings → ChromaDB."""
    target_dir = directory or PAPERS_DIR
    start_time = time.time()

    try:
        client = get_client()
        client.update_current_trace(
            input={"directory": str(target_dir)},
            tags=["ingestion"],
        )
    except Exception:
        pass

    logger.info(f"═══ Starting Paper Ingestion Pipeline ═══")
    logger.info(f"Source directory: {target_dir}")

    already_indexed = get_indexed_sources()

    # Step 1: Extract
    documents = _ingest_pdfs(target_dir)

    if not documents:
        return {"status": "no_papers", "message": "No PDF files found", "time": 0}

    new_docs = [d for d in documents if d.metadata.get("source") not in already_indexed]

    if not new_docs:
        return {
            "status": "up_to_date",
            "message": "All papers already indexed",
            "papers": len(already_indexed),
            "time": time.time() - start_time,
        }

    # Step 2: Chunk
    chunks = _chunk_docs(new_docs)

    # Step 3: Index
    new_count = _index(chunks)

    # Step 4: Refresh BM25 index
    _refresh_bm25()

    elapsed = time.time() - start_time
    stats = get_collection_stats()

    return {
        "status": "success",
        "new_chunks": new_count,
        "total_chunks": stats["total_chunks"],
        "total_papers": stats["total_papers"],
        "papers": stats["papers"],
        "time": round(elapsed, 2),
    }


@observe(name="pdf_extraction")
def _ingest_pdfs(target_dir):
    return ingest_directory(target_dir)


@observe(name="chunking")
def _chunk_docs(docs):
    return chunk_documents(docs)


@observe(name="embedding_and_indexing")
def _index(chunks):
    return index_chunks(chunks)


def _refresh_bm25():
    """Refresh the BM25 index after new documents are indexed."""
    try:
        bm25_index = get_bm25_index()
        bm25_index.refresh()
    except Exception as e:
        logger.warning(f"Failed to refresh BM25 index: {e}")


def ingest_single_paper(pdf_path: Path) -> Dict[str, Any]:
    """Ingest a single PDF file."""
    start_time = time.time()
    logger.info(f"Ingesting single paper: {pdf_path.name}")

    documents = ingest_pdf(pdf_path)
    if not documents:
        return {"status": "error", "message": "Could not extract text from PDF"}

    chunks = chunk_documents(documents)
    new_count = index_chunks(chunks)

    # Refresh BM25 index with new documents
    _refresh_bm25()

    elapsed = time.time() - start_time
    return {
        "status": "success",
        "paper": pdf_path.name,
        "chunks": new_count,
        "time": round(elapsed, 2),
    }


# ── Query Classification ──────────────────────────────

def _is_casual_query(question: str) -> bool:
    """Detect if the query is casual/greeting rather than a research question."""
    q = question.strip().lower().rstrip("!?.,'\"")

    casual_phrases = {
        "hi", "hello", "hey", "hii", "hiii", "yo", "sup",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "what's up", "whats up",
        "thanks", "thank you", "thankyou", "thx",
        "bye", "goodbye", "see you", "ok", "okay",
        "who are you", "what are you", "what can you do",
        "help", "help me",
    }

    if q in casual_phrases:
        return True

    words = q.split()
    if len(words) <= 2 and not any(kw in q for kw in [
        "what", "how", "why", "explain", "describe", "compare",
        "summarize", "analyze", "define", "transformer", "attention",
        "model", "paper", "algorithm", "method", "result",
    ]):
        return True

    return False


@observe(name="casual_response", as_type="generation")
def _casual_response(question: str) -> Dict[str, Any]:
    """Generate a friendly response for casual queries using Groq directly."""
    from groq import Groq
    from backend.config import GROQ_API_KEY, GROQ_MODEL

    try:
        client_lf = get_client()
        client_lf.update_current_span(
            input={"query": question},
            metadata={"type": "casual", "model": GROQ_MODEL},
        )
    except Exception:
        pass

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly Research Paper QA assistant. "
                    "For casual messages like greetings, respond warmly and briefly. "
                    "Let the user know you can help them with questions about their "
                    "uploaded research papers. Keep it short and friendly (2-3 sentences max)."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=256,
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    usage = response.usage

    try:
        client_lf = get_client()
        client_lf.update_current_span(
            output=answer,
            metadata={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        )
    except Exception:
        pass

    return {
        "answer": answer,
        "sources": [],
        "model": GROQ_MODEL,
        "pipeline": [{"step": "direct_response", "time": 0}],
        "time": 0,
    }


# ── RAG Sub-steps (each traced as a span) ─────────────

@observe(name="hybrid_retrieval", as_type="retriever")
def _retrieve_step(question: str):
    """Retrieve relevant chunks using hybrid search (dense + BM25 + RRF)."""
    retrieved_chunks = hybrid_retrieve(question)

    try:
        client_lf = get_client()
        client_lf.update_current_span(
            output={
                "count": len(retrieved_chunks),
                "method": "hybrid_rrf",
                "sources": list(set(
                    c.get("metadata", {}).get("source", "?") for c in retrieved_chunks
                )),
            },
        )
    except Exception:
        pass

    return retrieved_chunks


@observe(name="reranking")
def _rerank_step(question: str, retrieved_chunks):
    """Re-rank chunks using FlashRank with source diversity."""
    reranked = rerank(question, retrieved_chunks)

    try:
        client_lf = get_client()
        client_lf.update_current_span(
            output={
                "count": len(reranked),
                "sources": list(set(
                    c.get("metadata", {}).get("source", "?") for c in reranked
                )),
                "top_score": reranked[0].get("rerank_score") if reranked else None,
            },
        )
    except Exception:
        pass

    return reranked


@observe(name="llm_generation", as_type="generation")
def _generate_step(question: str, reranked_chunks):
    """Generate answer using Groq LLM."""
    result = generate(question, reranked_chunks)

    try:
        client_lf = get_client()
        client_lf.update_current_span(
            model="llama-3.3-70b-versatile",
            output=result.get("answer", "")[:500],
            metadata={"source_count": len(result.get("sources", []))},
        )
    except Exception:
        pass

    return result


# ── Main Query Pipeline ──────────────────────────────

@observe(name="rag_query")
def query(question: str) -> Dict[str, Any]:
    """
    Full query pipeline: classify → (casual response OR hybrid RAG pipeline).
    Automatically traced with Langfuse v3 @observe.
    """
    start_time = time.time()

    logger.info(f"═══ Query Pipeline ═══")
    logger.info(f"Question: {question}")

    try:
        client_lf = get_client()
        client_lf.update_current_trace(
            input={"query": question},
            tags=["query"],
        )
    except Exception:
        pass

    # ── Classification ──
    is_casual = _is_casual_query(question)

    if is_casual:
        logger.info("Classified as casual query → direct response")
        result = _casual_response(question)
        result["time"] = round(time.time() - start_time, 2)

        try:
            client_lf = get_client()
            client_lf.update_current_trace(
                output={"answer": result["answer"], "type": "casual"},
            )
        except Exception:
            pass

        backend.tracing.flush()
        return result

    # ── Hybrid RAG Pipeline ──
    pipeline_log = []

    # Step 1: Hybrid Retrieval (dense + BM25 + RRF)
    step_start = time.time()
    retrieved_chunks = _retrieve_step(question)
    retrieval_time = round(time.time() - step_start, 3)
    pipeline_log.append({
        "step": "hybrid_retrieval",
        "count": len(retrieved_chunks),
        "time": retrieval_time,
        "method": "dense + BM25 + RRF",
    })

    if not retrieved_chunks:
        no_result = {
            "answer": "No relevant documents found. Please make sure research papers have been uploaded and indexed.",
            "sources": [],
            "model": "",
            "pipeline": pipeline_log,
            "time": round(time.time() - start_time, 2),
        }
        backend.tracing.flush()
        return no_result

    # Step 2: Re-ranking
    step_start = time.time()
    reranked_chunks = _rerank_step(question, retrieved_chunks)
    rerank_time = round(time.time() - step_start, 3)
    pipeline_log.append({"step": "reranking", "count": len(reranked_chunks), "time": rerank_time})

    # Step 3: Generation
    step_start = time.time()
    result = _generate_step(question, reranked_chunks)
    gen_time = round(time.time() - step_start, 3)
    pipeline_log.append({"step": "generation", "model": result.get("model", ""), "time": gen_time})

    elapsed = time.time() - start_time
    result["pipeline"] = pipeline_log
    result["time"] = round(elapsed, 2)

    # Update trace with final output
    try:
        client_lf = get_client()
        client_lf.update_current_trace(
            output={
                "answer": result["answer"][:500],
                "source_count": len(result.get("sources", [])),
                "total_time": result["time"],
            },
        )
    except Exception:
        pass

    backend.tracing.flush()
    logger.info(f"═══ Query Complete ({elapsed:.2f}s) ═══")
    return result
