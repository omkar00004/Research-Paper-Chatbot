"""
Generation module.
Constructs grounded prompts and generates answers via Groq LLM.
Enhanced for research paper accuracy with equation/table/figure awareness.
"""
import logging
from typing import List, Dict, Any

from groq import Groq

from backend.config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

logger = logging.getLogger(__name__)

# Singleton Groq client
_client: Groq | None = None


def _get_client() -> Groq:
    """Get or create the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it before running the chatbot."
            )
        _client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"Groq client initialized (model: {GROQ_MODEL})")
    return _client


SYSTEM_PROMPT = """You are a highly accurate research paper question-answering assistant.
Your answers must be strictly grounded in the provided context from academic research papers.

RULES:
1. ONLY answer based on the provided context. Do NOT use external knowledge.
2. If the context does not contain enough information to answer, say "I cannot find sufficient information in the provided research papers to answer this question."
3. Cite your sources precisely using ONLY the provided Source ID numbers in brackets. For example: "The methodology focuses on text analysis [1]." or "This is a key finding [1][3]." Do NOT write out the paper name or page number in the generated text.
4. Be precise, concise, and academic in tone.
5. If multiple papers provide relevant information, synthesize them and cite all corresponding source IDs.
6. Use bullet points or numbered lists for clarity when appropriate.
7. For technical concepts, provide clear explanations as found in the papers.
8. Preserve mathematical equations, formulas, and notation exactly as they appear in the source context. Use LaTeX-style notation where possible.
9. When referencing tables or figures mentioned in the context, note their original numbering (e.g., "Table 2", "Figure 3").
10. Pay attention to the relevance scores provided -  sources with higher scores are more likely to contain the answer.
11. If the question asks about methodology, results, or specific sections, focus your answer on the most relevant chunks and cite them accurately.
12. Do not fabricate citations or source IDs. Only use the [Source ID: X] numbers provided in the context blocks."""


def _build_context(chunks: List[Dict[str, Any]], sources: List[Dict[str, str]]) -> str:
    """Build context string from re-ranked chunks mapped to unique source IDs."""
    source_map = {}
    for i, s in enumerate(sources, 1):
        key = f"{s['paper']}::p{s['page']}"
        source_map[key] = i

    context_parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        key = f"{source}::p{page}"
        source_idx = source_map.get(key, 1)

        # Include relevance score if available
        score_info = ""
        if "rerank_score" in chunk:
            score_info = f" | Relevance: {chunk['rerank_score']:.3f}"
        elif "rrf_score" in chunk:
            score_info = f" | Relevance: {chunk['rrf_score']:.4f}"

        context_parts.append(
            f"[Source ID: {source_idx}] Paper: \"{source}\" | Page: {page}{score_info}\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(context_parts)


def _extract_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract unique source references from chunks."""
    seen = set()
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        key = f"{source}::p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper": source,
                "page": str(page),
                "section": meta.get("section", ""),
                "chunk_id": chunk.get("id", ""),
            })
    return sources


def generate(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a grounded answer using Groq LLM.

    Args:
        query: The user's question.
        chunks: Re-ranked context chunks.

    Returns:
        Dict with 'answer', 'sources', and 'model'.
    """
    if not chunks:
        return {
            "answer": "I cannot find any relevant information in the indexed research papers. "
                      "Please make sure papers have been uploaded and indexed.",
            "sources": [],
            "model": GROQ_MODEL,
        }

    client = _get_client()
    sources = _extract_sources(chunks)
    context = _build_context(chunks, sources)

    user_prompt = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive, well-structured answer based ONLY on the context above. "
        f"Cite key points using the [Source ID: X] brackets provided."
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
        )

        answer = response.choices[0].message.content
        logger.info(f"Generated answer ({len(answer)} chars) using {GROQ_MODEL}")

        return {
            "answer": answer,
            "sources": sources,
            "model": GROQ_MODEL,
        }

    except Exception as e:
        logger.error(f"Groq generation error: {e}")
        return {
            "answer": f"An error occurred while generating the answer: {str(e)}",
            "sources": sources,
            "model": GROQ_MODEL,
        }
