#!/usr/bin/env python3
"""
Generate Test Set for RAGAS Evaluation.

Samples chunks from ChromaDB, uses the Groq LLM to draft candidate Q&A pairs
grounded in those chunks, and saves them for human review before finalizing.

Usage:
    python generate_testset.py                  # Generate and flag for review
    python generate_testset.py --approve-all    # Auto-approve all pairs
    python generate_testset.py --num-pairs 30   # Generate 30 pairs (default: 25)
"""
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.indexing import _get_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "eval"
CANDIDATE_FILE = EVAL_DIR / "candidate_qa_pairs.json"
TEST_SET_FILE = EVAL_DIR / "test_set.json"

# Prompt for generating Q&A pairs from a chunk
QA_GENERATION_PROMPT = """You are an expert at creating evaluation question-answer pairs from research paper excerpts.

Given the following excerpt from a research paper, generate {num_pairs} diverse question-answer pairs that:
1. Are directly answerable from the excerpt content ONLY
2. Cover different aspects of the excerpt (methodology, findings, concepts, etc.)
3. Have concise but complete answers (2-4 sentences each)
4. Include a mix of factual, conceptual, and analytical questions
5. Do NOT ask questions that require external knowledge

Paper: "{source}"
Page: {page}

Excerpt:
---
{chunk_text}
---

Respond with ONLY a valid JSON array of objects. Each object must have exactly two keys: "question" and "answer".
Example format:
[
  {{"question": "What method does the paper propose for X?", "answer": "The paper proposes..."}},
  {{"question": "How does Y compare to Z according to the results?", "answer": "According to the results..."}}
]

Generate exactly {num_pairs} question-answer pairs:"""


def sample_chunks(num_chunks: int = 50) -> list:
    """Sample diverse chunks from ChromaDB, spread across papers."""
    collection = _get_collection()
    total = collection.count()

    if total == 0:
        logger.error("ChromaDB collection is empty. Please ingest papers first.")
        sys.exit(1)

    logger.info(f"Collection has {total} chunks total")

    # Fetch all chunks
    results = collection.get(
        include=["documents", "metadatas"],
        limit=total,
    )

    # Group by source paper
    by_source = defaultdict(list)
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        source = meta.get("source", "Unknown")
        by_source[source].append({
            "id": results["ids"][i],
            "text": doc,
            "metadata": meta,
        })

    logger.info(f"Found {len(by_source)} papers: {list(by_source.keys())}")

    # Sample proportionally from each paper
    sampled = []
    papers = list(by_source.keys())
    chunks_per_paper = max(2, num_chunks // len(papers))

    for paper in papers:
        chunks = by_source[paper]
        # Filter to substantial chunks (>150 chars)
        substantial = [c for c in chunks if len(c["text"]) > 150]
        if not substantial:
            substantial = chunks

        n = min(chunks_per_paper, len(substantial))
        sampled.extend(random.sample(substantial, n))

    # Shuffle and trim
    random.shuffle(sampled)
    sampled = sampled[:num_chunks]

    logger.info(f"Sampled {len(sampled)} chunks from {len(papers)} papers")
    return sampled


def generate_qa_pairs(chunks: list, num_pairs: int = 25) -> list:
    """Use Groq LLM to generate Q&A pairs from sampled chunks."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    all_pairs = []

    # Decide how many pairs per chunk to reach target
    pairs_per_chunk = max(1, num_pairs // len(chunks))
    remaining = num_pairs

    for i, chunk in enumerate(chunks):
        if remaining <= 0:
            break

        n = min(pairs_per_chunk, remaining)
        if i == len(chunks) - 1:
            n = remaining  # Last chunk gets whatever is left

        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "?")

        logger.info(f"  [{i+1}/{len(chunks)}] Generating {n} pairs from '{source}' page {page}")

        prompt = QA_GENERATION_PROMPT.format(
            num_pairs=n,
            source=source,
            page=page,
            chunk_text=chunk["text"][:2000],  # Limit chunk length for prompt
        )

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You generate evaluation Q&A pairs from research paper excerpts. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            # Handle cases where LLM wraps in ```json ... ```
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            pairs = json.loads(content)

            for pair in pairs:
                if "question" in pair and "answer" in pair:
                    all_pairs.append({
                        "question": pair["question"],
                        "ground_truth": pair["answer"],
                        "source_chunk": chunk["text"][:500],
                        "source_paper": source,
                        "page": str(page),
                        "chunk_id": chunk["id"],
                        "flagged_for_review": True,
                    })
                    remaining -= 1

        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse LLM response as JSON: {e}")
            continue
        except Exception as e:
            logger.warning(f"  Error generating pairs: {e}")
            continue

    logger.info(f"Generated {len(all_pairs)} candidate Q&A pairs")
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate test set for RAGAS evaluation")
    parser.add_argument("--num-pairs", type=int, default=25, help="Number of Q&A pairs to generate (default: 25)")
    parser.add_argument("--approve-all", action="store_true", help="Auto-approve all generated pairs")
    args = parser.parse_args()

    EVAL_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("RAGAS Test Set Generator")
    logger.info("=" * 60)

    # Sample chunks
    num_chunks = min(args.num_pairs * 3, 60)  # Sample more chunks than needed
    chunks = sample_chunks(num_chunks)

    # Generate Q&A pairs
    logger.info(f"\nGenerating {args.num_pairs} Q&A pairs using {GROQ_MODEL}...")
    pairs = generate_qa_pairs(chunks, args.num_pairs)

    if not pairs:
        logger.error("No Q&A pairs generated. Check your Groq API key and try again.")
        sys.exit(1)

    # Save candidate pairs
    with open(CANDIDATE_FILE, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"\n✅ Saved {len(pairs)} candidate Q&A pairs to {CANDIDATE_FILE}")

    if args.approve_all:
        # Auto-approve all pairs
        for pair in pairs:
            pair["flagged_for_review"] = False
        with open(TEST_SET_FILE, "w") as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"✅ Auto-approved all pairs → saved to {TEST_SET_FILE}")
    else:
        logger.info(f"\n📋 REVIEW INSTRUCTIONS:")
        logger.info(f"   1. Open {CANDIDATE_FILE}")
        logger.info(f"   2. Review each Q&A pair for accuracy")
        logger.info(f"   3. Remove any bad pairs")
        logger.info(f"   4. Set 'flagged_for_review': false for approved pairs")
        logger.info(f"   5. Save as {TEST_SET_FILE}")
        logger.info(f"\n   OR run: python generate_testset.py --approve-all")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'CANDIDATE Q&A PAIRS SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Paper':<35} {'Question Preview':<30}")
    print(f"{'-'*70}")
    for i, pair in enumerate(pairs, 1):
        paper = pair["source_paper"][:33]
        question = pair["question"][:28]
        print(f"{i:<4} {paper:<35} {question:<30}")
    print(f"{'='*70}")
    print(f"Total: {len(pairs)} pairs from {len(set(p['source_paper'] for p in pairs))} papers")


if __name__ == "__main__":
    main()
