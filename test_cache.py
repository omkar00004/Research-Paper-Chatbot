#!/usr/bin/env python3
"""
Semantic Cache Performance Test.

Runs 50 queries (mix of exact repeats, paraphrases, and unique queries)
through the RAG pipeline and reports cache hit rate, latency comparison,
and estimated cost savings.

Usage:
    python test_cache.py
"""
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must import tracing before any @observe usage
import backend.tracing  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_FILE = EVAL_DIR / "cache_test_results.json"


# ── Test Query Definitions ────────────────────────────
# Mix of 15 unique, 15 exact repeats, 20 paraphrases

UNIQUE_QUERIES = [
    "What is the attention mechanism in transformers?",
    "How does BERT handle bidirectional context?",
    "What are the key findings about content moderation?",
    "Explain the concept of transfer learning in NLP.",
    "What evaluation metrics are used in the papers?",
    "How does the proposed method compare to baselines?",
    "What datasets were used for training?",
    "Describe the architecture of the proposed model.",
    "What are the limitations mentioned in the research?",
    "How is data preprocessing handled?",
    "What is the role of embeddings in the system?",
    "Explain the training procedure described in the papers.",
    "What future work is suggested by the authors?",
    "How does the system handle multi-modal inputs?",
    "What are the main contributions of the research?",
    "Describe the methodology used in the experiments.",
    "What is the significance of the results?",
    "How does fine-tuning affect model performance?",
    "What challenges does the paper address?",
    "Explain the loss function used in training.",
]

# Exact repeats of the first 15 unique queries
EXACT_REPEATS = UNIQUE_QUERIES[:15]

# Paraphrases of the first 15 unique queries
PARAPHRASE_QUERIES = [
    "Can you explain how attention works in transformer models?",
    "How does BERT process context from both directions?",
    "What were the important discoveries about moderating content?",
    "What is transfer learning in natural language processing?",
    "Which evaluation metrics do the papers report?",
    "How does their method perform against baseline approaches?",
    "What training datasets are mentioned in the papers?",
    "What is the model architecture they propose?",
    "What limitations are discussed in these papers?",
    "How is the data preprocessed before training?",
    "What role do word embeddings play in the system?",
    "Describe how the models were trained according to the papers.",
    "What directions for future research do the authors propose?",
    "How are multi-modal inputs processed by the system?",
    "What are the primary contributions of these research papers?",
]


def build_query_list() -> list:
    """Build the 50-query test list with labels."""
    queries = []

    # First 20: unique queries (these will be cache misses)
    for q in UNIQUE_QUERIES:
        queries.append({"query": q, "type": "unique"})

    # Next 15: exact repeats (should be cache hits)
    for q in EXACT_REPEATS:
        queries.append({"query": q, "type": "exact_repeat"})

    # Next 15: paraphrases (may or may not hit cache depending on threshold)
    for q in PARAPHRASE_QUERIES:
        queries.append({"query": q, "type": "paraphrase"})

    return queries


def run_test():
    """Run the 50-query cache test."""
    from backend.pipeline import query
    from backend.semantic_cache import clear_cache, get_cache_stats
    from backend.config import GROQ_INPUT_COST_PER_M, GROQ_OUTPUT_COST_PER_M

    EVAL_DIR.mkdir(exist_ok=True)

    # Clear cache for a clean test
    clear_cache()
    logger.info("Cleared semantic cache for clean test")

    queries = build_query_list()
    results = []

    hits = 0
    misses = 0
    hit_latencies = []
    miss_latencies = []

    logger.info(f"\nRunning {len(queries)} queries through the pipeline...\n")

    for i, item in enumerate(queries):
        q = item["query"]
        qtype = item["type"]

        logger.info(f"[{i+1:2d}/{len(queries)}] ({qtype:14s}) {q[:55]}...")

        start = time.time()
        result = query(q)
        elapsed = time.time() - start

        is_hit = result.get("cache_hit", False)

        if is_hit:
            hits += 1
            hit_latencies.append(elapsed)
        else:
            misses += 1
            miss_latencies.append(elapsed)

        results.append({
            "query": q,
            "type": qtype,
            "cache_hit": is_hit,
            "latency": round(elapsed, 4),
            "similarity": result.get("cache_similarity"),
        })

        status = "HIT ✓" if is_hit else "MISS ✗"
        sim = f" (sim={result.get('cache_similarity', 'N/A')})" if is_hit else ""
        logger.info(f"         → {status}{sim} in {elapsed:.3f}s")

    # Compute statistics
    total = len(queries)
    hit_rate = hits / total * 100 if total > 0 else 0
    avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0
    avg_miss_latency = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0

    # Estimate cost savings per 100 queries
    # Average Groq generation: ~800 input tokens + ~400 output tokens
    avg_input_tokens = 800
    avg_output_tokens = 400
    cost_per_query = (
        avg_input_tokens * GROQ_INPUT_COST_PER_M / 1_000_000
        + avg_output_tokens * GROQ_OUTPUT_COST_PER_M / 1_000_000
    )
    est_hits_per_100 = hit_rate  # hit_rate is already per 100
    cost_saved_per_100 = est_hits_per_100 * cost_per_query

    # Breakdown by query type
    type_stats = {}
    for qtype in ["unique", "exact_repeat", "paraphrase"]:
        type_results = [r for r in results if r["type"] == qtype]
        type_hits = sum(1 for r in type_results if r["cache_hit"])
        type_stats[qtype] = {
            "total": len(type_results),
            "hits": type_hits,
            "hit_rate": round(type_hits / len(type_results) * 100, 1) if type_results else 0,
        }

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": total,
        "hits": hits,
        "misses": misses,
        "hit_rate_pct": round(hit_rate, 1),
        "avg_hit_latency_s": round(avg_hit_latency, 4),
        "avg_miss_latency_s": round(avg_miss_latency, 4),
        "latency_speedup_x": round(avg_miss_latency / avg_hit_latency, 1) if avg_hit_latency > 0 else 0,
        "estimated_cost_saved_per_100_queries_usd": round(cost_saved_per_100, 4),
        "cost_per_query_usd": round(cost_per_query, 6),
        "type_breakdown": type_stats,
        "details": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*65}")
    print(f"{'SEMANTIC CACHE PERFORMANCE REPORT':^65}")
    print(f"{'='*65}")
    print(f"\n{'OVERALL':^65}")
    print(f"  Total Queries:        {total}")
    print(f"  Cache Hits:           {hits}")
    print(f"  Cache Misses:         {misses}")
    print(f"  Hit Rate:             {hit_rate:.1f}%")
    print(f"\n{'LATENCY':^65}")
    print(f"  Avg Hit Latency:      {avg_hit_latency:.4f}s")
    print(f"  Avg Miss Latency:     {avg_miss_latency:.4f}s")
    if avg_hit_latency > 0:
        print(f"  Speedup:              {avg_miss_latency / avg_hit_latency:.1f}x faster on hits")
    print(f"\n{'COST SAVINGS':^65}")
    print(f"  Cost per LLM query:   ${cost_per_query:.6f}")
    print(f"  Est. savings/100 q:   ${cost_saved_per_100:.4f}")
    print(f"\n{'BREAKDOWN BY QUERY TYPE':^65}")
    print(f"  {'Type':<16} {'Total':<8} {'Hits':<8} {'Hit Rate':<10}")
    print(f"  {'-'*42}")
    for qtype, stats in type_stats.items():
        print(f"  {qtype:<16} {stats['total']:<8} {stats['hits']:<8} {stats['hit_rate']:.1f}%")
    print(f"\n{'='*65}")
    print(f"Results saved to: {RESULTS_FILE}")

    cache_stats = get_cache_stats()
    print(f"Cache entries: {cache_stats['total_entries']}")
    print(f"Similarity threshold: {cache_stats['threshold']}")


if __name__ == "__main__":
    run_test()
