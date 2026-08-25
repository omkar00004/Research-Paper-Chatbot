#!/usr/bin/env python3
"""
RAGAS Evaluation Runner.

Runs the existing RAG pipeline against the approved test set and computes
RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall.

Results are stored with timestamps in eval/eval_runs.json and rendered as
a markdown table in eval/eval_results.md.

Usage:
    python run_eval.py                      # Run with hybrid retrieval (default)
    python run_eval.py --mode dense-only    # Run with dense-only retrieval
    python run_eval.py --mode hybrid        # Explicitly use hybrid (default)
"""
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must import tracing before any @observe usage
import backend.tracing  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = PROJECT_ROOT / "eval"
TEST_SET_FILE = EVAL_DIR / "test_set.json"
EVAL_RUNS_FILE = EVAL_DIR / "eval_runs.json"
EVAL_RESULTS_FILE = EVAL_DIR / "eval_results.md"


def load_test_set() -> list:
    """Load the approved test set."""
    if not TEST_SET_FILE.exists():
        logger.error(f"Test set not found at {TEST_SET_FILE}")
        logger.error("Run: python generate_testset.py --approve-all")
        sys.exit(1)

    with open(TEST_SET_FILE) as f:
        test_set = json.load(f)

    # Filter to approved pairs only
    approved = [p for p in test_set if not p.get("flagged_for_review", True)]

    if not approved:
        logger.error("No approved Q&A pairs found. All pairs are still flagged for review.")
        logger.error("Set 'flagged_for_review': false for approved pairs, or use --approve-all")
        sys.exit(1)

    logger.info(f"Loaded {len(approved)} approved Q&A pairs")
    return approved


def run_pipeline_queries(test_set: list, mode: str = "hybrid") -> list:
    """Run each test question through the RAG pipeline and collect results."""
    from backend.pipeline import query
    from backend.semantic_cache import clear_cache

    # Clear cache to ensure fresh retrieval for evaluation
    clear_cache()
    logger.info("Cleared semantic cache for clean evaluation")

    results = []

    for i, item in enumerate(test_set):
        question = item["question"]
        logger.info(f"  [{i+1}/{len(test_set)}] {question[:60]}...")

        start = time.time()

        if mode == "dense-only":
            # Patch hybrid_retrieve to use dense-only retrieval
            from backend.retrieval import retrieve as dense_retrieve
            with patch("backend.pipeline._retrieve_step") as mock_retrieve:
                # Create a traced wrapper around dense-only retrieval
                def dense_only_retrieve(q):
                    from backend.retrieval import retrieve
                    return retrieve(q)
                mock_retrieve.side_effect = dense_only_retrieve
                result = query(question)
        else:
            result = query(question)

        elapsed = time.time() - start

        # Extract contexts from pipeline result
        contexts = []
        if result.get("sources"):
            for src in result["sources"]:
                # Try to reconstruct context from source info
                contexts.append(f"Paper: {src.get('paper', '?')}, Page: {src.get('page', '?')}")

        # Also try to get actual chunk texts from the pipeline
        # The pipeline doesn't directly expose raw chunks in the result,
        # so we re-retrieve for context texts
        try:
            from backend.hybrid_retrieval import hybrid_retrieve
            from backend.retrieval import retrieve as dense_retrieve
            from backend.reranking import rerank

            if mode == "dense-only":
                retrieved = dense_retrieve(question)
            else:
                retrieved = hybrid_retrieve(question)

            reranked = rerank(question, retrieved)
            contexts = [chunk["text"] for chunk in reranked]
        except Exception as e:
            logger.warning(f"  Could not re-retrieve contexts: {e}")

        results.append({
            "question": question,
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "source_paper": item.get("source_paper", ""),
            "latency": round(elapsed, 3),
        })

        # Clear cache after each query to avoid cache hits during eval
        clear_cache()

    return results


def compute_ragas_metrics(results: list) -> dict:
    """Compute RAGAS metrics from pipeline results."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from backend.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL_NAME

    logger.info("Computing RAGAS metrics...")

    # Prepare dataset in RAGAS format
    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset = Dataset.from_dict(data)

    # Set up LLM and embeddings for RAGAS
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=2048,
        max_retries=10,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    def _extract_metric_score(res, metric_name):
        try:
            if hasattr(res, "__getitem__"):
                val = res[metric_name]
            elif hasattr(res, "scores"):
                val = res.scores.get(metric_name, None)
            else:
                val = getattr(res, metric_name, None)

            # If val is a list/series, compute mean
            if isinstance(val, (list, tuple)):
                val = sum(v for v in val if v is not None) / len(val) if val else None
            return round(float(val), 4) if val is not None else None
        except Exception:
            return None

    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            is_async=False,
        )

        scores = {
            "faithfulness": _extract_metric_score(ragas_result, "faithfulness"),
            "answer_relevancy": _extract_metric_score(ragas_result, "answer_relevancy"),
            "context_precision": _extract_metric_score(ragas_result, "context_precision"),
            "context_recall": _extract_metric_score(ragas_result, "context_recall"),
        }

        logger.info(f"RAGAS scores: {scores}")
        return scores

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        logger.error("Trying individual metrics...")

        scores = {}
        for metric in metrics:
            metric_name = getattr(metric, "name", str(metric))
            try:
                single_result = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    llm=llm,
                    embeddings=embeddings,
                    raise_exceptions=False,
                    is_async=False,
                )
                scores[metric_name] = _extract_metric_score(single_result, metric_name)
                logger.info(f"  {metric_name}: {scores[metric_name]}")
                time.sleep(2)
            except Exception as me:
                logger.warning(f"  Failed to compute {metric_name}: {me}")
                scores[metric_name] = None

        return scores


def save_run(mode: str, scores: dict, results: list, run_time: float):
    """Save evaluation run with timestamp to eval_runs.json."""
    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "num_questions": len(results),
        "scores": scores,
        "avg_latency": round(
            sum(r["latency"] for r in results) / len(results), 3
        ) if results else 0,
        "total_time": round(run_time, 2),
    }

    # Append to eval_runs.json
    runs = []
    if EVAL_RUNS_FILE.exists():
        with open(EVAL_RUNS_FILE) as f:
            runs = json.load(f)

    runs.append(run_entry)

    with open(EVAL_RUNS_FILE, "w") as f:
        json.dump(runs, f, indent=2)

    logger.info(f"Saved run to {EVAL_RUNS_FILE}")

    # Log to Langfuse v3
    try:
        from langfuse import get_client
        client = get_client()
        if client and hasattr(client, "span"):
            client.span(
                name="ragas_evaluation",
                input={"mode": mode, "num_questions": len(results)},
                output=scores,
                metadata=run_entry,
            )
            client.flush()
            logger.info("Logged evaluation run to Langfuse")
    except Exception as e:
        logger.warning(f"Could not log to Langfuse: {e}")

    return run_entry


def render_markdown(runs: list):
    """Render evaluation results as a markdown table."""
    if not runs:
        return

    md = ["# RAGAS Evaluation Results\n"]
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    md.append("## Latest Results\n")
    md.append("| Metric | Score |")
    md.append("|--------|-------|")

    latest = runs[-1]
    for metric, score in latest["scores"].items():
        score_str = f"{score:.4f}" if score is not None else "N/A"
        md.append(f"| {metric.replace('_', ' ').title()} | {score_str} |")

    md.append(f"\n_Mode: {latest['mode']} | Questions: {latest['num_questions']} | "
              f"Avg Latency: {latest['avg_latency']}s | "
              f"Run Time: {latest['total_time']}s_\n")

    # If multiple runs, show comparison
    if len(runs) > 1:
        md.append("\n## Run History\n")
        md.append("| Timestamp | Mode | Questions | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency |")
        md.append("|-----------|------|-----------|-------------|-----------------|-------------------|---------------|-------------|")

        for run in runs:
            ts = run["timestamp"][:19].replace("T", " ")
            scores = run["scores"]
            f_score = f"{scores.get('faithfulness', 'N/A')}" if scores.get('faithfulness') is not None else "N/A"
            ar_score = f"{scores.get('answer_relevancy', 'N/A')}" if scores.get('answer_relevancy') is not None else "N/A"
            cp_score = f"{scores.get('context_precision', 'N/A')}" if scores.get('context_precision') is not None else "N/A"
            cr_score = f"{scores.get('context_recall', 'N/A')}" if scores.get('context_recall') is not None else "N/A"

            md.append(
                f"| {ts} | {run['mode']} | {run['num_questions']} | "
                f"{f_score} | {ar_score} | {cp_score} | {cr_score} | "
                f"{run['avg_latency']}s |"
            )

    md.append("")

    with open(EVAL_RESULTS_FILE, "w") as f:
        f.write("\n".join(md))

    logger.info(f"Rendered results to {EVAL_RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense-only"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"RAGAS Evaluation Runner — Mode: {args.mode}")
    logger.info("=" * 60)

    # Load test set
    test_set = load_test_set()

    # Run pipeline queries
    logger.info(f"\nRunning {len(test_set)} queries through the pipeline...")
    run_start = time.time()
    results = run_pipeline_queries(test_set, mode=args.mode)
    query_time = time.time() - run_start
    logger.info(f"Queries complete in {query_time:.1f}s")

    # Compute RAGAS metrics
    scores = compute_ragas_metrics(results)

    total_time = time.time() - run_start

    # Save run
    run_entry = save_run(args.mode, scores, results, total_time)

    # Load all runs and render markdown
    runs = []
    if EVAL_RUNS_FILE.exists():
        with open(EVAL_RUNS_FILE) as f:
            runs = json.load(f)
    render_markdown(runs)

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'RAGAS EVALUATION RESULTS':^60}")
    print(f"{'='*60}")
    print(f"Mode:       {args.mode}")
    print(f"Questions:  {len(test_set)}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Avg Latency: {run_entry['avg_latency']}s per query")
    print(f"{'-'*60}")
    for metric, score in scores.items():
        name = metric.replace("_", " ").title()
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {name:<25} {score_str}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {EVAL_RESULTS_FILE}")
    print(f"Run history at:   {EVAL_RUNS_FILE}")


if __name__ == "__main__":
    main()
