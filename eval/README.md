# RAG benchmark (branch `eval/rag-benchmark`)

Automated, resumable, call-efficient evaluation of the Research Paper RAG Chatbot. All code is in `eval/`; the only edits outside it are
env-var switches in `backend/config.py`, a `RAG_RERANK=0` switch in `backend/pipeline.py`, and a Langfuse span label fix (defaults unchanged).

## Reproduce

```bash
pip install -r eval/requirements.txt          # exact versions used
# .env (gitignored) must contain FREELLMAPI_BASE_URL and FREELLMAPI_API_KEY (+ optional LANGFUSE_* keys for S3 tracing)
python -m eval.run preflight                  # env check + one tiny call per frozen model
python -m eval.run all --dry-run              # 5 questions per stage; prints projections; shares the cache
python -m eval.run all                        # full run; resumable: re-run the same command after any exit
python -m eval.run all --extras               # adds C0 (no retrieval) and C3 answers in S3 and C3 in S4
python -m eval.agreement                      # after you fill the two human_check CSVs
```

Single stages: `s0 s1 s2 s3 s4 s5 s6 report`. Flags: `--max-calls N`, `--max-cost USD`, `--dry-run`, `--extras`.
Exit codes: **3** quota/outage (checkpointed; the exact resume command is printed), **4** budget reached, **5** frozen-model drift.

## Call efficiency
- S2 (grid + conditions) uses **zero API calls**: local embeddings, BM25 and FlashRank.
- One sqlite cache (`eval/cache/cache.sqlite`): embeddings by (model, chunk-text hash); FlashRank scores by (reranker, query, chunk-text hash), so pool 10/20 reuse the pool-50 scores; every LLM call by hash of (provider, model, messages, params). Conditions that retrieve the identical ordered context send the identical prompt, so they share one answer and one judgment.
- `eval/state.json` records every (stage, condition, question id); completed items are skipped on re-run; writes are atomic (tmp + rename, debounced to 2 s, flushed on exit). S4/S5 additionally resume through the call cache (zero new calls).
- One shared client (`eval/llm.py`): 429/5xx retries (max 5, Retry-After / gateway `retryAtMs` honoured, exponential backoff + jitter, a shared per-model cooldown so threads do not burn tries), concurrency <= 4, `finish_reason` and returned model logged for every call, one automatic retry with 3x max_tokens when `finish_reason=length`, exit 3 when the provider says the quota is daily / the reset is > 15 min away / cumulative wait would exceed 15 min / 6 consecutive failures.
- Deviation to note: the gateway tags short per-minute cooldowns with `daily_quota_exhausted` in its attempt trail. The client therefore trusts an advertised reset time (waits when <= 15 min, exits otherwise) and treats "daily" with no reset hint as a daily quota.
- Budget guards `--max-calls/--max-cost` count HTTP attempts; cost is $0 (free tiers) unless `PRICE_IN_PER_M/PRICE_OUT_PER_M` are set.
- `results/call_ledger.md` is rewritten after every stage (calls, tokens, cache hits, retries, 429s, truncations by stage/provider/model).

## Frozen models (never substituted mid-run; `eval/cache/frozen_models.json` stores returned strings)
| role | requested | notes |
|---|---|---|
| answer | `gpt-oss-20b` | same model as production `GROQ_MODEL` (`openai/gpt-oss-20b`), temperature 0, 150 max tokens, `reasoning_effort=low` |
| question writer | `nemotron-3-super-120b` | different model and family from the answerer |
| judge (`JUDGE_MODEL`) | `nemotron-3-super-120b` | different family from the answerer; **same model as the writer** |

History: before any result existed, `qwen3.8-27b` (Groq daily cap, ~3 h), `gemini-3.6-flash`, `gemini-3.5-flash-lite` and `ministral-14b` (all rate-limited for hours after a handful of calls) were tried as writer/judge during the dry run and abandoned; only the NVIDIA route sustained volume. Their cached calls are not used. The gateway sometimes returns aliases of one model (e.g. `ministral-14b-latest` vs `-2512`); the drift check compares normalised base names and records every string seen.

## Corpus and the original data (important)
- `Research Paper/` holds 20 **templated stub PDFs of ~600 characters each** (2 chunks each) and the Chroma index (145 chunks) contains those plus `Attention Is All You Need`. The original 21-query test set was generated from that corpus. It is not a research-paper benchmark, so it is **not used** for S1-S3.
- The benchmark corpus is `uploads/` = the user's 23 real papers (2.09M characters; one is a 266-page PhD thesis = 28% of the text). Indexes are built in memory (exact cosine in numpy); `chroma_db` is untouched. Exact cosine vs Chroma HNSW is checked (`chroma_parity` in `results/s2_retrieval.json`).
- S4 (RAGAS continuity) therefore runs the 21 original queries on the legacy corpus with the production default config.
- Parsing and cleaning reuse the production code. The chunker uses production separators/size and records character offsets; `parity_check` asserts identical chunk texts to `backend.chunking.chunk_documents` at the default config (0 mismatches).

## Design decisions / assumptions
- **Conditions** (production defaults 512/128/MiniLM): C1 dense top-10; C2 = production dense-only mode (dense pool 20 -> FlashRank incl. the production source-diversity selection -> top 10); C3 = production hybrid (dense 30 + BM25 30 -> RRF k=60 -> top 20 -> FlashRank -> top 10); S = dev-selected grid cell; extra diagnostic row C2-nodiv (FlashRank order without diversity). FlashRank model: `ms-marco-TinyBERT-L-2-v2` (the library default, which production uses).
- **Dev grid**: chunk {256,512,1024} x overlap {0,10%} (26/51/102 chars) x embedder {all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, no query instruction} x pool {10,20,50} x reranker on/off = 72 cells, but pool has no effect with the reranker off, so 48 distinct cells. Selection: highest dev nDCG@5 (ties: Hit@5, then fewer chunks) on the 40 dev single-passage questions. The production default is not in the grid (overlap 128 = 25%); it is reported on dev for reference only.
- **Relevance**: a chunk is relevant to a gold span if it overlaps >= 50% of the span (contains it => 100%). Hit/Recall/MRR/nDCG are binary-relevance; nDCG's ideal list uses the number of relevant chunks in the index. Multi-passage Hit@k needs all spans; Recall@k is the fraction of spans.
- **Questions**: stratified round-robin over papers then section types (seed 20261004); passages are 450-1300 char sentence windows with filters against references, boilerplate, tables, affiliations; the writer returns question + short answer + verbatim quote; the quote must be found (whitespace/case/ligature-insensitive) inside the shown passage or the item is discarded. Also discarded: banned deictic phrases ("this paper", "the passage"...), answer already contained in the question, writer skip. Test: 100 single + 20 multi (passage pairs from different, topically close papers) + 20 unanswerable (writer-listed absent topics; kept only if none of its key terms occurs anywhere in the corpus text). Dev: 40 single (disjoint passages, hash-assigned) + the 21 legacy queries (continuity only). Candidates are processed in a fixed order until the quota is met; discards are counted in `results/s1_questions_report.json`.
- **S3 prompt**: production system prompt and context builder; only the final user instruction changed to ask for 1-3 sentences (150-token budget). Judge gets only the retrieved chunks and gold spans.
- **Metric definitions (S3)**: correct = judge "yes" (partial reported separately), answerable questions only; faithful = every claim supported by the retrieved context; abstention rate on unanswerable; hallucinated-answer rate = 1 - abstention on unanswerable questions.
- **Stats**: percentile bootstrap, 10,000 resamples over questions, seed fixed; paired bootstrap for differences. Absolute = metric points; relative = (A-B)/B, always labelled. No p-values.
- Langfuse tracing is only used in S3 (tags `cond:C1/C2/S`, `type:*`; `dry-run` for dry runs).
- Raw outputs and caches are gitignored (`eval/cache`, `eval/state.json`, `results/raw`, `results/dry`); summaries and `eval/questions.jsonl` are committed.

## What the full run taught us (also in the paste-back)
- **Nemotron 502s were partly length cut-offs**: in JSON mode the gateway answers `502 format_ignored: truncated JSON` when reasoning tokens eat `max_tokens`, instead of `finish_reason=length`. The client now retries once with 2x `max_tokens` and counts it as a truncation. Before that fix, 12 S1 candidates (3 test-single, 8 test-multi, 1 dev) were skipped as `llm_error`; the question set was kept frozen (not regenerated) and the skips are listed in `results/s1_questions_report.json`. S3/S5 items lost to the same error were re-run to completion (all conditions have 60 answerable + 20 unanswerable items).
- **Model alias**: some Nemotron calls were served as `nvidia/nemotron-3-super-120b-a12b:free` (a different host of the same model) instead of `nvidia/nemotron-3-super-120b-a12b`. Both strings are recorded in `eval/cache/frozen_models.json`; the host (and possibly numeric precision) is not controlled.
- **Exact vs Chroma**: all S2 numbers use exact cosine search. With Chroma's default HNSW settings C1 loses ~5 points of Hit@5 and ~7 of Hit@10 (`results/s2_chroma_hit_check.json`), so the deployed dense baseline is somewhat weaker than reported and C2-vs-C1 gaps against the deployed system would be slightly larger.
- RAGAS silently swallowed job errors (and once hung); S4 now re-raises quota/drift errors, retries failed jobs from the cache, and dumps a stack trace after 20 min.
- Run history: the first full run lost ~950 S1 items to a duplicate-pacer bug (no HTTP calls were made; those rows were removed from the ledger) and was restarted; S1 was resumed twice with a faster retry policy.

## Known limitations
- LLM-written questions share vocabulary with their gold passages (favours lexical/BM25 signals); only the verbatim-quote check and the human check guard validity.
- Writer and judge are the same model (disclosed); the judge differs from the answer generator.
- The dev set has 40 questions, so configuration selection is noisy; CIs are reported and the selected config is not claimed to be significantly better.
- Chunk-size comparisons are confounded by the amount of context per chunk and by MiniLM's 256-token limit at 1024 chars.
- The PhD thesis is 28% of the text but gets 1/23 of the quota. PDF ligatures ("veriﬁcation") are left as production leaves them.
- Free-tier gateway: routing across providers is invisible except via `X-Routed-Via`; answers are 150 tokens max.
- The human check (S6) is a template until you fill it.
