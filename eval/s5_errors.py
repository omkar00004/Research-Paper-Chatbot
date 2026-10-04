"""S5: error analysis of C2 retrieval failures (no gold span in the top 5). One cached judge call per failed question (cap 40)."""
import json, random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from eval.common import JUDGE_MODEL, RESULTS, SEED, rawdir, read_json, write_json
from eval import llm as L, corpus as C, s2_retrieval as R
from eval.s1_questions import words

CATS = {"chunk_boundary_split": "the gold evidence is cut across chunk boundaries so no single chunk holds most of it",
        "vocabulary_mismatch": "the gold chunk exists but the question's wording differs from it (synonyms, paraphrase, abbreviations), so it was not retrieved into the candidate pool",
        "reranker_demoted": "the gold chunk WAS in the candidate pool but the re-ranker (or its source-diversity selection) pushed it out of the top 5",
        "pdf_extraction_problem": "the gold text is garbled or altered by PDF extraction (ligatures, broken words, table/column interleaving, math), hurting matching",
        "multi_hop": "the question needs several passages / reasoning across passages, and retrieval got only the wrong ones",
        "other": "none of the above"}
PROMPT = """Classify why a retrieval system failed to bring the gold evidence into its top-5 results.

Question: {q}
Question type: {typ}
Gold evidence (verbatim): {quote}
Gold chunk text as indexed (may differ from the quote because of extraction): {gold_chunk}

Top-3 retrieved chunks:
{top3}

Pipeline facts (computed programmatically):
{facts}

Categories:
{cats}

Pick the single most likely category. Reply ONLY with JSON: {{"category": "<one of {names}>", "rationale": "<one line>"}}"""


def run(llm, state, limit=None):
    L.set_stage("S5")
    raw = rawdir(limit) / "s2_C2.jsonl"
    if not raw.exists(): raise SystemExit("Run s2 first.")
    c2 = [json.loads(l) for l in raw.read_text().splitlines()]
    qs = {q["id"]: q for q in R.load_questions(limit)}
    fails = [r for r in c2 if r["metrics"]["recall@5"] == 0]
    total = len(c2); random.Random(SEED).shuffle(fails); sel = fails[:40]
    docs = C.load_docs(); eng = R.engine(docs, "main", 512, 128, "all-MiniLM-L6-v2")
    print(f"S5: {len(fails)} of {total} answerable test questions have no gold span in the C2 top 5; analysing {len(sel)} (cap 40)")
    rows = []
    def one(r):
        q = qs[r["qid"]]; spans = q["gold_spans"]
        qv = R.embed("all-MiniLM-L6-v2", [q["question"]])[0]; s = eng.E @ qv; order = np.argsort(-s); rank_of = np.empty(len(s), int); rank_of[order] = np.arange(1, len(s) + 1)
        gold_idx = [i for i, c in enumerate(eng.chunks) if any(R.matches(c, sp) for sp in spans)]
        inter = [i for i, c in enumerate(eng.chunks) if any(c["doc"] == sp["doc"] and min(c["end"], sp["end"]) > max(c["start"], sp["start"]) for sp in spans)]
        best = int(min(rank_of[i] for i in gold_idx)) if gold_idx else None
        final = [k for k, cid in enumerate(r["ranked"], 1) if eng.id2i[cid] in set(gold_idx)]
        quote = " | ".join(sp["quote"] for sp in spans)
        gtxt = " || ".join(eng.chunks[i]["text"] for i in gold_idx[:2])[:900] if gold_idx else "(no chunk overlaps >=50% of the gold span)"
        odd = sum(ch in "ﬁﬂﬀﬃﬄ" for ch in quote) + len([w for w in quote.split() if len(w) > 25])
        facts = {"gold_chunks_in_index_meeting_50pct_rule": len(gold_idx), "chunks_touching_gold_span": len(inter), "best_dense_rank_of_gold_chunk_in_whole_corpus": best,
                 "gold_chunk_in_C2_candidate_pool_top20": bool(best and best <= 20), "gold_chunk_rank_in_final_top10": final[0] if final else None,
                 "ligature_or_overlong_token_count_in_gold_quote": odd, "question_words_found_in_gold_chunk": round(len(words(q["question"]) & words(gtxt)) / max(1, len(words(q["question"]))), 2)}
        top3 = "\n".join(f"[{i}] ({eng.chunks[eng.id2i[cid]]['doc']}) {eng.chunks[eng.id2i[cid]]['text'][:260]}" for i, cid in enumerate(r["ranked"][:3], 1))
        item = {"qid": q["id"], "type": q["type"], "question": q["question"], "gold_quote": quote, "facts": facts, "top3": top3}
        if state.done("s5", "C2", q["id"]): return {**item, **state.get("s5", "C2", q["id"])["out"]}
        try:
            j, _ = llm.json_chat(JUDGE_MODEL, [{"role": "user", "content": PROMPT.format(q=q["question"], typ=q["type"], quote=quote, gold_chunk=gtxt, top3=top3, facts=json.dumps(facts, indent=1),
                                                                                           cats="\n".join(f"- {k}: {v}" for k, v in CATS.items()), names="|".join(CATS))}], max_tokens=700)
        except (L.QuotaExit, L.BudgetExceeded, L.ModelDrift): raise
        except Exception as e:
            L.Ledger.log("failed", model=JUDGE_MODEL, why=str(e)[:150]); return None
        cat = str(j.get("category", "other")); cat = cat if cat in CATS else "other"
        out = {"category": cat, "rationale": str(j.get("rationale", ""))[:300]}
        state.mark("s5", "C2", q["id"], "done", out); return {**item, **out}
    with ThreadPoolExecutor(4) as ex: rows = [x for x in ex.map(one, sel) if x]
    state.flush()
    counts = Counter(r["category"] for r in rows)
    rule = Counter("reranker_demoted(rule)" if r["facts"]["gold_chunk_in_C2_candidate_pool_top20"] else "not_in_pool(rule)" for r in rows)
    write_json(RESULTS / ("s5_errors.json" if not limit else "dry/s5_errors.json"), {"judge": JUDGE_MODEL, "n_answerable_test": total, "n_failures_c2_top5": len(fails), "n_analysed": len(rows), "cap": 40,
                                                                                    "counts": dict(counts), "programmatic_pool_facts": dict(rule), "items": rows})
    print("  counts:", dict(counts), "| rule-based pool facts:", dict(rule))
